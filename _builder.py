"""AgentHint 对象构造（纯函数，无副作用）。

字段语义见 README 的 “The `X-Agent-Hint` header” 小节：
- 根会话不写 parent_session_id / derivation（缺省）。
- 缺省项整段省略（priority / cache_control / prediction / reason / tools）。
- strict_priority 明确不写：由可信控制面注入，普通 Agent 永不主动设置（N1）。
"""

# 引擎按块（128 token）缓存并整块哈希：不足一个块的前缀根本不会被存下来，
# 为它声明保留等于要求引擎保留一个不存在的东西。
MIN_CACHEABLE_PREFIX_TOKENS = 128

# 内部状态 → 状态事件词表（简化字段 RFC §3.3.2 / §3.5.2.3）。
EVENT_STATE = {"tool": "pause", "suspend": "pause", "resume": "running"}

_CJK_RANGES = ((0x2E80, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))
_CHARS_PER_TOKEN = 4


def request_hint(session_id, call_index, epoch, agent_type, state, state_since_ms,
                 parent_session_id=None, derivation=None,
                 priority=None, cache_control=None, prediction=None):
    """构造随 LLM 请求下发的完整 AgentHint（缺省项省略）。

    ``agent_status`` 只带 ``state`` 与 ``state_since_ms``：``reason``/``tools`` 是排障细节，
    不占 header 预算。按简化字段 RFC §3.3.2 折叠表，``tool`` 与 ``suspend`` 在消费侧都是
    "在等外部事件"（工具返回 / 人等审批），出口一律折成 ``pause``；``resume``（人放行了）
    是真正需要的边，只在请求头里出现一次。
    """
    session = {
        "session_id": session_id,
        "agent_type": agent_type,
        "call_index": call_index,
        "epoch": epoch,
    }
    if parent_session_id:
        session["parent_session_id"] = parent_session_id
    if derivation:
        session["derivation"] = derivation
    wire_state = "pause" if state in ("suspend", "tool") else state
    hint = {
        "session": session,
        "agent_status": {"state": wire_state, "state_since_ms": state_since_ms},
    }
    if priority:
        hint["priority"] = priority
    if cache_control:
        hint["cache_control"] = cache_control
    if prediction:
        hint["prediction"] = _fold_next_action(prediction)
    return hint


def _fold_next_action(prediction):
    """把 prediction.next_action.state 折进对外词汇表（tool/suspend → running）。

    `next_action` 由 ``next_action_map`` 配置驱动，配置里可能写 ``tool``；但对外状态只承认
    running/pause/resume/finalize，所以这里统一折叠，具体要调什么由 ``next_action.tools`` 表达。
    """
    action = prediction.get("next_action") if isinstance(prediction, dict) else None
    if not isinstance(action, dict):
        return prediction
    state = action.get("state")
    if state in ("tool", "suspend"):
        folded = dict(prediction)
        folded["next_action"] = {**action, "state": "running"}
        return folded
    return prediction


def _estimate_tokens(value):
    """估算消息的 token 数（无 tokenizer）。

    CJK 按字计 1 token（所服务的 tokenizer 大致如此），其余按 4 字符 1 token。
    只用于判断前缀能否占满一个块，绝不用于推导 TTL。
    """
    if isinstance(value, str):
        cjk = other = 0
        for ch in value:
            code = ord(ch)
            if any(lo <= code <= hi for lo, hi in _CJK_RANGES):
                cjk += 1
            else:
                other += 1
        return cjk + -(-other // _CHARS_PER_TOKEN)
    if isinstance(value, (list, tuple)):
        return sum(_estimate_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(_estimate_tokens(item) for item in value.values())
    return 0


def cache_control(request, end=1, ttl_s=0, task_end=None):
    """会话前缀的保留声明：**单一断点 + 显式 TTL**。

    只有一条断点，`pin` 已退役：
    - 消费侧从不区分 ``type``（只记日志），而 ``pin`` 这个名字暗示一个硬保证，
      没有任何一层能兑现；Dynamo 已移除等价字段，vLLM RFC 也已放弃 Pin。
    - 第二条断点在消费侧被折叠成"最宽 end + 最长显式 TTL"，不产生额外行为，
      只会多占 header 预算。

    断点位置：``task_end`` 合法且比 ``end`` 更宽时取它（保护到任务说明末尾），
    否则取 ``end``（稳定前缀末尾）。当前轮对话永远不打断点。

    ``ttl_s <= 0`` 时不产出任何条目——没有 TTL 就没有可声明的保留。
    前缀不足一个块（``MIN_CACHEABLE_PREFIX_TOKENS``）时同样不产出。
    ``request`` 含合法 messages 才产出，否则 None（畸形输入绝不抛错）。
    """
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list) or not messages or end <= 0:
        return None
    if end > len(messages):
        # 断点越界：消费侧会整条丢弃 range，这里宁可不发也不发一个错位的声明。
        return None
    if ttl_s <= 0:
        return None
    if _estimate_tokens(messages) < MIN_CACHEABLE_PREFIX_TOKENS:
        return None

    target = end
    if isinstance(task_end, int) and end < task_end <= len(messages):
        target = task_end
    return [{
        "type": "ephemeral",
        "ttl_s": ttl_s,
        "range": {"end": target, "unit": "message"},
    }]


def status_event(session_id, state, since_ms, reason=None, tools=None):
    """构造状态事件（异步上报）。session 只带 session_id（对齐字段文档 §4 示例）。

    事件通道词表只有 ``running | pause | finalize``（简化字段 RFC §3.3.2 / §3.5.2.3）：
    ``tool`` / ``suspend`` 折成 ``pause``，``resume`` 这种一次性边回落成 ``running``。
    """
    event = {
        "session": {"session_id": session_id},
        "agent_status": {"state": EVENT_STATE.get(state, state),
                         "state_since_ms": since_ms},
    }
    if reason:
        event["agent_status"]["reason"] = reason
    if tools:
        event["agent_status"]["tools"] = tools
    return event


TTL_MIN_S = 60
TTL_MAX_S = 3600
TTL_SAFETY_FACTOR = 2
TTL_FALLBACK_GAP_S = 300


def estimate_ttl_s(tool_expected_duration_s=None, interval_ms=None):
    """按运行时信号估算保留时长（交接文档 §3.3）。

    优先级：下一次工具调用的预计时长 > 最近一次请求间隔 > 协议默认 300s。

    乘以安全系数后落在协议窗口 ``[TTL_MIN_S, TTL_MAX_S]`` 内：低于下限抬到 60s，
    高于上限**截断到 3600s**（实测结论，P3 2026-10-21：保留生效的充要条件是
    ``ttl_s > 空闲间隔``；不声明时 600s 间隔命中 0%，声明 3600s 命中 99.8%。
    因此截断严格优于丢弃——丢弃等于完全放弃保护）。
    """
    gap = None
    if isinstance(tool_expected_duration_s, (int, float)) and tool_expected_duration_s > 0:
        gap = float(tool_expected_duration_s)
    elif isinstance(interval_ms, (int, float)) and interval_ms > 0:
        gap = interval_ms / 1000.0
    else:
        gap = TTL_FALLBACK_GAP_S

    raw = gap * TTL_SAFETY_FACTOR
    return max(TTL_MIN_S, min(TTL_MAX_S, int(round(raw))))
