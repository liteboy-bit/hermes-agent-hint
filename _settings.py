"""配置面（进程级快照）：env 与 plugins.entries.agent-hint.settings 二源合一。

优先级 settings > env > 缺省；register(ctx) 时 bind(ctx) 一次，之后全部读内存快照，
避免在热路径（每次 API 请求 / hook）重复走 ctx.get_config 的磁盘读。
"""
import json
import os

_ENV_PREFIX = "AGENT_HINT_"

_DEFAULTS = {
    "enabled": "0",
    "agent_type": "hermes",
    "priority_level": "3",
    "priority_reason": "user_waiting",
    "status_sink": "",
    "status_url": "",
    "suspend_after_s": "0",
    "cache_pin_end": "auto",
    "cache_ttl_s": "auto",
    "cache_task_end": "0",
    "next_action_map": "",
    # 部署方声明的会话血缘（JSON）。Hermes 原生只有 spawn（subagent_start），
    # fork 只能由创建该会话的一方声明。
    "lineage": "",
}

_ctx = None
_snapshot = None


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on", "enabled")


def _int(v, default=0):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def bind(ctx):
    """register(ctx) 时调用一次；随后 load 全量快照。"""
    global _ctx
    _ctx = ctx


def _load():
    global _snapshot
    out = {}
    for key, default in _DEFAULTS.items():
        val = None
        if _ctx is not None:
            try:
                val = _ctx.get_config(key, None)
            except Exception:
                val = None
        if val is None:
            val = os.environ.get(_ENV_PREFIX + key.upper(), None)
        if val is None:
            val = default
        out[key] = val
    _snapshot = out
    return out


def _reset():
    """测试/回收辅助：清空快照以强制下次读取重新加载。"""
    global _snapshot
    _snapshot = None


def _get(key):
    if _snapshot is None:
        _load()
    return _snapshot.get(key, _DEFAULTS.get(key))


def enabled():
    return _truthy(_get("enabled"))


def agent_type():
    return str(_get("agent_type") or "hermes")


# RFC §3.3.1 的 reason 词表：出口必须落在这三档内，词表外取值在消费侧没有语义。
_RFC_REASONS = ("user_waiting", "critical_path", "background")


def _fold_priority_level(level):
    """把任意配置值折叠进二值域 {1, 3}。

    消费侧只拿 ``priority.level`` 与一条准入线比较，多于两档的取值没有对应语义，
    多出来的档位只会让调用方以为存在它们又拿不到的区分。语义丰富度由 ``reason`` 承担。

    ``level >= 3`` 是"有人正在等"的交互档，其余一律背景档；v1 遗留的 2/4 折叠后
    落在它们原本就在的那一侧（2 → 1、4 → 3），越界值同理。
    """
    return 3 if level >= 3 else 1


def _fold_priority_reason(reason, level):
    """把 reason 折叠进 RFC 词表 {user_waiting, critical_path, background}。

    词表外的值（宿主自造的 ``continuation: tool_followup`` 一类）不是协议语义：
    消费方拿 reason 做准入/排障判断时只会失配。冒号前缀命中词表的取前缀
    （``background: context_compaction`` → ``background``）；完全词表外的按档位
    兜底——免排队档必然有人在等（``user_waiting``），背景档一律 ``background``。
    兜底是防御性折叠，不是语义猜测：发出去的必须是消费方能读懂的词。
    """
    text = str(reason or "").strip()
    if not text:
        return "user_waiting" if level >= 3 else "background"
    head = text.split(":", 1)[0].strip().lower()
    if head in _RFC_REASONS:
        return head
    return "user_waiting" if level >= 3 else "background"


def priority():
    """priority 段；level<=0 或空 -> None（整段省略）。"""
    level = _int(_get("priority_level"), 0)
    if level <= 0:
        return None
    level = _fold_priority_level(level)
    reason = _fold_priority_reason(_get("priority_reason"), level)
    return {"level": level, "reason": reason}


def status_sink():
    raw = str(_get("status_sink") or "").strip().lower()
    if raw in ("ndjson", "http", "none"):
        return raw
    return "http" if status_url() else "none"


def status_url():
    return str(_get("status_url") or "").strip()


def suspend_after_s():
    return max(0, _int(_get("suspend_after_s"), 0))


def cache_pin_end():
    """断点位置：``auto``（默认）= 当前逻辑输入末尾，``0`` = 不发，正整数 = 显式下标。

    历史默认是 1（只护住 system 那一条）。``pin`` 退役后这个默认已经错位：单条声明
    保护的是"当前前缀"，而 system 之外的部分同样是下一轮的前缀。改成 ``auto`` 后
    与 Pi 一致，断点落在整段逻辑输入末尾；显式配置仍然生效，便于定位问题。
    """
    raw = str(_get("cache_pin_end") if _get("cache_pin_end") is not None else "auto").strip().lower()
    if raw in ("", "auto"):
        return None
    return max(0, _int(raw, 0))


def cache_ttl_s():
    """ephemeral 保留秒数：None=按运行时信号推导（默认），0=只发 pin，正整数=固定值。

    固定值按 Motor 的钳制口径校验：低于 60 无效、超过 3600 超协议上限；两者都按 0 处理
    ——宁可不声明，也不发一个会被消费侧静默改写的值。
    """
    raw = str(_get("cache_ttl_s") or "").strip().lower()
    if raw in ("", "auto", "none", "null"):
        return None
    ttl = _int(raw, 0)
    if ttl <= 0:
        return 0
    if ttl < 60 or ttl > 3600:
        return 0
    return ttl


def cache_task_end():
    """第二条断点（任务说明末尾）的消息下标；0（缺省）= 不产出第二条断点。"""
    return max(0, _int(_get("cache_task_end"), 0))


def next_action_map():
    """prediction.next_action 的**通用**映射（JSON 字符串，默认为空）。

    形如 ``{"<最近工具名前缀>": ["<下一步工具名>", <剩余调用数>, "<state>"], "__default__": [...]}``；
    键按长度降序匹配（最长前缀优先），``__default__`` 为兜底键。
    空值/非法 JSON -> ``{}``，此时整段省略 next_action/remaining_calls（不猜）。
    该字段是启发式估计，不是埋点统计量：网关不应把它当高置信信号使用。
    """
    raw = str(_get("next_action_map") or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def lineage():
    """部署方声明的会话血缘；未声明 -> None。

    Hermes 的原生信号只有 ``subagent_start``（= spawn）。要上报 ``fork``（会话分支），
    只能由创建该会话的一方声明：``{"relation": "fork", "parent_session_id": "...",
    "fork_point": N}``。非法值一律忽略（宁可不报，也不报一个错的祖先）。
    """
    raw = str(_get("lineage") or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    relation = data.get("relation")
    parent = data.get("parent_session_id")
    if relation not in ("fork", "spawn") or not isinstance(parent, str) or not parent:
        return None
    line = {"relation": relation, "parent_session_id": parent}
    if relation == "spawn":
        join = data.get("join")
        line["join"] = join if join in ("sync", "async") else "sync"
    else:
        point = data.get("fork_point")
        if isinstance(point, int) and not isinstance(point, bool) and point >= 0:
            line["fork_point"] = point
    return line
