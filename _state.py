"""每会话进程内状态（唯一可变状态，全程加锁）。

结构：session_id -> {call_index, last_api_request_id, epoch, state, state_since_ms,
inflight:set, parent_session_id, child_role}，OrderedDict + 上限 256（LRU 淘汰）。
子 Agent 父映射另存 child_session_id -> (parent_session_id, child_role)。
"""
import threading
import time
from collections import OrderedDict

_MAX_SESSIONS = 256

_lock = threading.Lock()
_sessions = OrderedDict()
_subagent = {}  # child_session_id -> (parent_session_id, child_role)
_watchdogs = {}  # session_id -> threading.Timer
_epoch_provider = None  # 可插接缝：callable(session_id) -> epoch int；None = 内部计数器（T4）


def _now_ms():
    return int(time.time() * 1000)


def _new():
    return {
        "call_index": 0,
        "last_api_request_id": "",
        "epoch": 0,
        "state": "running",
        "state_since_ms": _now_ms(),
        "inflight": set(),
        "parent_session_id": None,
        "child_role": None,
        # 上一次请求的消息条数：条数回落即判定发生了一次上下文压缩（见 note_message_count）。
        "last_message_count": None,
    }


def _reset():
    """测试/回收辅助：清空全部进程内状态。"""
    global _epoch_provider
    with _lock:
        _sessions.clear()
        _subagent.clear()
        _epoch_provider = None
        for t in list(_watchdogs.values()):
            t.cancel()
        _watchdogs.clear()


def _purge_subagent_refs(sid):
    """删除 sid 作为 child 的父映射 + sid 作为 parent 的 children 映射（调用处持锁，review A2）。"""
    _subagent.pop(sid, None)
    for child in [c for c, p in _subagent.items() if p[0] == sid]:
        _subagent.pop(child, None)


def _current_epoch(sid, s):
    """epoch 接缝：provider 提供则用之，否则内部计数器（默认恒 0，T4）。
    provider 应为纯函数、不得重入 _state（调用处持锁）。"""
    if _epoch_provider is not None:
        try:
            return int(_epoch_provider(sid))
        except Exception:
            pass  # provider 异常回退内部计数器，不拖垮请求
    return s["epoch"]


def set_epoch_provider(fn):
    """注入 epoch 推导函数（T4 接缝）；传 None 恢复内部计数器。"""
    global _epoch_provider
    with _lock:
        _epoch_provider = fn


def bump_epoch(sid):
    """上下文重写/压缩-fork 时递增会话 epoch。返回新 epoch。"""
    with _lock:
        s = _sessions.get(sid)
        if s is None:
            return 0
        s["epoch"] += 1
        return s["epoch"]


def note_message_count(sid, count):
    """记录本次请求的消息条数；比上次少即判定发生过一次上下文压缩。

    Hermes 没有 compaction 事件（上游 issue #118382），压缩后请求里的消息条数会回落，
    这是插件侧唯一能观察到的信号。返回新 epoch；未压缩返回 None。
    """
    with _lock:
        s = _touch(sid)
        last = s.get("last_message_count")
        s["last_message_count"] = count
        if last is not None and count < last:
            s["epoch"] += 1
            return s["epoch"]
    return None


def _touch(sid):
    s = _sessions.get(sid)
    if s is None:
        if len(_sessions) >= _MAX_SESSIONS:
            evicted_sid, _ = _sessions.popitem(last=False)  # LRU 淘汰最旧
            _purge_subagent_refs(evicted_sid)  # 随 LRU 一并清 _subagent（review A2）
        s = _new()
        _sessions[sid] = s
    else:
        _sessions.move_to_end(sid)
    return s


def note_call(sid, api_request_id=""):
    """登记一次（可能重试的）LLM 请求。返回 (call_index, state, since_ms, epoch, resumed)。

    - 同 api_request_id 重复 -> call_index 不涨（重试/回退不虚增，R2）。
    - resumed 恒为 False：finalize 是会话级终态（on_session_finalize 后会话即被
      drop，不存在 finalize 之后再发请求的序列），旧的 M4 翻转路径已随会话级
      语义一并删除。
    """
    with _lock:
        s = _touch(sid)
        if api_request_id and api_request_id == s["last_api_request_id"]:
            pass  # 同一次逻辑调用重试，call_index 不涨
        else:
            s["call_index"] += 1
            s["last_api_request_id"] = api_request_id or ""
        return (s["call_index"], s["state"], s["state_since_ms"], _current_epoch(sid, s), False)


def set_state(sid, state):
    """状态切换；仅变化时更新 state_since_ms。返回当前 state_since_ms。"""
    with _lock:
        s = _touch(sid)
        if s["state"] != state:
            s["state"] = state
            s["state_since_ms"] = _now_ms()
        return s["state_since_ms"]


def tool_started(sid, call_id):
    """工具在飞 +1；从非 tool 切入时更新 since。返回 since_ms。"""
    with _lock:
        s = _touch(sid)
        if call_id:
            s["inflight"].add(call_id)
        if s["state"] != "tool":
            s["state"] = "tool"
            s["state_since_ms"] = _now_ms()
        return s["state_since_ms"]


def tool_finished(sid, call_id):
    """工具在飞 -1；集合清空才回 running（R3，并行工具批安全）。返回新 since_ms 或 None。"""
    with _lock:
        s = _touch(sid)
        if call_id:
            s["inflight"].discard(call_id)
        if s["inflight"]:
            return None
        if s["state"] != "running":
            s["state"] = "running"
            s["state_since_ms"] = _now_ms()
        return s["state_since_ms"]


def clear_inflight(sid):
    """turn 结束兜底：清在飞集合（防丢 post 导致卡 tool）。"""
    with _lock:
        s = _sessions.get(sid)
        if s is not None:
            s["inflight"].clear()


def drop(sid):
    """会话真关闭回收（on_session_finalize）。"""
    with _lock:
        _sessions.pop(sid, None)
        _purge_subagent_refs(sid)  # 清 sid 自身父映射 + 其 children 映射（review A2）
        t = _watchdogs.pop(sid, None)
        if t is not None:
            t.cancel()


def agent_type(sid, default):
    """子 Agent 用 child_role 覆盖 agent_type；否则用配置默认。"""
    with _lock:
        entry = _subagent.get(sid)
        return entry[1] if entry and entry[1] else default


def lineage(sid):
    """返回 (parent_session_id, derivation)。

    优先级：原生 ``subagent_start`` 记录（spawn）> 部署方声明（fork 或 spawn）> 根会话 None。
    Hermes 没有 fork 事件，所以一条 fork 血缘只能由创建该会话的一方声明出来。
    """
    with _lock:
        entry = _subagent.get(sid)
    if entry and entry[0]:
        return (entry[0], {"relation": "spawn", "join": "sync"})
    from . import _settings  # 局部导入：_settings 不依赖本模块，避免加载期循环
    declared = _settings.lineage()
    if declared:
        derivation = {"relation": declared["relation"]}
        if declared["relation"] == "spawn":
            derivation["join"] = declared.get("join", "sync")
        elif "fork_point" in declared:
            derivation["fork_point"] = declared["fork_point"]
        return (declared["parent_session_id"], derivation)
    return (None, None)


def subagent_start(parent_session_id, child_session_id, child_role):
    """记录子 Agent 父映射；child_session_id 为空 -> 防御跳过（M6）。"""
    if not child_session_id:
        return
    with _lock:
        _subagent[child_session_id] = (parent_session_id, child_role)
        _touch(child_session_id)


def subagent_stop(child_session_id):
    """清理子 Agent 父映射。"""
    if not child_session_id:
        return
    with _lock:
        _subagent.pop(child_session_id, None)


def start_suspend_watchdog(sid, seconds):
    """审批等待超阈 -> suspend（默认关；seconds<=0 不启用，R16）。"""
    if seconds <= 0:
        return

    def _fire():
        with _lock:
            s = _sessions.get(sid)
            if s is None or s["state"] != "pause":
                return
            s["state"] = "suspend"
            s["state_since_ms"] = _now_ms()
            since = s["state_since_ms"]
        # 锁外发事件，避免死锁（延迟导入避免模块加载期循环依赖）
        try:
            from . import _builder, _sink
            _sink.emit(_builder.status_event(sid, "suspend", since_ms=since, reason="checkpoint"))
        except Exception:
            pass

    with _lock:
        t = _watchdogs.pop(sid, None)
        if t is not None:
            t.cancel()
        t = threading.Timer(seconds, _fire)
        t.daemon = True
        _watchdogs[sid] = t
    t.start()


def cancel_suspend_watchdog(sid):
    with _lock:
        t = _watchdogs.pop(sid, None)
    if t is not None:
        t.cancel()
