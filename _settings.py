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
    "finalize_on": "session_end",
    "suspend_after_s": "0",
    "cache_pin_end": "1",
    "cache_ttl_s": "auto",
    "cache_task_end": "0",
    "next_action_map": "",
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


def _fold_priority_level(level):
    """把任意配置值折叠进二值域 {1, 3}。

    消费侧只拿 ``priority.level`` 与一条准入线比较，多于两档的取值没有对应语义，
    多出来的档位只会让调用方以为存在它们又拿不到的区分。语义丰富度由 ``reason`` 承担。

    ``level >= 3`` 是"有人正在等"的交互档，其余一律背景档；v1 遗留的 2/4 折叠后
    落在它们原本就在的那一侧（2 → 1、4 → 3），越界值同理。
    """
    return 3 if level >= 3 else 1


def priority():
    """priority 段；level<=0 或空 -> None（整段省略）。"""
    level = _int(_get("priority_level"), 0)
    if level <= 0:
        return None
    reason = str(_get("priority_reason") or "").strip() or "user_waiting"
    return {"level": _fold_priority_level(level), "reason": reason}


def status_sink():
    raw = str(_get("status_sink") or "").strip().lower()
    if raw in ("ndjson", "http", "none"):
        return raw
    return "http" if status_url() else "none"


def status_url():
    return str(_get("status_url") or "").strip()


def finalize_on():
    v = str(_get("finalize_on") or "session_end").strip().lower()
    return v if v in ("session_end", "session_finalize") else "session_end"


def suspend_after_s():
    return max(0, _int(_get("suspend_after_s"), 0))


def cache_pin_end():
    return max(0, _int(_get("cache_pin_end"), 1))


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
