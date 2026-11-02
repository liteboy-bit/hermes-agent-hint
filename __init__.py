"""agent-hint 插件入口：注册 llm_request 中间件 + 10 个 hooks。

零侵入：只依赖 ctx（register_hook / register_middleware / get_config）与标准库，
不 import 任何 hermes 内部模块。关闭态（enabled=0，默认）中间件与 hooks 均在入口早退（us 级）。
"""
from . import _builder, _codec, _settings, _sink, _state, _stats


def register(ctx):
    _settings.bind(ctx)  # load 时快照全部 settings（env + plugins.entries.agent-hint.settings）

    for name, cb in (
        ("on_session_start", _on_session_start),
        ("on_session_end", _on_session_end),
        ("on_session_finalize", _on_session_finalize),
        ("pre_tool_call", _on_pre_tool_call),
        ("post_tool_call", _on_post_tool_call),
        ("pre_approval_request", _on_pre_approval_request),
        ("post_approval_response", _on_post_approval_response),
        ("subagent_start", _on_subagent_start),
        ("subagent_stop", _on_subagent_stop),
        ("post_api_request", _on_post_api_request),
    ):
        ctx.register_hook(name, cb)

    # 中间件常驻注册（关闭态在回调内早退，us 级）：manifest 声明的能力必须与真实注册一致，
    # 否则上游能力探针（hermes plugins validate）会判为「声明与现实不符」。
    ctx.register_middleware("llm_request", _on_llm_request)


# --- 工具函数 ---

def _sid(kw):
    """会话标识：多数 hook 传 ``session_id``；审批类 hook 在调用点传 ``session_key``，
    并在 observability context 已绑定时由 dispatcher 额外补上 ``session_id``
    （见上游 ``tools/approval_context.py::_fire_approval_hook``；未绑定时该键不存在）。
    两者都认，取到哪个用哪个。"""
    return str(kw.get("session_id") or kw.get("session_key") or "")


def _on():
    return _settings.enabled()


# --- hooks ---

def _on_session_start(**kw):
    if not _on():
        return
    sid = _sid(kw)
    if not sid:
        return
    since = _state.set_state(sid, "running")
    _sink.emit(_builder.status_event(sid, "running", since_ms=since))


def _on_session_end(**kw):
    if not _on():
        return
    sid = _sid(kw)
    if not sid:
        return
    _state.clear_inflight(sid)  # turn 结束清在飞兜底
    if _settings.finalize_on() == "session_end":
        reason = None if kw.get("completed") else str(kw.get("turn_exit_reason") or "interrupted")
        since = _state.set_state(sid, "finalize")
        _sink.emit(_builder.status_event(sid, "finalize", since_ms=since, reason=reason))


def _on_session_finalize(**kw):
    if not _on():
        return
    sid = _sid(kw)
    if not sid:
        return
    if _settings.finalize_on() == "session_finalize":
        since = _state.set_state(sid, "finalize")
        _sink.emit(_builder.status_event(sid, "finalize", since_ms=since))
    _state.drop(sid)  # 会话真关闭回收（接缝）


def _on_pre_tool_call(**kw):
    if not _on():
        return None
    sid = _sid(kw)
    if not sid:
        return None
    name = str(kw.get("tool_name") or "")
    call_id = str(kw.get("tool_call_id") or "")
    _stats.note_tool(sid, name)  # prediction 启发式：最近工具名
    since = _state.tool_started(sid, call_id)
    _sink.emit(_builder.status_event(sid, "tool", since_ms=since,
                                     reason="await_tool",
                                     tools=[{"name": name, "call_id": call_id}]))
    return None  # 纯观察，绝不返回 block/approve/modify


def _on_post_tool_call(**kw):
    if not _on():
        return
    sid = _sid(kw)
    if not sid:
        return
    _stats.note_tool_done(sid)  # prediction：结算工具耗时，供 TTL 估计
    since = _state.tool_finished(sid, str(kw.get("tool_call_id") or ""))
    if since is not None:  # 集合清空 -> 回 running（R3）
        _sink.emit(_builder.status_event(sid, "running", since_ms=since))


def _on_pre_approval_request(**kw):
    if not _on():
        return
    sid = _sid(kw)
    if not sid:
        return
    since = _state.set_state(sid, "pause")
    _sink.emit(_builder.status_event(sid, "pause", since_ms=since, reason="human_approval"))
    _state.start_suspend_watchdog(sid, _settings.suspend_after_s())


def _on_post_approval_response(**kw):
    if not _on():
        return
    sid = _sid(kw)
    if not sid:
        return
    _state.cancel_suspend_watchdog(sid)
    since = _state.set_state(sid, "running")
    _sink.emit(_builder.status_event(sid, "running", since_ms=since))


def _on_subagent_start(**kw):
    if not _on():
        return
    parent_sid = str(kw.get("parent_session_id") or "")
    child_sid = str(kw.get("child_session_id") or "")
    child_role = str(kw.get("child_role") or "")
    _state.subagent_start(parent_sid, child_sid, child_role)


def _on_subagent_stop(**kw):
    if not _on():
        return
    child_sid = str(kw.get("child_session_id") or "")
    _state.subagent_stop(child_sid)


def _on_post_api_request(**kw):
    if not _on():
        return
    _stats.collect(**kw)


# --- middleware ---

def _on_llm_request(**kw):
    if not _on():
        return None
    req = kw.get("request")
    if not isinstance(req, dict):
        return None
    sid = _sid(kw)
    if not sid:
        return None
    try:
        call_index, state, since_ms, epoch, resumed = _state.note_call(
            sid, str(kw.get("api_request_id") or ""))
        if resumed:  # M4：finalize -> running 边，补发 running 事件
            _sink.emit(_builder.status_event(sid, "running", since_ms=since_ms))
        parent_session_id, derivation = _state.lineage(sid)
        # 一次快照同时喂 prediction 与 TTL 推导，避免重复统计。
        prediction = _stats.snapshot(sid, next_action_map=_settings.next_action_map())
        ttl_s = _settings.cache_ttl_s()
        if ttl_s is None:
            observed = prediction or {}
            next_action = observed.get("next_action") or {}
            tools = next_action.get("tools") or []
            ttl_s = _builder.estimate_ttl_s(
                tool_expected_duration_s=tools[0].get("expected_duration_s") if tools else None,
                interval_ms=observed.get("interval_ms"),
            )
        hint = _builder.request_hint(
            session_id=sid, call_index=call_index, epoch=epoch,
            agent_type=_state.agent_type(sid, _settings.agent_type()),
            parent_session_id=parent_session_id, derivation=derivation,
            state=state, state_since_ms=since_ms,
            priority=_settings.priority(),
            cache_control=_builder.cache_control(
                req,
                # None = auto：断点落在当前逻辑输入末尾（与 Pi 一致）。显式配置仍然生效。
                end=_settings.cache_pin_end() if _settings.cache_pin_end() is not None
                else len(req.get("messages") or []),
                ttl_s=ttl_s or 0,
                task_end=_settings.cache_task_end() or None,
            ),
            prediction=prediction,
        )
        req = dict(req)  # 浅拷贝顶层（COW，绝不改入参）
        headers = dict(req.get("extra_headers") or {})
        headers["X-Agent-Hint"] = _codec.encode(hint)
        req["extra_headers"] = headers
        return {"request": req}
    except Exception as e:
        _sink._warn_once("llm_request", "hint build failed (kept request as-is): " + str(e))
        return None  # 保持请求原样（fail-safe，warn-once 见 stderr）
