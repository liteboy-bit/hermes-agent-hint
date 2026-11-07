"""agent-hint 插件单测（v0.5.0）。

覆盖 _codec / _builder / _state / _stats / _settings / _sink 六模块 + 中间件/hooks 契约。
装载：conftest 的 ah_plugin fixture（importlib 包方式，相对导入可用）。
"""
import json

import pytest


# ---------------- fixtures ----------------

@pytest.fixture(autouse=True)
def _reset_global(ah_plugin, monkeypatch):
    """每例前重置进程内状态 + 清缓存快照；结束后再清，避免跨例污染。"""
    ah_plugin._state._reset()
    ah_plugin._stats._reset()
    ah_plugin._settings._ctx = None
    ah_plugin._settings._reset()
    ah_plugin._sink._ndjson_path = None
    ah_plugin._sink._warned.clear()
    yield
    ah_plugin._state._reset()
    ah_plugin._stats._reset()
    ah_plugin._settings._ctx = None
    ah_plugin._settings._reset()


class _FakeCtx:
    def __init__(self, cfg):
        self._cfg = cfg

    def get_config(self, key, default=None):
        return self._cfg.get(key, default)


# ---------------- _codec ----------------

def test_codec_ascii_roundtrip(ah_plugin):
    c = ah_plugin._codec
    obj = {"session": {"session_id": "s1", "call_index": 1}}
    assert c.decode(c.encode(obj)) == obj


def test_codec_non_ascii(ah_plugin):
    c = ah_plugin._codec
    obj = {"session": {"session_id": "会话-一"}}
    enc = c.encode(obj)
    assert "=" not in enc  # 无 padding
    assert c.decode(enc) == obj


def test_codec_decode_invalid_raises(ah_plugin):
    with pytest.raises(ValueError):
        ah_plugin._codec.decode("!!not-base64!!")


# ---------------- _builder ----------------

def test_request_hint_full_fields(ah_plugin):
    b = ah_plugin._builder
    h = b.request_hint(session_id="s1", call_index=2, epoch=0, agent_type="office-agent",
                       state="running", state_since_ms=1000,
                       priority={"level": 3, "reason": "user_waiting"},
                       cache_control={"type": "pin", "range": {"end": 1}},
                       prediction={"remaining_calls": 4})
    assert h["session"] == {"session_id": "s1", "agent_type": "office-agent",
                            "call_index": 2, "epoch": 0}
    assert h["agent_status"] == {"state": "running", "state_since_ms": 1000}
    assert "strict_priority" not in h  # 控制面注入，Agent 永不写
    assert h["priority"]["level"] == 3


def test_request_hint_omits_optional(ah_plugin):
    b = ah_plugin._builder
    h = b.request_hint(session_id="s1", call_index=1, epoch=0, agent_type="office-agent",
                       state="running", state_since_ms=1)
    # 根会话：无 parent_session_id / derivation
    assert "parent_session_id" not in h["session"]
    assert "derivation" not in h["session"]
    assert "priority" not in h
    assert "cache_control" not in h
    assert "prediction" not in h


def test_request_hint_subagent(ah_plugin):
    b = ah_plugin._builder
    h = b.request_hint(session_id="s-child", call_index=1, epoch=0, agent_type="leaf",
                       state="running", state_since_ms=1,
                       parent_session_id="s-parent",
                       derivation={"relation": "spawn", "join": "sync"})
    assert h["session"]["parent_session_id"] == "s-parent"
    assert h["session"]["derivation"] == {"relation": "spawn", "join": "sync"}


def test_status_event_reason_tools_optional(ah_plugin):
    b = ah_plugin._builder
    e = b.status_event("s1", "running", since_ms=5)
    assert e == {"session": {"session_id": "s1"},
                 "agent_status": {"state": "running", "state_since_ms": 5}}
    e2 = b.status_event("s1", "tool", since_ms=6, reason="await_tool",
                        tools=[{"name": "research_web_search", "call_id": "c1"}])
    assert e2["agent_status"]["reason"] == "await_tool"
    assert e2["agent_status"]["tools"][0]["call_id"] == "c1"


LONG_PREFIX = "lorem ipsum dolor sit amet " * 40  # ~1080 chars, 约 270 token，超过一个块


def test_cache_control_requires_explicit_ttl(ah_plugin):
    b = ah_plugin._builder
    req = {"messages": [{"role": "system"}, {"role": "user"}]}
    # pin 已退役：没有显式 TTL 就没有可声明的保留
    assert b.cache_control(req, end=1) is None
    assert b.cache_control({"messages": []}, end=1) is None
    assert b.cache_control({"no": "messages"}, end=1) is None
    assert b.cache_control({"messages": [{"role": "system"}]}, end=0) is None
    # 断点越界：宁可不发，也不发一个错位的声明
    assert b.cache_control(req, end=5) is None


def test_cache_control_ephemeral_carries_ttl(ah_plugin):
    b = ah_plugin._builder
    req = {"messages": [{"role": "system"}, {"role": "user", "content": LONG_PREFIX}, {"role": "assistant"}]}
    controls = b.cache_control(req, end=1, ttl_s=120)
    assert controls == [
        {"type": "ephemeral", "ttl_s": 120, "range": {"end": 1, "unit": "message"}},
    ]
    # ttl_s<=0：不发任何条目（单一断点 + 显式 TTL 是唯一形态）
    assert b.cache_control(req, end=1, ttl_s=0) is None


def test_cache_control_second_breakpoint_widens_the_ephemeral(ah_plugin):
    b = ah_plugin._builder
    req = {"messages": [{"role": f"m{i}", "content": LONG_PREFIX} for i in range(6)]}
    controls = b.cache_control(req, end=2, ttl_s=300, task_end=5)
    assert controls == [{"type": "ephemeral", "ttl_s": 300, "range": {"end": 5, "unit": "message"}}]
    # task_end 不大于 end 时不加宽（等于只是重复声明）
    assert b.cache_control(req, end=2, ttl_s=300, task_end=2)[0]["range"]["end"] == 2
    # task_end 越界时不采信
    assert b.cache_control(req, end=2, ttl_s=300, task_end=99)[0]["range"]["end"] == 2


def test_cache_ttl_distinguishes_auto_pin_only_and_fixed(ah_plugin, monkeypatch):
    s = ah_plugin._settings
    # auto（缺省 / 显式写法）-> None，表示按运行时信号推导
    for raw in ("", "auto", "none", "null"):
        monkeypatch.setenv("AGENT_HINT_CACHE_TTL_S", raw)
        s._reset()
        assert s.cache_ttl_s() is None, raw
    # 0 或越界/非法值 -> 只发 pin（不发一个会被消费侧改写的值）
    for raw in ("0", "59", "3601", "-5", "abc"):
        monkeypatch.setenv("AGENT_HINT_CACHE_TTL_S", raw)
        s._reset()
        assert s.cache_ttl_s() == 0, raw
    # 合法区间 -> 固定值
    monkeypatch.setenv("AGENT_HINT_CACHE_TTL_S", "120")
    s._reset()
    assert s.cache_ttl_s() == 120


def test_estimate_ttl_s_prefers_tool_budget_then_interval(ah_plugin):
    b = ah_plugin._builder
    assert b.estimate_ttl_s(tool_expected_duration_s=120, interval_ms=5000) == 240
    assert b.estimate_ttl_s(interval_ms=5000) == 60      # 10s -> 抬到下限 60
    assert b.estimate_ttl_s() == 600                      # 默认 gap 300 * 2
    assert b.estimate_ttl_s(tool_expected_duration_s=7200) == b.TTL_MAX_S  # 超上限 -> 截断到 3600
    assert b.estimate_ttl_s(tool_expected_duration_s=0, interval_ms=0) == 600


# ---------------- _state ----------------

def test_call_index_starts_at_one(ah_plugin):
    s = ah_plugin._state
    assert s.note_call("s1", "r1")[0] == 1


def test_call_index_dedup_same_request_id(ah_plugin):
    s = ah_plugin._state
    assert s.note_call("s1", "r1")[0] == 1
    assert s.note_call("s1", "r1")[0] == 1  # 重试不涨
    assert s.note_call("s1", "r2")[0] == 2  # 新调用涨


def test_parallel_tools_gate_running(ah_plugin):
    s = ah_plugin._state
    s.tool_started("s1", "c1")
    s.tool_started("s1", "c2")
    assert s.tool_finished("s1", "c1") is None  # 在飞未清空
    assert s.tool_finished("s1", "c2") is not None  # 清空 -> running


def test_unknown_call_id_ignored(ah_plugin):
    s = ah_plugin._state
    s.tool_started("s1", "c1")
    assert s.tool_finished("s1", "unknown") is None  # 未知 id 忽略，在飞仍非空
    assert s.tool_finished("s1", "c1") is not None


def test_state_since_ms_updates_on_change_only(ah_plugin):
    s = ah_plugin._state
    first = s.set_state("s1", "running")
    second = s.set_state("s1", "running")  # 无变化
    assert first == second
    third = s.set_state("s1", "tool")
    assert third >= first  # 变化则更新时间戳


def test_finalize_is_session_level(ah_plugin, tmp_path, monkeypatch):
    """会话级 finalize：turn 结束（on_session_end）不发 finalize，会话关闭（on_session_finalize）才发。"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    monkeypatch.setenv("AGENT_HINT_STATUS_SINK", "ndjson")
    ah_plugin._settings._reset()
    ah_plugin._sink._ndjson_path = None
    ah_plugin._on_session_start(session_id="s1")
    path = tmp_path / "agent-hint-events.ndjson"
    assert len(path.read_text().strip().split("\n")) == 1, "only the opening running event so far"
    ah_plugin._on_session_end(session_id="s1", completed=True)
    assert len(path.read_text().strip().split("\n")) == 1, "a turn end must not emit finalize"
    ah_plugin._on_session_finalize(session_id="s1")
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[-1])["agent_status"]["state"] == "finalize", \
        "only session close emits finalize, exactly once"


def test_subagent_map_and_cleanup(ah_plugin):
    s = ah_plugin._state
    s.subagent_start("s-parent", "s-child", "leaf")
    assert s.agent_type("s-child", "office-agent") == "leaf"
    assert s.lineage("s-child") == ("s-parent", {"relation": "spawn", "join": "sync"})
    s.subagent_stop("s-child")
    assert s.lineage("s-child") == (None, None)
    # 空 child_session_id 防御跳过
    s.subagent_start("s-parent", "", "leaf")
    assert s.lineage("") == (None, None)


# ---------------- _stats ----------------

def test_collect_then_snapshot(ah_plugin):
    st = ah_plugin._stats
    st.collect(session_id="s1", usage={"input_tokens": 10, "output_tokens": 20},
               api_duration=0.5, first_chunk_at=100.0, started_at=99.5, ended_at=100.5)
    snap = st.snapshot("s1")
    assert snap["output_tokens"] == 20
    assert snap["latency_ms"] == 500  # (100.0 - 99.5) * 1000
    # 未配置 next_action_map -> next_action/remaining_calls 整段省略（不猜）
    assert "next_action" not in snap
    assert "remaining_calls" not in snap


def test_usage_none_skipped(ah_plugin):
    st = ah_plugin._stats
    st.collect(session_id="s1", usage=None, api_duration=1.0)
    assert st.snapshot("s1") is None  # 空样本不产出 prediction


def test_no_samples_snapshot_none(ah_plugin):
    assert ah_plugin._stats.snapshot("s-nope") is None


def test_next_action_map_generic(ah_plugin):
    """next_action/remaining_calls 由通用映射驱动（不再硬编码任何业务 SOP）。"""
    st = ah_plugin._stats
    mapping = {
        "step_a_": ["step_b_run", 8, "tool"],
        "step_b_": [None, 0, "finalize"],
        "__default__": ["step_a_go", 10, "tool"],
    }
    st.collect(session_id="s1", usage={"output_tokens": 5}, ended_at=100.0)
    st.note_tool("s1", "step_a_go")
    snap = st.snapshot("s1", next_action_map=mapping)
    assert snap["next_action"]["tools"][0]["name"] == "step_b_run"
    assert snap["next_action"]["prob"] == 0.5
    assert snap["remaining_calls"] == 8
    st.note_tool("s1", "step_b_run")
    assert st.snapshot("s1", next_action_map=mapping)["next_action"]["state"] == "finalize"
    # 未知工具名 -> 走 __default__ 兜底
    st.note_tool("s1", "whatever_x")
    assert st.snapshot("s1", next_action_map=mapping)["next_action"]["tools"][0]["name"] == "step_a_go"
    # 无 __default__ 且不匹配 -> 整段省略
    st.note_tool("s1", "unknown_zzz")
    snap2 = st.snapshot("s1", next_action_map={"step_a_": ["step_b_run", 8, "tool"]})
    assert "next_action" not in snap2


def test_next_action_map_dict_entry_and_longest_prefix(ah_plugin):
    st = ah_plugin._stats
    st.collect(session_id="s2", usage={"output_tokens": 1}, ended_at=1.0)
    st.note_tool("s2", "web_search_long")
    snap = st.snapshot("s2", next_action_map={
        "web_": {"tool": "wrong", "remaining": 1, "state": "tool"},
        "web_search_long": {"tool": "next_tool", "remaining": 2, "state": "tool"},
    })
    assert snap["next_action"]["tools"][0]["name"] == "next_tool"
    assert snap["remaining_calls"] == 2


# ---------------- _settings ----------------

def test_settings_precedence(ah_plugin, monkeypatch):
    s = ah_plugin._settings
    monkeypatch.setenv("AGENT_HINT_AGENT_TYPE", "from-env")
    s.bind(_FakeCtx({"agent_type": "from-settings"}))
    assert s.agent_type() == "from-settings"  # settings > env


def test_settings_env_over_default(ah_plugin, monkeypatch):
    s = ah_plugin._settings
    monkeypatch.setenv("AGENT_HINT_AGENT_TYPE", "from-env")
    s._reset()
    assert s.agent_type() == "from-env"
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    s._reset()
    assert s.enabled() is True


def test_priority_zero_or_empty_omitted(ah_plugin, monkeypatch):
    s = ah_plugin._settings
    monkeypatch.setenv("AGENT_HINT_PRIORITY_LEVEL", "0")
    s._reset()
    assert s.priority() is None
    monkeypatch.setenv("AGENT_HINT_PRIORITY_LEVEL", "")
    s._reset()
    assert s.priority() is None
    monkeypatch.setenv("AGENT_HINT_PRIORITY_LEVEL", "3")
    monkeypatch.setenv("AGENT_HINT_PRIORITY_REASON", "critical_path")
    s._reset()
    assert s.priority() == {"level": 3, "reason": "critical_path"}


def test_status_sink_defaults(ah_plugin, monkeypatch):
    s = ah_plugin._settings
    monkeypatch.delenv("AGENT_HINT_STATUS_SINK", raising=False)
    monkeypatch.delenv("AGENT_HINT_STATUS_URL", raising=False)
    s._reset()
    assert s.status_sink() == "none"
    monkeypatch.setenv("AGENT_HINT_STATUS_URL", "http://x")
    s._reset()
    assert s.status_sink() == "http"
    monkeypatch.setenv("AGENT_HINT_STATUS_SINK", "ndjson")
    s._reset()
    assert s.status_sink() == "ndjson"


def test_suspend_and_cache_pin_parse(ah_plugin, monkeypatch):
    s = ah_plugin._settings
    monkeypatch.setenv("AGENT_HINT_SUSPEND_AFTER_S", "120")
    monkeypatch.setenv("AGENT_HINT_CACHE_PIN_END", "3")
    s._reset()
    assert s.suspend_after_s() == 120
    assert s.cache_pin_end() == 3
    monkeypatch.setenv("AGENT_HINT_SUSPEND_AFTER_S", "-5")
    s._reset()
    assert s.suspend_after_s() == 0  # 负数钳为 0


# ---------------- _sink ----------------

def test_ndjson_append(tmp_path, ah_plugin, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_HINT_STATUS_SINK", "ndjson")
    ah_plugin._settings._reset()
    ah_plugin._sink._ndjson_path = None
    ah_plugin._sink.emit({"session": {"session_id": "s1"},
                          "agent_status": {"state": "running", "state_since_ms": 1}})
    ah_plugin._sink.emit({"session": {"session_id": "s1"},
                          "agent_status": {"state": "finalize", "state_since_ms": 2}})
    lines = (tmp_path / "agent-hint-events.ndjson").read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["agent_status"]["state"] == "running"
    assert json.loads(lines[1])["agent_status"]["state"] == "finalize"


def test_sink_never_raises(ah_plugin, monkeypatch):
    monkeypatch.setenv("AGENT_HINT_STATUS_SINK", "ndjson")
    ah_plugin._settings._reset()
    ah_plugin._sink._ndjson_path = None
    # 强制写入失败：monkeypatch open 抛 IOError -> emit 吞掉不抛
    import builtins
    real_open = builtins.open

    def _boom(*a, **k):
        raise IOError("disk full")

    monkeypatch.setattr(builtins, "open", _boom)
    ah_plugin._sink.emit({"session": {"session_id": "s1"},
                          "agent_status": {"state": "running", "state_since_ms": 1}})
    monkeypatch.setattr(builtins, "open", real_open)


# ---------------- middleware / hooks ----------------

def test_middleware_injects_header_cow(ah_plugin, monkeypatch):
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    ah_plugin._settings._reset()
    req = {"model": "x", "messages": [{"role": "system"}, {"role": "user", "content": LONG_PREFIX}]}
    result = ah_plugin._on_llm_request(request=req, session_id="s1", api_request_id="r1")
    assert result is not None
    assert result["request"] is not req  # COW
    assert req.get("extra_headers") is None  # 不改入参
    hint = ah_plugin._codec.decode(result["request"]["extra_headers"]["X-Agent-Hint"])
    assert hint["session"]["call_index"] == 1
    assert hint["agent_status"]["state"] == "running"
    # 默认 auto：没有可用的运行时信号时按协议默认 gap（300s * 2）声明保留窗口；
    # pin 已退役，只发一条 ephemeral。
    assert hint["cache_control"] == [
        # 断点默认 auto：落在当前逻辑输入末尾（system + user = 2），不再只护 system 那条
        {"type": "ephemeral", "ttl_s": 600, "range": {"end": 2, "unit": "message"}},
    ]


def test_middleware_request_not_dict(ah_plugin, monkeypatch):
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    ah_plugin._settings._reset()
    assert ah_plugin._on_llm_request(request="not-a-dict", session_id="s1") is None
    assert ah_plugin._on_llm_request(request={}, session_id="") is None


def test_middleware_aux_request_bare(ah_plugin, monkeypatch):
    """辅助请求（压缩摘要 / 会话标题）整段豁免（RFC §1.3 / §3.3.5）。

    宿主在中间件调用点声明：``auxiliary=1`` 或 ``purpose`` 属辅助类。插件原样
    放行——不带 hint、不推进 call_index；被豁免的调用不占用序号。
    """
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    ah_plugin._settings._reset()
    req = {"model": "x", "messages": [{"role": "user", "content": "summarize"}]}
    # auxiliary 显式声明
    out = ah_plugin._on_llm_request(request=req, session_id="s1",
                                    api_request_id="r1", auxiliary=True)
    assert out is None
    assert "extra_headers" not in req          # 入参原样，未被动过
    # purpose 声明（大小写不敏感）
    for purpose in ("compaction", "Summarization", "title"):
        out = ah_plugin._on_llm_request(request=dict(req), session_id="s1",
                                        api_request_id="r2", purpose=purpose)
        assert out is None, purpose
    # 非辅助 purpose 不豁免：同一条请求照常装饰，且 call_index 从 1 起
    # （前面被豁免的三次调用没有推进序号）
    out = ah_plugin._on_llm_request(request=dict(req), session_id="s1",
                                    api_request_id="r3", purpose="chat")
    hint = ah_plugin._codec.decode(out["request"]["extra_headers"]["X-Agent-Hint"])
    assert hint["session"]["call_index"] == 1


def test_pre_tool_call_returns_none(ah_plugin, monkeypatch):
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    ah_plugin._settings._reset()
    assert ah_plugin._on_pre_tool_call(session_id="s1", tool_name="research_web_search",
                                       tool_call_id="c1") is None  # 绝不干预


def test_hooks_disabled_early_return(ah_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_HINT_ENABLED", "0")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_HINT_STATUS_SINK", "ndjson")
    ah_plugin._settings._reset()
    ah_plugin._sink._ndjson_path = None
    ah_plugin._on_session_start(session_id="s1")
    ah_plugin._on_pre_tool_call(session_id="s1", tool_name="t", tool_call_id="c")
    assert not (tmp_path / "agent-hint-events.ndjson").exists()  # 关闭态零事件


def test_subagent_hint_via_middleware(ah_plugin, monkeypatch):
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    ah_plugin._settings._reset()
    ah_plugin._state.subagent_start("s-parent", "s-child", "leaf")
    result = ah_plugin._on_llm_request(
        request={"messages": [{"role": "system"}]}, session_id="s-child", api_request_id="r1")
    hint = ah_plugin._codec.decode(result["request"]["extra_headers"]["X-Agent-Hint"])
    assert hint["session"]["parent_session_id"] == "s-parent"
    assert hint["session"]["derivation"] == {"relation": "spawn", "join": "sync"}
    assert hint["session"]["agent_type"] == "leaf"


# ---------------- review 补强：锁 / subagent 清理 / epoch 接缝 ----------------

def test_stats_snapshot_thread_safe(ah_plugin):
    """review A1：snapshot 与 collect 并发不得抛（deque 迭代竞态）。"""
    import threading
    st = ah_plugin._stats
    caught = []

    def _worker(base):
        for j in range(300):
            st.collect(session_id="s%d" % (base % 4), usage={"output_tokens": j},
                       api_duration=0.001, started_at=0.0, first_chunk_at=0.001, ended_at=0.002)
            try:
                st.snapshot("s%d" % (base % 4))
            except Exception as ex:  # 断言有错才走
                caught.append(ex)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not caught, "snapshot 并发下不应抛异常: %r" % caught
    assert st.snapshot("s0") is not None


def test_subagent_drop_purges_own_ref(ah_plugin):
    """review A2：未走 subagent_stop 直接 drop(child) 也应清 child 自身父映射。"""
    s = ah_plugin._state
    s.subagent_start("s-parent", "s-child", "leaf")
    assert s.lineage("s-child")[0] == "s-parent"
    s.subagent_stop("s-child")  # 正常路径
    assert s.lineage("s-child") == (None, None)
    # 中断/异常路径：直接 drop(child)
    s.subagent_start("s-parent", "s-child2", "leaf")
    s.drop("s-child2")
    assert s.lineage("s-child2") == (None, None)
    assert "s-child2" not in s._subagent


def test_subagent_lru_eviction_purges_ref(ah_plugin):
    """review A2：LRU 淘汰会话时一并清理其 _subagent 父映射。"""
    s = ah_plugin._state
    s.subagent_start("p0", "c0", "leaf")  # 建 c0 会话（最旧）+ 父映射
    assert "c0" in s._subagent
    for i in range(s._MAX_SESSIONS):
        s.note_call("s%d" % i, "r")  # 填满触发 c0 淘汰
    assert "c0" not in s._subagent  # 随 LRU 一并清理


def test_epoch_counter_and_provider_seam(ah_plugin):
    """review B：epoch 计数器 + epoch_provider 接缝（默认恒 0）。"""
    s = ah_plugin._state
    assert s.note_call("s1", "r1")[3] == 0        # 默认内部计数器恒 0
    assert s.bump_epoch("s1") == 1                # 计数器递增（接缝）
    assert s.note_call("s1", "r2")[3] == 1
    s.set_epoch_provider(lambda _sid: 42)         # provider 接缝覆盖
    assert s.note_call("s1", "r3")[3] == 42
    s.set_epoch_provider(lambda _sid: 1 / 0)      # provider 异常 -> 回退内部计数器
    assert s.note_call("s1", "r4")[3] == 1
    s.set_epoch_provider(None)                    # 恢复内部计数器
    assert s.note_call("s1", "r5")[3] == 1
    assert s.bump_epoch("s-nonexistent") == 0     # 未知会话安全返回 0


def test_priority_folds_onto_the_two_level_contract(ah_plugin, monkeypatch):
    """priority.level 只保留 {1, 3}：消费侧只跟一条准入线比较。

    v1 遗留的 2/4 折叠后仍落在原来那一侧（2 → 1、4 → 3），越界值同理；
    reason 原样保留，用来承载真正丰富的语义。
    """
    s = ah_plugin._settings
    for configured, expected in (("1", 1), ("2", 1), ("3", 3), ("4", 3), ("90", 3), ("7", 3)):
        monkeypatch.setenv("AGENT_HINT_PRIORITY_LEVEL", configured)
        monkeypatch.setenv("AGENT_HINT_PRIORITY_REASON", "critical_path")
        s._reset()
        assert s.priority() == {"level": expected, "reason": "critical_path"}, configured


def test_priority_reason_folds_onto_the_rfc_vocab(ah_plugin, monkeypatch):
    """reason 出口必须落在 RFC 词表 {user_waiting, critical_path, background}。

    词表外的取值在消费侧没有语义（拿它做准入判断只会失配）：冒号前缀命中词表的
    取前缀（``background: context_compaction`` → ``background``），完全词表外的按
    档位兜底——免排队档必然有人在等，背景档一律 background。
    """
    s = ah_plugin._settings
    cases = [
        ("background: context_compaction", "1", 1, "background"),
        ("background: context_compaction", "3", 3, "background"),
        ("continuation: tool_followup", "2", 1, "background"),
        ("CRITICAL_PATH: retry", "3", 3, "critical_path"),
        ("totally-custom-reason", "3", 3, "user_waiting"),
        ("totally-custom-reason", "1", 1, "background"),
        ("", "3", 3, "user_waiting"),
        ("", "1", 1, "background"),
    ]
    for raw, level, folded, reason in cases:
        monkeypatch.setenv("AGENT_HINT_PRIORITY_LEVEL", level)
        monkeypatch.setenv("AGENT_HINT_PRIORITY_REASON", raw)
        s._reset()
        assert s.priority() == {"level": folded, "reason": reason}, (raw, level)

def test_cache_pin_end_defaults_to_the_end_of_the_logical_input(ah_plugin, monkeypatch):
    """断点默认落在当前逻辑输入末尾，不再只护住 system 那一条。"""
    s = ah_plugin._settings
    monkeypatch.delenv("AGENT_HINT_CACHE_PIN_END", raising=False)
    s._reset()
    assert s.cache_pin_end() is None  # None = auto

    monkeypatch.setenv("AGENT_HINT_CACHE_PIN_END", "auto")
    s._reset()
    assert s.cache_pin_end() is None

    monkeypatch.setenv("AGENT_HINT_CACHE_PIN_END", "0")
    s._reset()
    assert s.cache_pin_end() == 0

    monkeypatch.setenv("AGENT_HINT_CACHE_PIN_END", "3")
    s._reset()
    assert s.cache_pin_end() == 3


def test_emitted_breakpoint_covers_the_whole_input(ah_plugin, monkeypatch):
    """发出去的 range.end 等于消息条数（含 system），与 Pi 一致。"""
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    monkeypatch.delenv("AGENT_HINT_CACHE_PIN_END", raising=False)
    monkeypatch.delenv("AGENT_HINT_CACHE_TASK_END", raising=False)
    ah_plugin._settings._reset()

    hooks, mw = {}, {}

    class _C:
        def register_hook(self, name, cb):
            hooks[name] = cb

        def register_middleware(self, name, cb):
            mw[name] = cb

    ah_plugin.register(_C())
    req = {"model": "m", "messages": [
        {"role": "system", "content": "sys " + "x " * 400},
        {"role": "user", "content": "hello " + "y " * 200},
        {"role": "assistant", "content": "hi"},
    ]}
    out = mw["llm_request"](request=req, session_id="s-end", api_request_id="r-1")
    raw = (out.get("request") or {}).get("extra_headers", {}).get("X-Agent-Hint")
    import base64 as _b64
    import json as _json
    hint = _json.loads(_b64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode())
    assert hint["cache_control"][0]["range"]["end"] == 3

def test_declared_fork_lineage_is_published(ah_plugin, monkeypatch):
    """Hermes 原生只有 spawn；fork 由部署方声明。"""
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    monkeypatch.setenv("AGENT_HINT_LINEAGE",
                       '{"relation":"fork","parent_session_id":"s-root","fork_point":7}')
    ah_plugin._settings._reset()

    lineage = ah_plugin._settings.lineage()
    assert lineage == {"relation": "fork", "parent_session_id": "s-root", "fork_point": 7}

    class _C:
        def register_hook(self, name, cb):
            pass

        def register_middleware(self, name, cb):
            pass

    ah_plugin.register(_C())
    parent, derivation = ah_plugin._state.lineage("s-child")
    assert parent == "s-root"
    assert derivation == {"relation": "fork", "fork_point": 7}


def test_native_subagent_wins_over_declared_lineage(ah_plugin, monkeypatch):
    """子 agent 的原生 spawn 优先于配置声明。"""
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    monkeypatch.setenv("AGENT_HINT_LINEAGE",
                       '{"relation":"fork","parent_session_id":"s-root","fork_point":1}')
    ah_plugin._settings._reset()
    ah_plugin._state.subagent_start("s-parent", "s-sub", "worker")
    parent, derivation = ah_plugin._state.lineage("s-sub")
    assert parent == "s-parent"
    assert derivation == {"relation": "spawn", "join": "sync"}


def test_declared_lineage_is_ignored_when_malformed(ah_plugin, monkeypatch):
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    monkeypatch.setenv("AGENT_HINT_LINEAGE", '{"relation":"adopted"}')
    ah_plugin._settings._reset()
    assert ah_plugin._settings.lineage() is None
    assert ah_plugin._state.lineage("s-root") == (None, None)

def test_resume_state_is_reported_once_after_approval(ah_plugin, monkeypatch):
    """被放行的这一条请求报 resume，之后回到 running。"""
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    ah_plugin._settings._reset()

    mw = {}
    class _C:
        def register_hook(self, name, cb):
            pass

        def register_middleware(self, name, cb):
            mw[name] = cb

    ah_plugin.register(_C())
    ah_plugin._on_pre_approval_request(session_id="s-res", kind="approval")
    ah_plugin._on_post_approval_response(session_id="s-res", choice="once")

    req = {"model": "m", "messages": [
        {"role": "system", "content": "sys " + "x " * 400},
        {"role": "user", "content": "hi"}]}
    out = mw["llm_request"](request=req, session_id="s-res", api_request_id="r-1")
    hint = ah_plugin._codec.decode(out["request"]["extra_headers"]["X-Agent-Hint"])
    assert hint["agent_status"]["state"] == "resume"

    out2 = mw["llm_request"](request=req, session_id="s-res", api_request_id="r-2")
    hint2 = ah_plugin._codec.decode(out2["request"]["extra_headers"]["X-Agent-Hint"])
    assert hint2["agent_status"]["state"] == "running"


def test_epoch_bumps_when_the_context_shrinks(ah_plugin, monkeypatch):
    """Hermes 没有 compaction 事件：消息条数回落即判定压缩发生，epoch +1。"""
    monkeypatch.setenv("AGENT_HINT_ENABLED", "1")
    ah_plugin._settings._reset()
    mw = {}
    class _C:
        def register_hook(self, name, cb):
            pass

        def register_middleware(self, name, cb):
            mw[name] = cb

    ah_plugin.register(_C())

    def emitted_epoch(api_id):
        req = {"model": "m", "messages": [
            {"role": "system", "content": "sys " + "x " * 400},
            {"role": "user", "content": "hi"}]}
        out = mw["llm_request"](request=req, session_id="s-epoch", api_request_id=api_id)
        return ah_plugin._codec.decode(out["request"]["extra_headers"]["X-Agent-Hint"])["session"]["epoch"]

    ah_plugin._on_pre_api_request(session_id="s-epoch", message_count=40)
    assert emitted_epoch("r-1") == 0
    # 条数增长：不是压缩
    ah_plugin._on_pre_api_request(session_id="s-epoch", message_count=48)
    assert emitted_epoch("r-2") == 0
    # 条数回落：压缩发生
    ah_plugin._on_pre_api_request(session_id="s-epoch", message_count=12)
    assert emitted_epoch("r-3") == 1