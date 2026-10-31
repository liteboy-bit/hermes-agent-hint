# agent-hint

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Hermes >= 0.21.6](https://img.shields.io/badge/hermes-%E2%89%A50.21.6-8A2BE2.svg)](https://github.com/NousResearch/hermes-agent)
[![stdlib only](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#safety-properties)
[![tests](https://img.shields.io/badge/tests-36%20passing-brightgreen.svg)](#tests)
[![measured impact](https://img.shields.io/badge/report-benchmark%20%2B%20field%20test-orange.svg)](docs/benchmark-report.md)

**English** · **[中文](#中文)**

> Give your inference engine — scheduler, gateway, KV-cache manager — a **standard,
> machine-readable view of the agent session** driving each request: who the session is, where it
> is in its lifecycle, how urgent it is, and what it is likely to do next.

The plugin attaches a single HTTP header — `X-Agent-Hint` — to **main LLM requests only**, and
optionally emits session lifecycle events out-of-band. It never touches the request body, never
modifies Hermes core, and is **off by default**.

```
LLM request ──► llm_request middleware ──► + X-Agent-Hint header ──► inference engine
                                          ▲
   hooks (session / tool / approval / subagent / api) ──► status events (ndjson | http | none)
```

## Why

An OpenAI-compatible HTTP API carries no notion of "session". An agent, however, has one: turn
boundaries, tool calls, approval waits, sub-agents, and a strong affinity to whatever instance
holds its KV cache. Engines that can see that structure can schedule better — pin a session to the
instance that already holds its cache, let a user-blocked session outrank a background one, or hold
capacity for a session that is about to resume.

### Measured impact

Not a hypothesis — measured against a MindIE-Motor deployment serving Qwen3-8B on an 8-card
Ascend 910B4 NPU server (vLLM engines, prefix caching on):

| What was measured | No hint | With hint |
|---|---|---|
| Which instance a session lands on | 6 requests scattered **3:3** across two instances, `aff=-`, `prio=None` | same session pinned: `aff=fresh` then **`aff=hit` ×3** (4 requests); **7 consecutive affinity hits** in an 8-round run; the no-hint control logged none |
| Prefix-cache hit rate | cold prefix: **0.0%** | warm prefix on the same instance: **84–99%** |
| p50 TTFT, ~3 k-token prefix | 2.079 s | **1.598 s → 1.30× (≈0.48 s saved)** |

Cost of the above: **one header**, no request-body change, no dependency, off by default. Nine real
agent requests were captured in flight — zero header decode failures, and not one request body
carried a hint field.

**Full evidence, method, and honest limitations →**
**[`docs/benchmark-report.md`](docs/benchmark-report.md)** ·
**[`docs/benchmark-report.zh-CN.md`](docs/benchmark-report.zh-CN.md)**

## Install

```bash
hermes plugins install liteboy-bit/hermes-agent-hint   # or: hermes plugins install <repo-url>
```

Or drop the directory into `$HERMES_HOME/plugins/agent-hint/` (per-profile) and enable it:

```bash
hermes plugins enable agent-hint
# if the enable command is unavailable in your setup, set the list directly in config.yaml:
#   plugins:
#     enabled: ["agent-hint"]
```

> **Installing does not turn it on.** Two switches must both be on, or the middleware returns
> immediately and no header is sent:
>
> 1. the plugin is enabled (`plugins.enabled` / `hermes plugins enable agent-hint`),
> 2. the plugin's own master switch is on: `settings.enabled: true`, or `AGENT_HINT_ENABLED=1`.

## Configure

Settings live under `plugins.entries.agent-hint.settings` in `config.yaml`; every key can also be set
via an `AGENT_HINT_<KEY>` environment variable. Precedence: **settings > env > built-in default**.
Every key is optional — the defaults are the safe defaults.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. Off = no header, no events. |
| `agent_type` | `hermes` | Value written to `session.agent_type`. Set per deployment/role. |
| `priority_level` | `3` | `priority.level`, folded to the two-level contract: `>= 3` becomes `3` (a person is waiting), anything else becomes `1`. `<= 0` omits the whole `priority` block. |
| `priority_reason` | `user_waiting` | `priority.reason` text. |
| `status_sink` | `none` | Event transport: `ndjson`, `http`, or `none`. |
| `status_url` | *(empty)* | Target URL for the `http` sink. Empty = no network traffic at all. |
| `finalize_on` | `session_end` | When to emit `finalize`: `session_end` or `session_finalize`. |
| `suspend_after_s` | `0` | Emit `suspend` if an approval stays pending this long. `0` disables. |
| `cache_pin_end` | `1` | Breakpoint position (`range.end`) of the single `ephemeral` entry; `0` omits the range. Historical name, kept for config compatibility. |
| `cache_ttl_s` | `auto` | Retention seconds for the ephemeral range. `auto` derives it from the observed gap (next tool budget, else the inter-call interval, else 300 s, doubled and clamped to 60–3600); `0` sends no declaration at all; a number fixes it. |
| `cache_task_end` | `0` | Message index of the task-description end. When wider than `cache_pin_end` it becomes the position of the **single** breakpoint; the declaration itself stays one entry. `0` uses `cache_pin_end`. |
| `next_action_map` | *(empty)* | Generic `prediction.next_action` heuristic (JSON). Empty = omit. |

Example:

```yaml
plugins:
  entries:
    agent-hint:
      settings:
        enabled: true
        agent_type: my-agent
        status_sink: http
        status_url: http://127.0.0.1:8080/agent-status
        next_action_map: '{"search_": ["summarize_run", 6, "tool"], "__default__": [null, 0, "tool"]}'
```

## The `X-Agent-Hint` header

`base64url(UTF-8 JSON)`, padding stripped — header values must stay ASCII and single-line. Field
classes carried in the JSON:

| Class | Fields | Notes |
|---|---|---|
| `session` | `session_id`, `agent_type`, `call_index`, `epoch` | Identity + call counter; `epoch` increments when the context is rewritten. |
| `agent_status` | `state` (`running` / `tool` / `pause` / `finalize`), `state_since_ms`, `reason` | Lifecycle, out-of-band from the request path. |
| `priority` | `level`, `reason` | Soft priority hint; the engine decides. |
| `cache_control` | `ephemeral`, `ttl_s`, `range` | One retention declaration for the session's prefix (`pin` is accepted by consumers as a retired alias). |
| `prediction` | `latency_ms`, `output_tokens`, `interval_ms`, `next_action`, `remaining_calls` | Statistics are measured; `next_action`/`remaining_calls` are **heuristics** and are omitted unless `next_action_map` is configured. |

`strict_priority` is deliberately **not** emitted: a client-side plugin must not be able to claim a
hard priority. Trust decisions belong to the control plane.

Status events (same shapes, plus a monotonic `X-Agent-Hint-Seq` on the `http` sink) are emitted for
session start/end, tool start/finish, approval wait/resume, sub-agent start/stop, and finalize.

## Safety properties

- **Off by default.** Nothing is injected and nothing is sent until you enable it.
- **No network unless you point it at a URL.** `status_sink` defaults to `none`; `ndjson` writes
  locally, `http` sends only to `status_url`.
- **Fail-safe.** Every hook and every sink swallows its own errors (warn-once to stderr) — a broken
  hint can never break a turn. The middleware returns the request untouched on any failure.
- **Pure stdlib, no core access.** Only `ctx.register_hook` / `ctx.register_middleware` /
  `ctx.get_config` are used; no Hermes internals are imported and no core object is patched.
- **Observers stay observers.** `pre_tool_call` never returns a block/approve/modify directive.
- **Unsupported by design:** `plugins.isolation: host` (a per-profile host process) is not covered —
  the middleware would have to serialize the whole request across the process boundary on every LLM
  call. This plugin is intended for the default `in_process` isolation.

## Compatibility

`requires_hermes: ">=0.21.6"`. Verified against an **unmodified** Hermes v0.21.6 checkout — the
plugin uses only public, documented surfaces (`VALID_HOOKS` ∩ the `llm_request` middleware), so no
core patch is needed.

Hook payloads are read defensively and by keyword; a hook whose payload changes shape degrades to
"no event" rather than an error.

## Tests

```bash
python3 -m pytest tests/     # 36 tests, no network, no Hermes install required
```

## Verification

Beyond the unit tests, three independent results back this plugin (all on an **unmodified** Hermes
v0.21.6 checkout):

1. **Upstream gate** — `hermes plugins validate .` passes in full: manifest fields, `requires_hermes`,
   `config_schema`, the capability probe (`register()` run in isolation, declared hooks/middleware
   matching registrations), security scan, and "no core override".
2. **Standalone turn + real engine** — installed as the *only* plugin into a clean `HERMES_HOME`
   (no bundle, no companion plugins), a real `hermes -z` turn produced two `X-Agent-Hint` headers
   (`call_index` 1→2) and the event sequence `running → tool → running → finalize`; with the plugin
   disabled the same run was completely silent. The header bytes emitted by this plugin's own
   `_builder`/`_codec` were then sent to a MindIE-Motor scheduler, whose log shows the session being
   pinned — `session affinity hit session_id=… instance_id=4` after a first request with
   `prio=3 aff=fresh` and a second with `prio=3 aff=hit`, while an identical request without the
   header logged `prio=None aff=-`.
3. **Benchmark + field test** — cold/warm prefix-cache measurement, hint-off/on routing comparison,
   and 9 captured real-agent requests, with the numbers in the [report](docs/benchmark-report.md).

## License

MIT — see [`LICENSE`](LICENSE).

---

# 中文

**中文** · **[English](#agent-hint)**

> 给下游推理引擎（调度器 / 网关 / KV 缓存管理器）一个**标准化、机器可读的「agent 会话视图」**：
> 这个请求属于哪个会话、会话处在生命周期哪个阶段、有多紧急、接下来大概要做什么。

插件把**单个 HTTP 头 `X-Agent-Hint`** 附加到**主 LLM 请求**上，并可按需把会话生命周期事件走旁路
上报。它**绝不修改请求体**、**绝不改动 Hermes 内核**，且**默认关闭**。

```
LLM 请求 ──► llm_request 中间件 ──► 附加 X-Agent-Hint 头 ──► 推理引擎
                                    ▲
   hooks（会话 / 工具 / 审批 / 子代理 / API）──► 状态事件（ndjson | http | none）
```

## 为什么需要它

OpenAI 兼容的 HTTP API 里没有「会话」这个概念。但 agent 有：回合边界、工具调用、审批等待、子代理，
以及「强烈倾向于回到持有自己 KV 缓存的实例」这一事实。能看见这层结构的引擎可以调度得更好 —— 把会话
钉在已持有其缓存的实例上、让被用户阻塞的会话压过后台任务、或为即将恢复的会话预留容量。

### 实测影响

这不是推测 —— 在一套 MindIE-Motor 部署上实测（8 卡 Ascend 910B4 NPU 服务器，服务 Qwen3-8B，
vLLM 引擎开启前缀缓存）：

| 测什么 | 无 hint | 带 hint |
|---|---|---|
| 会话落到哪个实例 | 6 个请求在两个实例间**3:3 散列**，`aff=-`、`prio=None` | 同一会话被钉住：`aff=fresh` 后 **`aff=hit` ×3**（连续 4 个请求）；8 轮序列里**连续 7 次亲和命中**；无 hint 对照组一条亲和日志都没有 |
| 前缀缓存命中率 | 冷前缀：**0.0%** | 同实例暖前缀：**84–99%** |
| 约 3k token 前缀的 p50 TTFT | 2.079s | **1.598s → 1.30×（省约 0.48s）** |

以上收益的代价：**一个 header**，不改请求体，无依赖，默认关闭。真机抓取 9 条 agent 真实请求 ——
header 解码 0 失败，且没有任何一个请求体携带 hint 字段。

**完整证据、测量方法与如实列出的局限 →**
**[`docs/benchmark-report.zh-CN.md`](docs/benchmark-report.zh-CN.md)** ·
**[`docs/benchmark-report.md`](docs/benchmark-report.md)**

## 安装

```bash
hermes plugins install liteboy-bit/hermes-agent-hint   # 或 hermes plugins install <仓库地址>
```

也可以直接把目录放进 `$HERMES_HOME/plugins/agent-hint/`（按 profile 生效）再启用：

```bash
hermes plugins enable agent-hint
# 若你的环境里 enable 命令不可用，可直接在 config.yaml 里写清单：
#   plugins:
#     enabled: ["agent-hint"]
```

> **装上不等于打开。** 两个开关必须同时为开，否则中间件直接返回、一个 header 都不会发：
>
> 1. 插件已启用（`plugins.enabled` / `hermes plugins enable agent-hint`）；
> 2. 插件自己的总开关：`settings.enabled: true`，或环境变量 `AGENT_HINT_ENABLED=1`。

## 配置

配置位于 `config.yaml` 的 `plugins.entries.agent-hint.settings`；每个键也可用 `AGENT_HINT_<KEY>`
环境变量设置。优先级：**settings > 环境变量 > 内置默认值**。所有键都是可选的 —— 默认值即安全值。

| 键 | 默认 | 含义 |
|---|---|---|
| `enabled` | `false` | 总开关。关闭 = 不发 header、不发事件。 |
| `agent_type` | `hermes` | 写入 `session.agent_type` 的值，按部署/角色自定。 |
| `priority_level` | `3` | `priority.level`，折叠进二值域：`>= 3` 记为 `3`（有人正在等），其余记为 `1`。`<= 0` 则整个 `priority` 段不发射。 |
| `priority_reason` | `user_waiting` | `priority.reason` 文本。 |
| `status_sink` | `none` | 事件通道：`ndjson`、`http`、`none`。 |
| `status_url` | *(空)* | `http` 通道的目标地址。留空 = 完全不产生网络流量。 |
| `finalize_on` | `session_end` | 何时发射 `finalize`：`session_end` 或 `session_finalize`。 |
| `suspend_after_s` | `0` | 审批挂起超过该秒数时发射 `suspend`；`0` 关闭。 |
| `cache_pin_end` | `1` | 单条 `ephemeral` 的断点位置（`range.end`）；`0` 则不带 range。历史命名，保留以兼容既有配置。 |
| `cache_ttl_s` | `auto` | ephemeral 保留秒数。`auto` 按运行时信号推导（下一次工具预算 → 请求间隔 → 默认 300s，乘 2 后钳到 60–3600）；`0` 不再发送任何声明；填数字则固定。 |
| `cache_task_end` | `0` | 任务说明末尾的消息下标。比 `cache_pin_end` 更宽时，它成为**唯一那条**断点的位置；声明本身仍只有一条。`0` 表示沿用 `cache_pin_end`。 |
| `next_action_map` | *(空)* | 通用的 `prediction.next_action` 启发式（JSON）。留空 = 不发射。 |

示例：

```yaml
plugins:
  entries:
    agent-hint:
      settings:
        enabled: true
        agent_type: my-agent
        status_sink: http
        status_url: http://127.0.0.1:8080/agent-status
        next_action_map: '{"search_": ["summarize_run", 6, "tool"], "__default__": [null, 0, "tool"]}'
```

## `X-Agent-Hint` 头

`base64url(UTF-8 JSON)`，去掉 padding —— header 值必须保持 ASCII 与单行。JSON 中承载的字段类别：

| 类别 | 字段 | 说明 |
|---|---|---|
| `session` | `session_id`、`agent_type`、`call_index`、`epoch` | 身份 + 调用计数；上下文被重写时 `epoch` 递增。 |
| `agent_status` | `state`（`running` / `tool` / `pause` / `finalize`）、`state_since_ms`、`reason` | 生命周期状态，走请求路径之外。 |
| `priority` | `level`、`reason` | 软优先级提示；最终由引擎决定。 |
| `cache_control` | `ephemeral`、`ttl_s`、`range` | 针对会话前缀的一条保留声明（消费侧仍接受 `pin` 作为已退役别名）。 |
| `prediction` | `latency_ms`、`output_tokens`、`interval_ms`、`next_action`、`remaining_calls` | 统计值为实测量；`next_action`/`remaining_calls` 属**启发式**，未配置 `next_action_map` 时不发射。 |

`strict_priority` **有意不发射**：客户端插件不应有能力声明硬优先级，信任决策属于控制面。

状态事件（结构同上，`http` 通道额外带单调递增的 `X-Agent-Hint-Seq`）覆盖：会话开始/结束、工具
开始/结束、审批等待/恢复、子代理开始/停止、以及 finalize。

## 安全属性

- **默认关闭。** 未启用前不注入任何内容、不外发任何数据。
- **不配 URL 就无网络。** `status_sink` 默认 `none`；`ndjson` 只写本地，`http` 只发往 `status_url`。
- **故障安全。** 每个 hook 与每个 sink 都自吞异常（只向 stderr 警告一次）—— 坏掉的 hint 永远无法
  破坏一次对话；中间件在任何失败情况下原样返回请求。
- **纯标准库，不碰内核。** 仅使用 `ctx.register_hook` / `ctx.register_middleware` / `ctx.get_config`；
  不导入 Hermes 内部模块，不 patch 任何核心对象。
- **旁观者只做旁观。** `pre_tool_call` 永不返回 block/approve/modify 指令。
- **有意不支持：** `plugins.isolation: host`（每 profile 独立宿主进程）不在覆盖范围 —— 那样中间件就得
  在每次 LLM 调用时把整个请求序列化跨进程传递。本插件面向默认的 `in_process` 隔离模式。

## 兼容性

`requires_hermes: ">=0.21.6"`。已在**未改动**的 Hermes v0.21.6 检出上验证 —— 插件只使用公开且
有文档的接口（`VALID_HOOKS` ∩ `llm_request` 中间件），无需任何内核补丁。

hook 载荷按关键字防御式读取；若某 hook 的载荷结构变化，会退化为「不发事件」而不是报错。

## 测试

```bash
python3 -m pytest tests/     # 36 个用例，无网络，无需安装 Hermes
```

## 验证

除单元测试外，有三项独立结果支撑本插件（均在**未改动**的 Hermes v0.21.6 检出上完成）：

1. **上游校验闸门** —— `hermes plugins validate .` 全项通过：manifest 字段、`requires_hermes`、
   `config_schema`、能力探针（隔离运行 `register()`，声明的 hooks/中间件与实际注册一致）、安全扫描、
   以及「未覆盖内核」。
2. **独立插件跑真实对话 + 真实引擎** —— 作为**唯一**插件装入干净的 `HERMES_HOME`（无 bundle、无配套
   插件），一次真实的 `hermes -z` 对话产生了两个 `X-Agent-Hint` 头（`call_index` 1→2）与事件序列
   `running → tool → running → finalize`；关闭插件后同一次运行完全静默。随后把本插件自己的
   `_builder`/`_codec` 产出的 header 字节发给 MindIE-Motor 调度器，其日志显示会话被钉住 ——
   首个请求 `prio=3 aff=fresh`、第二个 `prio=3 aff=hit` 并打出
   `session affinity hit session_id=… instance_id=4`，而同样内容不带 header 的请求记为
   `prio=None aff=-`。
3. **benchmark + 真机验证** —— 前缀缓存冷暖对照、hint 关/开的路由对照、以及 9 条真实 agent 请求抓取，
   数字见[报告](docs/benchmark-report.zh-CN.md)。

## 许可

MIT —— 见 [`LICENSE`](LICENSE)。
