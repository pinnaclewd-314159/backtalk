# Offline fallback: cloud Claude → local Qwen3-4B

**Status:** approved by Sir 2026-09-09, design only. Not yet implemented — next step is the writing-plans skill to turn this into an implementation plan.

## Problem

Sir lives in rural America and loses internet occasionally. When that happens, backtalk's entire "brain" — a persistent Claude Agent SDK session (`backtalk/brain.py`'s `WarmBrain`, a full Claude Code CLI instance with vault access, skills, and tool ecosystem) — goes dark, taking voice control of the house with it. Sir wants a local fallback that keeps basic capability (chiefly Home Assistant device control) working through an outage, explicitly scoped to "good enough for everyday chat offline," not full capability parity — that gap is real and accepted, not something to chase.

~90% of expected offline traffic is Home Assistant intents (turn things on/off, set values), confirmed by Sir. The model (Qwen3-4B-Instruct-2507, Q4_K_M, CPU inference via llama.cpp) was already benchmarked and reliability-tested against a 10-prompt HA-style harness — see [[Active Priorities]] and `local-llm-benchmark/` for that work. This spec covers the switchover mechanism around that already-chosen model, not model selection itself.

## Scope

- **In scope:** detecting a real internet/API outage vs. a transient hiccup; falling over from `WarmBrain` to a local Qwen3-4B-backed path; Home Assistant device control offline through the same tool shape already validated (`tools/homeassistant.py`); a short list of canned utility replies (current time, current date, simple arithmetic) that need no model call; an honest fallback line for anything else; switching back to cloud Claude once connectivity is confirmed stable; reusing the existing permission gate unchanged so offline actions get the same safety treatment as online ones.
- **Out of scope:** general open-ended chat offline (explicitly rejected — a small model with no tools/vault/skills would sometimes just be wrong, and that's worse than an honest "I can't do that right now"); any change to cross-channel Telegram sync (`tools/cross_channel_log.py`) or the Telegram bridge itself — both already depend on real internet access independently and simply go down with the same outage, no special handling needed; model selection (already decided); satellite-specific behavior beyond what `_speak_reply_satellite` already handles generically.

## Design

### Component 1 — `backtalk/connectivity.py` (new)

A `ConnectivityMonitor` that polls the actual dependency being relied on — not "the internet" generically, but reachability of Anthropic's API — via a lightweight HTTPS request to the API base URL every ~20 seconds, ~4 second timeout. Any response at all (not necessarily HTTP 200) counts as reachable: it proves DNS, TLS, and the network path all worked.

- Symmetric hysteresis: 2 consecutive failed polls flips `is_online` to `False`; 2 consecutive successful polls flips it back to `True`. Matches Sir's approved switch-back confirmation requirement, applied the same way on the way down.
- Exposes `is_online: bool` (read anywhere), `start()` (spawns the background asyncio poll loop, called once from `amain()`), `stop()`.
- A real, direct turn failure is stronger evidence than an indirect poll (see Error handling) and can set `is_online = False` immediately, bypassing the hysteresis — the hysteresis governs the ambient poll signal only.

### Component 2 — `backtalk/local_brain.py` (new)

`LocalBrain` implements exactly one method that matters: `async def ask_stream(utterance) -> AsyncIterator[str]`, the same shape as `WarmBrain.ask_stream`, so it drops into the existing `speak_reply`/`_speak_reply_local`/`_speak_reply_satellite` streaming-to-mouth plumbing unchanged (interruption handling, cross-channel logging hooks all keep working as-is for both cloud and offline turns). Deliberately has no `interrupt`/`reset_turn`/`command`/`start`/`stop` — those exist on `WarmBrain` to work around real Claude Agent SDK stream-desync bugs and session bookkeeping that simply don't apply to a stateless HTTP call to `llama-server`. `main.py` skips those calls when the active brain is `LocalBrain` rather than faking a shared interface for them.

`ask_stream` internally, in order:
1. **Canned utility check** — a short, fixed regex/keyword list: current time, current date, simple arithmetic (`\d+\s*[+\-*/]\s*\d+`-style). Matched utterances are answered instantly with no model call at all. Deliberately small and not meant to grow into a general capability — anything not on the list falls through to step 2.
2. **HA tool-call attempt** — POST to `llama-server`'s OpenAI-compatible `/v1/chat/completions` with the system prompt + two-tool schema already validated in `local-llm-benchmark/tool_call_test.py` (`call_ha_service`, `get_entity_state`), generalized from the test's small mocked entity list to the real one (pulled the same way `tools/homeassistant.py cmd_map` already does).
3. **Execute** — a returned `call_ha_service` tool call goes through the *same* permission gate cloud mode uses (see Safety below) before actually calling `tools/homeassistant.py`'s (newly extracted, see below) `call_service()`/`get_state()` functions directly — no subprocess, no CLI argv parsing.
4. **Honest fallback** — if the model returns neither a canned match nor a valid tool call, yield: *"I'm offline right now — I can only handle device control and a few basics until the connection's back."*

### Component 3 — `main.py` wiring

- At the top of `handle()`: `active_brain = brain if connectivity.is_online else local_brain`. Decided once per turn, never mid-response — a turn already streaming from cloud when the connection drops rides out whatever error handling already exists for a failed stream; the *next* turn re-evaluates.
- Cloud-only bookkeeping (`reset_turn`, `command`, session-resume tracking) is called only when `active_brain is brain`.
- A spoken + logged notice fires once on each transition: *"Sir, I've lost connectivity — falling back to local device control only,"* and on recovery, *"Sir, connectivity's back, I'm reconnected."* Silent degradation isn't acceptable here — a home-automation outage should be announced, not discovered.
- **WarmBrain lifecycle across the outage:** on the transition to offline, `main.py` calls `await brain.stop()` — cleanly disconnects the SDK session rather than leaving a dead connection dangling in the background. On the transition back online, `await brain.start()` reconnects, which already attempts to resume the previous session via the existing `resume_last_session`/`SESSION_FILE` mechanism (`brain.py`) — conversation memory survives the outage rather than starting fresh.

### Component 4 — the model server

New Windows Scheduled Task, same pattern as the existing "Jarvis - Voicebox Server" task (`AtLogOn` trigger, 3-retry/1-minute restart policy):

```
llama-server.exe -m <path to Qwen3-4B-Instruct-2507-Q4_K_M.gguf> --host 127.0.0.1 --port 8712 -t 4 -c 4096 --jinja
```

`-t 4` deliberately, not 8 — the `llama-bench` results in [[Active Priorities]] showed no benefit past 4 physical cores on this box's DDR3 dual-channel memory bandwidth (token generation is bandwidth-bound, not core-bound). Config for host/port/model path/poll settings lives in a new `local_fallback` block in `backtalk.json` + `config.py` DEFAULTS, following the existing `voicebox` block's pattern.

### Safety / permission gating

`LocalBrain` calls the same `can_use_tool` gate (`make_permission_gate` in `main.py`) before executing any `call_ha_service` tool call, respecting whatever `permission_mode` is currently configured (`ask`/`bypassPermissions`/etc.) — no separate, weaker gate for the degraded state. The gate wraps the tool-execution step, not the brain object, so this composes for free: no new gate-specific code needed in `LocalBrain` itself, just calling into the existing gate before executing a call.

### Required refactor: `tools/homeassistant.py`

`cmd_call`/`cmd_state` are currently CLI-shaped (parse `sys.argv`, print to stdout). Extract the actual request logic into plain, importable functions:
- `call_service(domain: str, service: str, entity_id: str, data: dict | None = None) -> dict`
- `get_state(entity_id: str) -> dict`

Both existing CLI commands and the new `LocalBrain` call these directly — mechanical extraction, not a redesign. No behavior change for the existing CLI usage.

### Error handling / edge cases

- **Double failure** — `llama-server` itself is down while backtalk is in offline mode (e.g. the Scheduled Task hasn't started yet, or crashed). `LocalBrain.ask_stream`'s HTTP call fails; caught and yields *"I can't reach my local fallback either right now."* Never hangs, never raises into `handle()`.
- **Reactive catch** — if a turn actually attempts `WarmBrain.ask_stream` and it fails/times out, that immediately sets `connectivity.is_online = False` for all subsequent turns, bypassing the poll's 2-failure hysteresis — a real failed call is more definitive than an indirect probe.
- **Mid-turn drops** — no special handling; see Component 3 above.
- **Lost tool-call reliability drift** — if a future `llama-server`/Qwen update changes tool-call behavior, `tool_call_test.py`'s harness (already built) is the regression check — rerun it before swapping model files or llama.cpp builds.

### Testing

- Extend `local-llm-benchmark/tool_call_test.py` to run against the real entity list (not the test's small mocked one) as a standing regression check.
- Force a real connectivity loss (disable the network adapter briefly) and confirm: fallback triggers within ~40s (2 failed 20s polls), HA control still works through it, the spoken transition notice fires, and it switches back cleanly ~40s after reconnecting.
- Confirm the permission gate still fires in `ask` mode while offline (a gated action still asks for spoken confirmation).
- Confirm canned time/date/arithmetic replies never hit `llama-server` (check its request log / stdout is silent for those turns).
- Confirm the double-failure case (kill `llama-server` while offline) gives the clean spoken error, not a hang.
- Confirm conversation memory survives the round trip — ask something before the outage, go through a simulated outage and recovery, confirm cloud Claude still has that context after reconnecting (session resume working).
