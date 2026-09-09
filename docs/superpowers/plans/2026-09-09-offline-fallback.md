# Offline Fallback (cloud Claude → local Qwen3-4B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When backtalk loses reachability to Anthropic's API, fall over from the cloud Claude Agent SDK session (`WarmBrain`) to a local Qwen3-4B-Instruct-2507 model (via `llama-server`, CPU) that handles Home Assistant device control and a short list of canned utility replies, then fall back to cloud Claude once connectivity is confirmed stable again.

**Architecture:** A `ConnectivityMonitor` (plain module, matching `signals.py`'s style) polls Anthropic's API on a timer and exposes `is_online()`; `main.py`'s `handle()` picks `brain` or a new `LocalBrain` per turn based on that flag. `LocalBrain` implements only `ask_stream()` — the one method `speak_reply`'s existing streaming-to-mouth machinery actually calls — so cloud and offline turns share the same voice UX, interruption handling, and cross-channel logging unchanged. `LocalBrain` talks to `llama-server`'s OpenAI-compatible API with the tool schema already validated in `local-llm-benchmark/tool_call_test.py`, executing real HA calls through the same permission gate cloud mode uses.

**Tech Stack:** Python 3.11/3.12 (backtalk's existing `.venv`), `httpx` (already a dependency) for both the connectivity probe and the `llama-server` calls, `llama-server.exe` (already downloaded to `local-llm-benchmark/llama-cpp/`) serving the already-selected `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.

**Spec:** `backtalk/docs/superpowers/specs/2026-09-09-offline-fallback-design.md`

## Global Constraints

- Model is already decided: Qwen3-4B-Instruct-2507, Q4_K_M GGUF — do not re-litigate model choice in this plan.
- `llama-server` runs with `-t 4` (not 8) — `llama-bench` showed no benefit past 4 physical cores on this box's DDR3 dual-channel bandwidth (see [[Active Priorities]]).
- Offline scope is HA intents + a short canned-utility list (time, date, simple arithmetic) only — never open-ended chat. Anything else gets the honest fallback line, not a freelanced answer.
- Offline HA actions go through the *same* permission gate (`make_permission_gate`) cloud mode uses — no separate, weaker gate for the degraded state.
- The routing decision (`brain` vs `local_brain`) is made once per turn, never mid-response.
- No pytest in this codebase — this project's convention is an `if __name__ == "__main__":` self-test block with plain `assert` statements (see `tools/cross_channel_log.py`), run via `python -m <module>` or `python <path>`. Follow that pattern, not a pytest suite.
- This is a real edit to `backtalk/main.py`, the file running the live voice session. Loading changes requires restarting the voice line, which ends whatever conversation is running — flag this to Sir immediately before that restart, per the pattern already used for the cross-channel memory rollout.

---

### Task 1: Extract `call_service`/`get_state` from `tools/homeassistant.py`

**Files:**
- Modify: `tools/homeassistant.py`

**Interfaces:**
- Produces: `call_service(domain: str, service: str, entity_id: str, data: dict | None = None) -> dict | None`, `get_state(entity_id: str) -> dict` — both plain, synchronous, importable functions. `LocalBrain` (Task 3) calls these via `loop.run_in_executor` since they block on network I/O.

Mechanical extraction only — no behavior change to the existing CLI (`cmd_call`/`cmd_state` keep working identically).

- [ ] **Step 1: Add the two functions, keep `cmd_call`/`cmd_state` calling them**

In `tools/homeassistant.py`, right before `def cmd_states():`, add:

```python
def call_service(domain: str, service: str, entity_id: str,
                  data: dict | None = None) -> dict | None:
    body = {"entity_id": entity_id}
    if data:
        body.update(data)
    return _request("POST", f"/api/services/{domain}/{service}", body)


def get_state(entity_id: str) -> dict:
    return _request("GET", f"/api/states/{entity_id}")
```

Then replace the body of `cmd_call` and `cmd_state`:

```python
def cmd_state(entity_id: str):
    print(json.dumps(get_state(entity_id), indent=2))


def cmd_call(domain: str, service: str, entity_id: str, extra: list[str]):
    data = {}
    for kv in extra:
        k, _, v = kv.partition("=")
        data[k] = v
    result = call_service(domain, service, entity_id, data)
    print(json.dumps(result, indent=2) if result else "OK")
```

- [ ] **Step 2: Manually verify the CLI still behaves identically**

Run: `python tools/homeassistant.py search "shop"` — confirm it prints entities exactly as before (this exercises `_request` unchanged; `call`/`state` themselves need real HA credentials to fully exercise, which is why this step is a manual smoke check, not an automated test — same reasoning as this file's existing lack of a self-test block).

- [ ] **Step 3: Commit**

```bash
git add tools/homeassistant.py
git commit -m "refactor: extract call_service/get_state as importable functions"
```

---

### Task 2: `backtalk/backtalk/connectivity.py` (new)

**Files:**
- Create: `backtalk/backtalk/connectivity.py`

**Interfaces:**
- Produces: `is_online() -> bool`, `start(url: str, on_change=None, interval_s=20.0, timeout_s=4.0, threshold=2, initial_online=True) -> None`, `async stop() -> None`, `force_offline() -> None`.
- Consumes: nothing from earlier tasks.
- `on_change`, when given, is an `async def on_change(online: bool)` called every time the state actually flips (not on the initial `start()` assignment). `main.py` (Task 5) supplies this to speak the transition notice and stop/start `WarmBrain`.

- [ ] **Step 1: Write the module**

```python
# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tracks whether backtalk can actually reach Anthropic's API — the real
dependency, not "the internet" in the abstract. A background poll every
`interval_s` probes it; symmetric hysteresis (`threshold` consecutive
results either direction) avoids flapping on a single blip.

`force_offline()` is the reactive half: a real failed turn against
WarmBrain is stronger evidence than an indirect poll, so it flips
immediately and bypasses the failure side of the hysteresis. The
success side (switching back) is untouched — still needs `threshold`
consecutive clean polls, since a turn succeeding once during a flaky
recovery isn't the same guarantee.

See backtalk/docs/superpowers/specs/2026-09-09-offline-fallback-design.md.
"""
import asyncio

import httpx

from backtalk.vlog import log

_STATE = {"online": True, "consec_ok": 0, "consec_fail": 0}
_task: asyncio.Task | None = None
_on_change = None
_URL = ""
_INTERVAL_S = 20.0
_TIMEOUT_S = 4.0
_THRESHOLD = 2


def is_online() -> bool:
    return _STATE["online"]


async def _probe() -> bool:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            await client.get(_URL)
        return True
    except Exception:
        return False


async def _flip(online: bool):
    _STATE["online"] = online
    log(f"[connectivity] {'back online' if online else 'offline'}")
    if _on_change:
        try:
            await _on_change(online)
        except Exception as e:
            log(f"[connectivity] on_change callback failed: {e!r}")


async def _poll_loop():
    while True:
        await asyncio.sleep(_INTERVAL_S)
        ok = await _probe()
        if ok:
            _STATE["consec_fail"] = 0
            _STATE["consec_ok"] += 1
            if not _STATE["online"] and _STATE["consec_ok"] >= _THRESHOLD:
                await _flip(True)
        else:
            _STATE["consec_ok"] = 0
            _STATE["consec_fail"] += 1
            if _STATE["online"] and _STATE["consec_fail"] >= _THRESHOLD:
                await _flip(False)


def start(url: str, on_change=None, interval_s: float = 20.0,
          timeout_s: float = 4.0, threshold: int = 2,
          initial_online: bool = True) -> None:
    """Begin the background poll. `initial_online` sets the starting
    state directly with NO on_change callback fired — on_change only
    fires for state changes the poll loop (or force_offline) detects
    going forward, never for this initial assignment. Callers that
    already know the boot outcome (Task 5's amain()) rely on this to
    avoid a duplicate/racing notification right at startup."""
    global _task, _on_change, _URL, _INTERVAL_S, _TIMEOUT_S, _THRESHOLD
    _URL, _on_change = url, on_change
    _INTERVAL_S, _TIMEOUT_S, _THRESHOLD = interval_s, timeout_s, threshold
    _STATE["online"] = initial_online
    _STATE["consec_ok"] = _STATE["consec_fail"] = 0
    _task = asyncio.create_task(_poll_loop())


async def stop() -> None:
    global _task
    if _task:
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
        _task = None


def force_offline() -> None:
    """Reactive override — call this the moment a real turn against
    WarmBrain fails/times out. Safe to call from a synchronous
    exception handler inside a running asyncio task."""
    _STATE["consec_ok"] = 0
    _STATE["consec_fail"] = _THRESHOLD
    if _STATE["online"]:
        asyncio.create_task(_flip(False))
```

- [ ] **Step 2: Write and run a self-test**

Append to the bottom of `backtalk/backtalk/connectivity.py`:

```python
if __name__ == "__main__":
    # Quick self-test / smoke check, not a full test suite.
    import unittest.mock as mock

    async def _run():
        calls = []

        async def fake_probe_ok():
            return True

        async def fake_probe_fail():
            return False

        async def on_change(online):
            calls.append(online)

        global _probe
        start("http://example.invalid", on_change=on_change,
              interval_s=0.01, timeout_s=0.01, threshold=2)
        assert is_online() is True, "should start online by default"

        _probe = fake_probe_fail
        await asyncio.sleep(0.03)   # 1 failed poll: not enough yet
        assert is_online() is True, "one failure must not flip it"
        await asyncio.sleep(0.03)   # 2nd failed poll: should flip
        assert is_online() is False, "two consecutive failures should flip offline"
        assert calls == [False], f"on_change should have fired once with False, got {calls}"

        _probe = fake_probe_ok
        await asyncio.sleep(0.03)
        assert is_online() is False, "one success must not flip it back yet"
        await asyncio.sleep(0.03)
        assert is_online() is True, "two consecutive successes should flip back online"
        assert calls == [False, True], f"on_change should have fired twice, got {calls}"

        await stop()

        # force_offline bypasses the failure hysteresis
        start("http://example.invalid", on_change=on_change, interval_s=999)
        calls.clear()
        force_offline()
        await asyncio.sleep(0.01)   # let the scheduled task run
        assert is_online() is False, "force_offline should flip immediately"
        assert calls == [False]
        await stop()
        print("connectivity self-test: OK")

    asyncio.run(_run())
```

Run: `cd backtalk && .venv/Scripts/python.exe -m backtalk.connectivity`
Expected: `connectivity self-test: OK`

- [ ] **Step 3: Commit**

```bash
git add backtalk/connectivity.py
git commit -m "feat: add ConnectivityMonitor for offline-fallback detection"
```

---

### Task 3: `backtalk/backtalk/local_brain.py` (new)

**Files:**
- Create: `backtalk/backtalk/local_brain.py`

**Interfaces:**
- Consumes: `homeassistant.call_service(domain, service, entity_id, data=None)`, `homeassistant.get_state(entity_id)` (Task 1).
- Produces: `class LocalBrain` with `async def ask_stream(self, utterance: str) -> AsyncIterator[str]` — same shape `speak_reply` already calls on `WarmBrain`. `__init__(self, can_use_tool, base_url="http://127.0.0.1:8712", timeout_s=60.0)` — `can_use_tool` is the exact `gate` callable `make_permission_gate(mouth)` already produces in `main.py` (Task 5 passes the same instance used for `WarmBrain`).
- The gate is called as `await can_use_tool("HAServiceCall", {"domain", "service", "entity_id", "data"}, None)` and must return an object with a `.behavior` attribute equal to `"allow"` to proceed — matches `PermissionResultAllow`/`PermissionResultDeny`'s existing shape in `main.py`. Task 5 adds the `"HAServiceCall"` branch to `_human_what`/`_full_detail` so the spoken ask reads sensibly instead of falling through to the generic "use the tool" line.

- [ ] **Step 1: Write the module**

```python
# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The offline brain — Qwen3-4B-Instruct-2507 via a local llama-server,
scoped deliberately to Home Assistant control plus a short list of
canned utility replies. NOT a smaller version of WarmBrain: no vault,
no skills, no general tool ecosystem. Anything outside that scope gets
an honest "I can't do that right now", never a freelanced answer.

Model/benchmark/tool-call-reliability work: local-llm-benchmark/ and
[[Active Priorities]]. Design: backtalk/docs/superpowers/specs/
2026-09-09-offline-fallback-design.md.
"""
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))
import homeassistant  # noqa: E402

from backtalk.vlog import log

_TIME_RE = re.compile(
    r"what(?:'s| is) the (?:current )?time|what time is it", re.I)
_DATE_RE = re.compile(
    r"what(?:'s| is) (?:today's )?date|what day is it", re.I)
_MATH_RE = re.compile(
    r"^\s*(?:what(?:'s| is)\s+)?(-?\d+(?:\.\d+)?)\s*"
    r"(plus|minus|times|divided by|x|\+|-|\*|/)\s*"
    r"(-?\d+(?:\.\d+)?)\s*\??\s*$", re.I)
_MATH_OPS = {"plus": "+", "minus": "-", "times": "*", "x": "*",
             "divided by": "/", "+": "+", "-": "-", "*": "*", "/": "/"}

FALLBACK_LINE = ("I'm offline right now. I can only handle device "
                  "control and a few basics until the connection's back.")
UNREACHABLE_LINE = "I can't reach my local fallback either right now."

# Real entity_ids, not mocked names — same set validated in
# local-llm-benchmark/tool_call_test.py's 10/10 reliability run. Extend
# this list by hand as you add rooms/devices; it is deliberately static
# rather than dynamically built from the full ~980-entity HA instance,
# which would blow the context and slow the model down for no benefit
# given the actual (small) scope of offline control.
SYSTEM_PROMPT = (
    "You are Jarvis, a home automation assistant. You control real "
    "devices in Sir's house via Home Assistant. Known entities: "
    "light.shop_benches (Shop Benches), light.electronics_bench "
    "(Electronics Bench), light.computer_benches (Computer Benches), "
    "switch.ewelink_switch_zr03_1_switch_2 (window fan, also wrapped as "
    "fan.window_fan_switch). Use the call_ha_service tool for any "
    "action (turn on/off, set a value). Use get_entity_state only to "
    "read a status, never to take an action. If the request has "
    "nothing to do with home automation, do not call any tool - just "
    "say you can't help with that right now."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_ha_service",
            "description": "Call a Home Assistant service to control a real device (lights, switches, fans, climate, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "HA domain, e.g. light, switch, fan, climate"},
                    "service": {"type": "string", "description": "HA service, e.g. turn_on, turn_off, set_temperature"},
                    "entity_id": {"type": "string", "description": "the exact entity_id to target"},
                    "data": {"type": "object", "description": "optional extra service params, e.g. {\"brightness_pct\": 50}"},
                },
                "required": ["domain", "service", "entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_state",
            "description": "Read the current state of a Home Assistant entity, no side effects.",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        },
    },
]


def _canned_reply(utterance: str) -> str | None:
    """A short, deliberately non-growing list — see the module docstring.
    Returns None (fall through to the model) for anything not on it."""
    if _TIME_RE.search(utterance):
        return f"It's {datetime.now().strftime('%I:%M %p').lstrip('0')}."
    if _DATE_RE.search(utterance):
        return f"It's {datetime.now().strftime('%A, %B %d')}."
    m = _MATH_RE.match(utterance.strip())
    if m:
        a_s, op_word, b_s = m.groups()
        op = _MATH_OPS[op_word.lower()]
        a, b = float(a_s), float(b_s)
        if op == "/" and b == 0:
            return "I can't divide by zero."
        result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
        result = int(result) if result == int(result) else round(result, 2)
        return f"That's {result}."
    return None


class LocalBrain:
    def __init__(self, can_use_tool, base_url: str = "http://127.0.0.1:8712",
                 timeout_s: float = 60.0):
        self._can_use_tool = can_use_tool
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        canned = _canned_reply(utterance)
        if canned is not None:
            yield canned
            return

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={
                        "model": "local",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": utterance},
                        ],
                        "tools": TOOLS,
                        "tool_choice": "auto",
                        "temperature": 0,
                    })
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            log(f"[local_brain] llama-server unreachable: {e!r}")
            yield UNREACHABLE_LINE
            return

        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            yield FALLBACK_LINE
            return

        loop = asyncio.get_running_loop()
        for tc in calls:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"])
            except Exception:
                continue
            if fn["name"] == "get_entity_state":
                entity_id = args.get("entity_id", "")
                try:
                    state = await loop.run_in_executor(
                        None, homeassistant.get_state, entity_id)
                    yield f"{entity_id} is {state.get('state', 'unknown')}."
                except Exception as e:
                    log(f"[local_brain] get_state failed: {e!r}")
                    yield f"I couldn't check that."
            elif fn["name"] == "call_ha_service":
                domain = args.get("domain", "")
                service = args.get("service", "")
                entity_id = args.get("entity_id", "")
                data_arg = args.get("data") or {}
                result = await self._can_use_tool(
                    "HAServiceCall",
                    {"domain": domain, "service": service,
                     "entity_id": entity_id, "data": data_arg},
                    None)
                if getattr(result, "behavior", "") != "allow":
                    yield "Okay, I won't do that."
                    continue
                try:
                    await loop.run_in_executor(
                        None, homeassistant.call_service, domain, service,
                        entity_id, data_arg)
                    yield "Done."
                except Exception as e:
                    log(f"[local_brain] call_service failed: {e!r}")
                    yield "That didn't work — check the log."
```

- [ ] **Step 2: Write and run a self-test against the real, already-running llama-server**

Append:

```python
if __name__ == "__main__":
    async def _allow_all(tool, tool_input, ctx):
        class _Allow:
            behavior = "allow"
        return _Allow()

    async def _run():
        brain = LocalBrain(can_use_tool=_allow_all)

        async def collect(utterance):
            return " ".join([s async for s in brain.ask_stream(utterance)])

        assert "2" not in await collect("what's 12 plus 7") or True  # sanity only
        out = await collect("what's 12 plus 7")
        assert "19" in out, f"expected 19 in {out!r}"
        out = await collect("what time is it")
        assert ":" in out, f"expected a time in {out!r}"
        out = await collect("turn on the shop lights")
        assert out.strip() in ("Done.",), f"expected Done., got {out!r}"
        out = await collect("what's the weather like today")
        assert out.strip() == FALLBACK_LINE.strip(), f"expected fallback line, got {out!r}"
        print("local_brain self-test: OK")

    asyncio.run(_run())
```

Run (llama-server must already be up on 8712 — start it manually first with the command from Task 6 if it isn't):
`cd backtalk && .venv/Scripts/python.exe -m backtalk.local_brain`
Expected: `local_brain self-test: OK`. **Note:** the "turn on the shop lights" case does a real HA call if your `tools/homeassistant.py` credentials are configured — confirm that's acceptable before running, or temporarily point `HOME_ASSISTANT_URL` at a non-production instance.

- [ ] **Step 3: Commit**

```bash
git add backtalk/local_brain.py
git commit -m "feat: add LocalBrain, the offline HA-intent fallback"
```

---

### Task 4: Config — `local_fallback` block

**Files:**
- Modify: `backtalk/backtalk/config.py`
- Modify: `backtalk/backtalk.json.example`

**Interfaces:**
- Produces: `CFG["local_fallback"]` dict with keys `enabled`, `base_url`, `health_check_url`, `poll_interval_s`, `poll_timeout_s`, `poll_threshold`. Task 5 reads these.

- [ ] **Step 1: Add the DEFAULTS block**

In `backtalk/backtalk/config.py`, right after the existing `"voicebox": { ... }` block closes, add:

```python
    # Offline fallback: cloud Claude -> local Qwen3-4B via llama-server,
    # for Home Assistant control + a short canned-utility list when the
    # internet is down. See backtalk/docs/superpowers/specs/
    # 2026-09-09-offline-fallback-design.md.
    "local_fallback": {
        "enabled": False,
        "base_url": "http://127.0.0.1:8712",
        "health_check_url": "https://api.anthropic.com",
        "poll_interval_s": 20.0,
        "poll_timeout_s": 4.0,
        "poll_threshold": 2,
    },
```

- [ ] **Step 2: Document it in the example config**

In `backtalk/backtalk.json.example`, add the same block (with `"enabled": true`, since the example is meant to be copied and used) as a top-level key, matching the existing `"voicebox"` entry's placement/style in that file.

- [ ] **Step 3: Manually verify the merge**

Run: `cd backtalk && .venv/Scripts/python.exe -c "from backtalk.config import CFG; print(CFG['local_fallback'])"`
Expected: prints the defaults dict (or your `backtalk.json` override, if you've added one).

- [ ] **Step 4: Commit**

```bash
git add backtalk/config.py backtalk/backtalk.json.example
git commit -m "feat: add local_fallback config block"
```

---

### Task 5: Wire it into `backtalk/main.py`

**Files:**
- Modify: `backtalk/main.py`

**Interfaces:**
- Consumes: `backtalk.connectivity` (Task 2), `backtalk.local_brain.LocalBrain` (Task 3), `CFG["local_fallback"]` (Task 4).

This is the task that touches the live voice pipeline. Several independent edits to the same file — do them in the order below so each is individually sane, but commit once at the end since they only make sense together.

- [ ] **Step 1: Import the new modules**

Near the top of `backtalk/main.py`, alongside the existing `from backtalk import satellites` / `from backtalk import signals`:

```python
from backtalk import connectivity
from backtalk.local_brain import LocalBrain
```

- [ ] **Step 2: Teach the permission gate to describe an HA service call**

In `_human_what`, add a branch before the final generic fallback (`name = getattr(ctx, "display_name", ...)`):

```python
    if tool == "HAServiceCall":
        verb = {"turn_on": "turn on", "turn_off": "turn off"}.get(
            d.get("service", ""), d.get("service", "control"))
        return f"{verb} {d.get('entity_id', 'a device')}"
```

In `_full_detail`, add the matching branch before its own generic fallback:

```python
    if tool == "HAServiceCall":
        extra = f" with {d.get('data')}" if d.get("data") else ""
        return (f"call Home Assistant service {d.get('domain')}."
                f"{d.get('service')} on {d.get('entity_id')}{extra}")
```

- [ ] **Step 3: Guard cloud-only console verbs**

Near the top of `_run_console_inner(verb)`, before the existing `_deny_pending()` line, add:

```python
    if (not connectivity.is_online()
            and (verb in ("clear", "compact", "deep", "fast", "usage")
                 or verb.startswith("effort:"))):
        mouth.say("That's not available while we're offline.")
        return
```

- [ ] **Step 4: Create `local_brain` and register the connectivity callback in `amain()`**

Find where `brain` is constructed:

```python
    mouth = Mouth()
    ears = Ears(silence_ms=CFG.get("silence_ms", 480))
    brain = WarmBrain(model=model,
                      can_use_tool=make_permission_gate(mouth),
                      resume_id=resume_id)
```

Replace with:

```python
    mouth = Mouth()
    ears = Ears(silence_ms=CFG.get("silence_ms", 480))
    perm_gate = make_permission_gate(mouth)
    brain = WarmBrain(model=model, can_use_tool=perm_gate,
                      resume_id=resume_id)
    lf_cfg = CFG.get("local_fallback", {})
    local_brain = LocalBrain(can_use_tool=perm_gate,
                              base_url=lf_cfg.get("base_url",
                                                  "http://127.0.0.1:8712"))

    async def _on_connectivity_change(online: bool):
        if online:
            mouth.say("Sir, connectivity's back, I'm reconnected.")
            try:
                await brain.start()
            except Exception as e:
                log(f"[backtalk] brain reconnect failed: {e!r}")
        else:
            mouth.say("Sir, I've lost connectivity — falling back to "
                      "local device control only.")
            try:
                await brain.stop()
            except Exception:
                pass
```

- [ ] **Step 5: Handle a cold boot that starts offline, without killing the process**

Find the brain-connect block:

```python
    log("[backtalk] connecting the brain...")
    try:
        await asyncio.wait_for(brain.start(), 120)

        async def _warmup():
            async for _ in brain.ask_stream(
                    "Warmup ping - reply with the single word: ready"):
                pass
        await asyncio.wait_for(_warmup(), 180)
    except (Exception, asyncio.TimeoutError) as e:
        kind = ("timed out" if isinstance(e, asyncio.TimeoutError)
                else f"failed: {e!r}"[:220])
        log(f"[backtalk] BRAIN CONNECT {kind}")
        mouth.say("Bad news. The voice and the face are fine, but I "
                  "couldn't reach my brain, the Claude Code session. "
                  "Check this window for the error. The usual causes: "
                  "Claude Code isn't signed in, the internet is down, "
                  "or the plan is out of usage.")
        mouth.wait_done(timeout=30)
        raise SystemExit(1)
    log("[backtalk] brain warm")
    # the hidden warmup ping is plumbing, not conversation
    brain.session.update(turns=0, out_tokens=0, in_tokens=0, cost=0.0)
    # a configured effort level applies at launch (saved by the spoken
    # "set effort to X", or written by the person's agent on request)
    boot_effort = str(CFG.get("effort") or "").strip().lower()
    if boot_effort in _EFFORTS:
        await brain.command(f"/effort {boot_effort}")
        log(f"[backtalk] effort set to {boot_effort} (from config)")
    elif boot_effort:
        log(f"[backtalk] ignoring unknown effort {boot_effort!r} in config")
```

Replace with:

```python
    log("[backtalk] connecting the brain...")
    booted_offline = False
    try:
        await asyncio.wait_for(brain.start(), 120)

        async def _warmup():
            async for _ in brain.ask_stream(
                    "Warmup ping - reply with the single word: ready"):
                pass
        await asyncio.wait_for(_warmup(), 180)
    except (Exception, asyncio.TimeoutError) as e:
        kind = ("timed out" if isinstance(e, asyncio.TimeoutError)
                else f"failed: {e!r}"[:220])
        log(f"[backtalk] BRAIN CONNECT {kind}")
        if not lf_cfg.get("enabled"):
            mouth.say("Bad news. The voice and the face are fine, but I "
                      "couldn't reach my brain, the Claude Code session. "
                      "Check this window for the error. The usual causes: "
                      "Claude Code isn't signed in, the internet is down, "
                      "or the plan is out of usage.")
            mouth.wait_done(timeout=30)
            raise SystemExit(1)
        # Local fallback is enabled: don't die on a cold boot during an
        # outage — start in offline mode. connectivity.start() below is
        # told the real initial state directly (no on_change fired for
        # it), so this doesn't double up with the poll loop's own
        # detection.
        booted_offline = True
        mouth.say("I couldn't reach my brain at startup, so I'm "
                  "starting in local device-control mode until the "
                  "connection's back.")
    if lf_cfg.get("enabled"):
        connectivity.start(
            lf_cfg.get("health_check_url", "https://api.anthropic.com"),
            on_change=_on_connectivity_change,
            interval_s=lf_cfg.get("poll_interval_s", 20.0),
            timeout_s=lf_cfg.get("poll_timeout_s", 4.0),
            threshold=lf_cfg.get("poll_threshold", 2),
            initial_online=not booted_offline)
    if not booted_offline:
        log("[backtalk] brain warm")
        brain.session.update(turns=0, out_tokens=0, in_tokens=0, cost=0.0)
        boot_effort = str(CFG.get("effort") or "").strip().lower()
        if boot_effort in _EFFORTS:
            await brain.command(f"/effort {boot_effort}")
            log(f"[backtalk] effort set to {boot_effort} (from config)")
        elif boot_effort:
            log(f"[backtalk] ignoring unknown effort {boot_effort!r} in config")
```

- [ ] **Step 6: Route each turn to the right brain in `handle()`**

Find, inside `handle()`:

```python
            signals.set_state("thinking")
            signals.static_start()
            # Clean the pipe: drain the interrupted turn's leftovers so the
            # new question can't pair with a stale ResultMessage. A gate
            # that fired in the meantime resolves first, or the drain would
            # wait on a ResultMessage the CLI is withholding for an answer.
            _deny_pending()
            await brain.reset_turn()
```

Replace the last two lines with:

```python
            _deny_pending()
            active_brain = brain if connectivity.is_online() else local_brain
            if active_brain is brain:
                await brain.reset_turn()
```

Then find:

```python
            speak_task = asyncio.create_task(
                speak_reply(brain, mouth, brain_text, source=source,
                            epoch=epoch, log_channel="voice" if source == "local" else None))
```

Replace `brain` with `active_brain`:

```python
            speak_task = asyncio.create_task(
                speak_reply(active_brain, mouth, brain_text, source=source,
                            epoch=epoch, log_channel="voice" if source == "local" else None))
```

- [ ] **Step 7: Reactive catch — a real cloud failure flips state immediately**

In `_speak_reply_local`, find:

```python
    try:
        async for sentence in brain.ask_stream(text):
            emit(sentence)
```

This function's `brain` parameter is now whichever brain `handle()` passed in (cloud or local) — a failure here should only force offline state when it was actually the cloud brain that failed. Change the signature and the except-free `try` block:

```python
async def _speak_reply_local(brain: WarmBrain, mouth: Mouth, text: str,
                              log_channel: str | None = None):
```

stays the same signature (it's duck-typed already), but wrap the streaming loop:

```python
    is_cloud = isinstance(brain, WarmBrain)
    try:
        try:
            async for sentence in brain.ask_stream(text):
                emit(sentence)
        except asyncio.CancelledError:
            raise
        except Exception:
            if is_cloud:
                connectivity.force_offline()
            raise
```

Note this nests inside the *existing* outer `try/except asyncio.CancelledError/finally` in that function — don't remove the existing structure, only wrap the `async for` loop itself as shown. The re-`raise` preserves the exact existing behavior (logged and propagated unchanged); the only addition is the `force_offline()` side-effect call.

Apply the identical pattern in `_speak_reply_satellite`'s `_rated_chunks()` generator, around:

```python
        try:
            async for sentence in brain.ask_stream(text):
```

wrap it the same way:

```python
        is_cloud = isinstance(brain, WarmBrain)
        try:
            try:
                async for sentence in brain.ask_stream(text):
                    ...  # existing body unchanged
            except asyncio.CancelledError:
                raise
            except Exception:
                if is_cloud:
                    connectivity.force_offline()
                raise
```

(Keep the existing outer `except asyncio.CancelledError` / `except Exception as e:` blocks in both functions exactly as they are — this step only adds the inner wrap around the streaming loop, it does not replace the outer handlers.)

- [ ] **Step 8: Stop the connectivity monitor on shutdown**

Find the existing shutdown line (near the end of `amain()`):

```python
        await brain.stop()
```

Add right after it:

```python
        if lf_cfg.get("enabled"):
            await connectivity.stop()
```

- [ ] **Step 9: Manually verify it still boots online (local fallback disabled by default)**

Run the normal launch (`Talk to Jarvis.bat` or the direct `uv run python -m backtalk.main` workaround already on record) with `local_fallback.enabled` left `false` in `backtalk.json`. Confirm: boots exactly as before, greets normally, a real turn works. This confirms the wiring is a no-op when the feature is off, before ever testing it live.

- [ ] **Step 10: Commit**

```bash
git add backtalk/main.py
git commit -m "feat: wire offline fallback into main.py's turn routing"
```

---

### Task 6: `llama-server` as a boot service

**Files:**
- No code files — a Windows Scheduled Task, created once by hand (or via a short PowerShell snippet, not part of backtalk's own repo).

- [ ] **Step 1: Move the model out of the scratch benchmark folder to a permanent home**

```powershell
New-Item -ItemType Directory -Force "C:\Users\JARVIS\my-agent\backtalk\models"
Move-Item "C:\Users\JARVIS\my-agent\local-llm-benchmark\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf" "C:\Users\JARVIS\my-agent\backtalk\models\"
```

(Leave `local-llm-benchmark/llama-cpp/` where it is — `llama-server.exe` there is fine to reference directly, or copy it alongside the model; either works, pick one and use that exact path in Step 2.)

- [ ] **Step 2: Create the Scheduled Task**

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Users\JARVIS\my-agent\local-llm-benchmark\llama-cpp\llama-server.exe" `
    -Argument '-m "C:\Users\JARVIS\my-agent\backtalk\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf" --host 127.0.0.1 --port 8712 -t 4 -c 4096 --jinja'
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId "JARVIS\JARVIS" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "Jarvis - Local LLM Fallback" -Action $action `
    -Trigger $trigger -Settings $settings -Principal $principal
```

(Mirrors the existing "Jarvis - Voicebox Server" task exactly — same trigger/restart-policy shape.)

- [ ] **Step 3: Start it now (don't wait for next logon) and verify**

```powershell
Start-ScheduledTask -TaskName "Jarvis - Local LLM Fallback"
Start-Sleep -Seconds 5
Get-NetTCPConnection -LocalPort 8712 -State Listen
```

Expected: one listening row on port 8712.

- [ ] **Step 4: Set `local_fallback.enabled: true` in `backtalk.json`**

Edit the live `backtalk.json` (not `.example`) to add:

```json
"local_fallback": { "enabled": true }
```

(Config merge is shallow-per-key, so this alone is enough — the other fields fall back to `DEFAULTS`.)

- [ ] **Step 5: Commit anything tracked**

`backtalk.json` is real config, not typically committed with secrets — check `.gitignore` before adding; if it's already ignored, nothing to commit here. If it's tracked, `git add backtalk.json && git commit -m "config: enable local fallback"`.

---

### Task 7: End-to-end manual verification

**Files:** none — this is the spec's Testing section, executed for real.

- [ ] **Step 1: Confirm normal (online) operation is unaffected**

Speak a normal request through backtalk with the feature enabled and internet up. Confirm identical behavior to before this project — `active_brain` should resolve to `brain` every time, `connectivity.is_online()` stays `True`.

- [ ] **Step 2: Force a real outage and confirm fallback**

Disable the network adapter (or block `api.anthropic.com` at the hosts-file/firewall level) for a couple of minutes. Confirm: within ~40s (2 failed 20s polls) the spoken transition notice fires ("falling back to local device control only"), and a spoken HA command (e.g. "turn on the shop lights") is executed via `LocalBrain`. Check `logs/backtalk.log` for `[connectivity] offline` and `[local_brain]` lines.

- [ ] **Step 3: Confirm the permission gate still fires while offline**

With `permission_mode` set to `ask`, issue an HA action while offline. Confirm the spoken "Permission check..." ask still happens (via the shared `perm_gate`) before the action executes.

- [ ] **Step 4: Confirm canned replies never touch `llama-server`**

Ask "what time is it" while offline. Confirm the reply is instant and `llama-server`'s own log shows no new request for that turn.

- [ ] **Step 5: Confirm the double-failure case**

Stop the "Jarvis - Local LLM Fallback" task while still offline, then issue an HA command. Confirm the spoken reply is "I can't reach my local fallback either right now" — no hang, no crash.

- [ ] **Step 6: Restart the local-LLM service, restore the network, confirm switch-back and memory continuity**

Restart the Scheduled Task. Re-enable the network adapter. Confirm: ~40s later, the "connectivity's back, I'm reconnected" notice fires, and a follow-up question referencing something said *before* the outage confirms `WarmBrain`'s session-resume brought the conversation memory back (not a fresh session).

- [ ] **Step 7: Confirm a cold boot during an outage doesn't crash**

With the network already down, launch backtalk fresh. Confirm it speaks the "starting in local device-control mode" line instead of exiting, and HA control works immediately.

---

## Self-Review Notes

- **Spec coverage:** all four components (ConnectivityMonitor, LocalBrain, main.py wiring, model server) and the required `tools/homeassistant.py` refactor each have a task; every item in the spec's Error Handling and Testing sections maps to a step in Task 5 or Task 7.
- **Beyond the literal spec text:** Task 5 Step 5 (cold-boot-while-offline) isn't spelled out in the spec, which focused on an outage starting *after* a successful boot. Flagging this explicitly: without it, the existing boot code's `raise SystemExit(1)` would still kill the process on a cold start during an outage, which defeats the point of having a fallback at all. Included as the natural, necessary completion of the feature — surfaced here rather than silently expanded.
- **Type/interface consistency:** `LocalBrain.ask_stream` and `WarmBrain.ask_stream` both yield `str` sentences; `speak_reply`/`_speak_reply_local`/`_speak_reply_satellite` consume either without change. `can_use_tool` is the same callable object passed to both brains. Checked.
