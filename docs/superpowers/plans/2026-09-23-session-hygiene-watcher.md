# Session Hygiene Watcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give backtalk's voice line an automatic, silent watcher that clears or compacts its own Claude session when it's been idle too long or context has filled up, using the existing `brain.command()` channel — no window automation, no new SDK capability.

**Architecture:** A new `SessionHygiene` class (`backtalk/hygiene.py`) with pure, independently-testable decision logic (`should_clear`, `should_compact`, `context_occupied_fraction`) plus a single per-tick action method (`tick`) that a thin `watch()` loop calls on an interval. `tick` is gated on the turn lock being free and connectivity being up, and drives `brain.command()` for a silent checkpoint turn followed by `/clear` or `/compact` — the exact channel the existing spoken "clear"/"compact" verbs already use (`main.py:1183`, `1187`).

**Tech Stack:** Python 3.12, `asyncio`, the already-installed `claude_agent_sdk`, `unittest`/`unittest.IsolatedAsyncioTestCase` (matches this repo's existing `tests/` convention — no new test dependency).

**Spec:** `backtalk/docs/superpowers/specs/2026-09-23-session-hygiene-watcher-design.md`

## Global Constraints

- `session_hygiene.enabled` defaults to `False` — the feature must ship off. Turning it on in `backtalk.json` is a separate, later config change.
- Never fire while `turn_lock.is_active()` is true — checked fresh every tick, never cached.
- Never fire while offline (`is_online_fn()` returns false) — skip the tick entirely, don't error.
- The watcher never calls `mouth.say()` — everything goes to `log()` only.
- `brain.command()` already returns the string `"error: the command timed out"` on failure rather than raising (`brain.py:307`) — any caller must treat a string starting with `"error:"` as failure, not success.
- The compaction counter lives in memory on the `SessionHygiene` instance and resets naturally on process restart — no persistence needed.
- No change to the existing spoken "clear"/"compact"/"deep"/"fast"/"usage" verbs in `main.py:1173-1219` — this plan adds a second caller of `brain.command()`, it doesn't touch the first.

## Review Focus

- **Context payload with no `categories` at all, or `None`** — `context_usage()` can itself return `None` (`brain.py:144`), and even a present `ctx_usage` may have an empty/malformed `categories` list. `context_occupied_fraction` must return `None` (not crash, not divide by zero) so `tick` treats it as "can't decide, skip this check."
- **A tick landing exactly when a turn starts** — the gating check happens once at the top of `tick`; a turn that starts a moment later must not be interrupted mid-flight by that same tick (nothing later in `tick` re-checks or blocks on the lock).
- **`brain.command()` returning the error sentinel** — on either the checkpoint call or the actual `/clear`/`/compact` call, `tick` must log and return without incrementing the compaction counter or treating the cycle as done.
- **The compaction cap boundary** — the 3rd successful compaction must still be a normal compact; only a 4th *trigger* (context still over threshold after the cap is reached) takes the heavier full-summary-then-clear path.
- **Going offline mid-cycle is out of scope for a single tick** — `is_online_fn()` is checked once at tick entry; a real connectivity drop mid-checkpoint is already handled by `brain.command()`'s own timeout/error-sentinel path (previous bullet), not by anything new here.

---

## Task 1: `session_hygiene` config defaults

**Files:**
- Modify: `backtalk/backtalk/config.py` (insert after the `n9_fallback` block, ~line 300)
- Test: `backtalk/tests/test_session_hygiene_config.py`

**Interfaces:**
- Produces: `CFG["session_hygiene"]` dict with keys `enabled`, `idle_clear_minutes`, `compact_context_threshold`, `max_compactions_per_session`, `check_interval_s`, consumed by later tasks via `CFG.get("session_hygiene", {})`.

- [ ] **Step 1: Write the failing test**

```python
"""session_hygiene must default to off, with sane thresholds, so
turning it on is a deliberate config change rather than a silent
behavior flip on upgrade."""
import unittest

from backtalk.config import DEFAULTS


class SessionHygieneConfigTest(unittest.TestCase):
    def test_block_exists_with_expected_keys(self):
        block = DEFAULTS["session_hygiene"]
        self.assertEqual(
            set(block.keys()),
            {"enabled", "idle_clear_minutes", "compact_context_threshold",
             "max_compactions_per_session", "check_interval_s"})

    def test_off_by_default(self):
        self.assertFalse(DEFAULTS["session_hygiene"]["enabled"])

    def test_default_thresholds_match_spec(self):
        block = DEFAULTS["session_hygiene"]
        self.assertEqual(block["idle_clear_minutes"], 60)
        self.assertEqual(block["compact_context_threshold"], 0.60)
        self.assertEqual(block["max_compactions_per_session"], 3)
        self.assertEqual(block["check_interval_s"], 60)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backtalk && python -m pytest tests/test_session_hygiene_config.py -v`
Expected: FAIL with `KeyError: 'session_hygiene'`

- [ ] **Step 3: Add the config block**

In `backtalk/backtalk/config.py`, insert immediately after the `"n9_fallback": { ... },` block (before the `"signals_dir"` comment):

```python
    "session_hygiene": {
        "enabled": False,
        # Minutes of no utterance from any source before an automatic
        # checkpoint + /clear fires.
        "idle_clear_minutes": 60,
        # Fraction of context occupied (0..1) that triggers an
        # automatic checkpoint + /compact.
        "compact_context_threshold": 0.60,
        # After this many automatic compactions in one running process,
        # a further trigger does a full summary + /clear instead of a
        # 4th compact.
        "max_compactions_per_session": 3,
        # How often the watcher checks idle time and context usage.
        "check_interval_s": 60,
    },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backtalk && python -m pytest tests/test_session_hygiene_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd backtalk
git add backtalk/config.py tests/test_session_hygiene_config.py
git commit -m "feat(hygiene): add session_hygiene config defaults, off by default"
```

---

## Task 2: `SessionHygiene` skeleton — activity tracking

**Files:**
- Create: `backtalk/backtalk/hygiene.py`
- Test: `backtalk/tests/test_hygiene_activity.py`

**Interfaces:**
- Produces: `SessionHygiene.__init__(self, cfg: dict)`, `SessionHygiene.mark_activity(self, now: float | None = None)`, `SessionHygiene.seconds_idle(self, now: float | None = None) -> float`. `now` is always `time.monotonic()`-style seconds; every method accepts an optional override so tests never need a real `time.sleep`.

- [ ] **Step 1: Write the failing test**

```python
"""Idle tracking must start from construction (a freshly-launched voice
line isn't "idle" from epoch zero), and mark_activity must reset the
clock every time it's called, from any source."""
import unittest

from backtalk.hygiene import SessionHygiene


class HygieneActivityTest(unittest.TestCase):
    def make(self, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        cfg.update(overrides)
        return SessionHygiene(cfg)

    def test_idle_starts_at_zero_on_construction(self):
        h = self.make()
        self.assertAlmostEqual(h.seconds_idle(now=100.0), 0.0, places=3)

    def test_seconds_idle_grows_with_time(self):
        h = self.make()
        h.mark_activity(now=100.0)
        self.assertAlmostEqual(h.seconds_idle(now=130.0), 30.0, places=3)

    def test_mark_activity_resets_the_clock(self):
        h = self.make()
        h.mark_activity(now=100.0)
        h.mark_activity(now=200.0)
        self.assertAlmostEqual(h.seconds_idle(now=205.0), 5.0, places=3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backtalk && python -m pytest tests/test_hygiene_activity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtalk.hygiene'`

- [ ] **Step 3: Write minimal implementation**

```python
# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Automatic session hygiene for the voice line: an idle voice session
clears itself, a full one compacts itself, both through the same
brain.command() channel the spoken "clear"/"compact" verbs already use
(main.py:1173-1219). See docs/superpowers/specs/
2026-09-23-session-hygiene-watcher-design.md for the design.

Deliberately silent: Sir is very likely not in the room when this
fires, so nothing here calls mouth.say(). Everything goes through
log() only.
"""
import time

from backtalk.vlog import log


class SessionHygiene:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._last_activity = time.monotonic()
        self._compactions_this_session = 0

    def mark_activity(self, now: float | None = None):
        self._last_activity = now if now is not None else time.monotonic()

    def seconds_idle(self, now: float | None = None) -> float:
        now = now if now is not None else time.monotonic()
        return now - self._last_activity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backtalk && python -m pytest tests/test_hygiene_activity.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd backtalk
git add backtalk/hygiene.py tests/test_hygiene_activity.py
git commit -m "feat(hygiene): add SessionHygiene activity tracking"
```

---

## Task 3: Context-occupied-fraction calculation

**Files:**
- Modify: `backtalk/backtalk/hygiene.py`
- Test: `backtalk/tests/test_hygiene_context_fraction.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: module-level `context_occupied_fraction(ctx_usage) -> float | None`, consumed by Task 4's `should_compact` caller (Task 6's `tick`). Mirrors the category-summing logic in `main.py:426-439` (`_spoken_usage`) but returns a raw 0..1 fraction instead of spoken text, and returns `None` (never raises) on anything malformed.

- [ ] **Step 1: Write the failing test**

```python
"""Must match _spoken_usage's own category rules (main.py:426-439):
'free' and 'buffer' categories are excluded from the occupied total.
Total = occupied + the 'Free space' category's own tokens. Anything
malformed must return None, never raise -- a bad context-usage payload
must never crash the watcher loop."""
import unittest

from backtalk.hygiene import context_occupied_fraction


class ContextFractionTest(unittest.TestCase):
    def test_typical_breakdown(self):
        ctx = {"categories": [
            {"name": "System prompt", "tokens": 3000},
            {"name": "Messages", "tokens": 27000},
            {"name": "Free space", "tokens": 60000},
            {"name": "Autocompact buffer", "tokens": 10000},
        ]}
        # occupied = 3000 + 27000 = 30000; total = 30000 + 60000 = 90000
        frac = context_occupied_fraction(ctx)
        self.assertAlmostEqual(frac, 30000 / 90000, places=4)

    def test_object_with_categories_attribute(self):
        class Cat:
            def __init__(self, name, tokens):
                self.name, self.tokens = name, tokens

        class Ctx:
            categories = None

        # Object form uses dicts nested under an attribute, matching
        # the real SDK shape main.py already handles via getattr().
        ctx = Ctx()
        ctx.categories = [{"name": "Messages", "tokens": 40},
                          {"name": "Free space", "tokens": 60}]
        frac = context_occupied_fraction(ctx)
        self.assertAlmostEqual(frac, 40 / 100, places=4)

    def test_none_input_returns_none(self):
        self.assertIsNone(context_occupied_fraction(None))

    def test_missing_categories_returns_none(self):
        self.assertIsNone(context_occupied_fraction({}))

    def test_empty_categories_returns_none(self):
        self.assertIsNone(context_occupied_fraction({"categories": []}))

    def test_no_free_space_category_returns_none(self):
        # Can't compute a fraction without knowing the total.
        ctx = {"categories": [{"name": "Messages", "tokens": 100}]}
        self.assertIsNone(context_occupied_fraction(ctx))

    def test_non_dict_category_entries_are_skipped(self):
        ctx = {"categories": ["garbage", None,
                              {"name": "Messages", "tokens": 10},
                              {"name": "Free space", "tokens": 90}]}
        frac = context_occupied_fraction(ctx)
        self.assertAlmostEqual(frac, 10 / 100, places=4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backtalk && python -m pytest tests/test_hygiene_context_fraction.py -v`
Expected: FAIL with `ImportError: cannot import name 'context_occupied_fraction'`

- [ ] **Step 3: Write minimal implementation**

Add to `backtalk/backtalk/hygiene.py`, below the imports:

```python
def context_occupied_fraction(ctx_usage) -> float | None:
    """0..1 fraction of context occupied, or None if it can't be
    computed. Mirrors _spoken_usage's category rules (main.py:426-439):
    'free' and 'buffer' categories never count as occupied; the total
    is occupied + the 'Free space' category's own tokens. Never
    raises -- a malformed payload must never crash the watcher."""
    try:
        cats = (getattr(ctx_usage, "categories", None)
                or (ctx_usage or {}).get("categories") or [])
    except AttributeError:
        return None
    occupied = 0
    free = 0
    saw_free = False
    for c in cats:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).lower()
        tokens = int(c.get("tokens") or 0)
        if "free" in name:
            free += tokens
            saw_free = True
        elif "buffer" in name:
            continue
        else:
            occupied += tokens
    if not saw_free:
        return None
    total = occupied + free
    if total <= 0:
        return None
    return occupied / total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backtalk && python -m pytest tests/test_hygiene_context_fraction.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd backtalk
git add backtalk/hygiene.py tests/test_hygiene_context_fraction.py
git commit -m "feat(hygiene): compute occupied context fraction from SDK breakdown"
```

---

## Task 4: `should_clear` / `should_compact` decisions

**Files:**
- Modify: `backtalk/backtalk/hygiene.py`
- Test: `backtalk/tests/test_hygiene_decisions.py`

**Interfaces:**
- Consumes: `self.cfg["idle_clear_minutes"]`, `self.cfg["compact_context_threshold"]`, `self.cfg["max_compactions_per_session"]`, `self._compactions_this_session` (Task 2).
- Produces: `SessionHygiene.should_clear(self, idle_s: float) -> bool`, `SessionHygiene.should_compact(self, fraction: float | None) -> bool`, `SessionHygiene.compaction_cap_reached(self) -> bool`, consumed by Task 6's `tick`.

- [ ] **Step 1: Write the failing test**

```python
import unittest

from backtalk.hygiene import SessionHygiene


class HygieneDecisionsTest(unittest.TestCase):
    def make(self, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        cfg.update(overrides)
        return SessionHygiene(cfg)

    def test_should_clear_false_under_threshold(self):
        h = self.make(idle_clear_minutes=60)
        self.assertFalse(h.should_clear(59 * 60))

    def test_should_clear_true_at_threshold(self):
        h = self.make(idle_clear_minutes=60)
        self.assertTrue(h.should_clear(60 * 60))

    def test_should_compact_false_under_threshold(self):
        h = self.make(compact_context_threshold=0.6)
        self.assertFalse(h.should_compact(0.59))

    def test_should_compact_true_at_threshold(self):
        h = self.make(compact_context_threshold=0.6)
        self.assertTrue(h.should_compact(0.6))

    def test_should_compact_false_when_fraction_unknown(self):
        h = self.make()
        self.assertFalse(h.should_compact(None))

    def test_compaction_cap_not_reached_initially(self):
        h = self.make(max_compactions_per_session=3)
        self.assertFalse(h.compaction_cap_reached())

    def test_compaction_cap_reached_after_max(self):
        h = self.make(max_compactions_per_session=3)
        h._compactions_this_session = 3
        self.assertTrue(h.compaction_cap_reached())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backtalk && python -m pytest tests/test_hygiene_decisions.py -v`
Expected: FAIL with `AttributeError: 'SessionHygiene' object has no attribute 'should_clear'`

- [ ] **Step 3: Write minimal implementation**

Add to the `SessionHygiene` class in `backtalk/backtalk/hygiene.py`, below `seconds_idle`:

```python
    def should_clear(self, idle_s: float) -> bool:
        return idle_s >= self.cfg["idle_clear_minutes"] * 60

    def should_compact(self, fraction: float | None) -> bool:
        if fraction is None:
            return False
        return fraction >= self.cfg["compact_context_threshold"]

    def compaction_cap_reached(self) -> bool:
        return (self._compactions_this_session
                >= self.cfg["max_compactions_per_session"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backtalk && python -m pytest tests/test_hygiene_decisions.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd backtalk
git add backtalk/hygiene.py tests/test_hygiene_decisions.py
git commit -m "feat(hygiene): add clear/compact/cap decision logic"
```

---

## Task 5: Checkpoint-then-command primitives

**Files:**
- Modify: `backtalk/backtalk/hygiene.py`
- Test: `backtalk/tests/test_hygiene_commands.py`

**Interfaces:**
- Consumes: a `brain`-shaped object exposing `async reset_turn()` and `async command(cmd: str) -> str` (matches `WarmBrain`, `brain.py:277-338`) — tests use a fake, never the real SDK.
- Produces: `async SessionHygiene.run_clear(self, brain) -> bool`, `async SessionHygiene.run_compact(self, brain) -> bool`, `async SessionHygiene.run_full_summary_and_clear(self, brain) -> bool`, consumed by Task 6's `tick`. Each returns `True` on success, `False` if either call returned an `"error:"`-prefixed string.

- [ ] **Step 1: Write the failing test**

```python
"""run_clear/run_compact/run_full_summary_and_clear must: call
reset_turn() first, send a checkpoint turn, then the actual slash
command, and report failure (without raising) if either call comes
back with brain.command()'s own "error:" sentinel (brain.py:307)."""
import unittest


class FakeBrain:
    def __init__(self, responses=None):
        self.reset_turn_calls = 0
        self.commands = []
        # responses: optional list of canned return values, one per
        # command() call, in order. Defaults to a generic ack for all.
        self._responses = list(responses or [])

    async def reset_turn(self):
        self.reset_turn_calls += 1

    async def command(self, cmd: str) -> str:
        self.commands.append(cmd)
        if self._responses:
            return self._responses.pop(0)
        return "ok"


from backtalk.hygiene import SessionHygiene


class HygieneCommandsTest(unittest.IsolatedAsyncioTestCase):
    def make(self):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        return SessionHygiene(cfg)

    async def test_run_clear_success(self):
        h = self.make()
        brain = FakeBrain()
        ok = await h.run_clear(brain)
        self.assertTrue(ok)
        self.assertEqual(brain.reset_turn_calls, 1)
        self.assertEqual(len(brain.commands), 2)
        self.assertIn("checkpoint", brain.commands[0].lower())
        self.assertEqual(brain.commands[1], "/clear")

    async def test_run_compact_success(self):
        h = self.make()
        brain = FakeBrain()
        ok = await h.run_compact(brain)
        self.assertTrue(ok)
        self.assertEqual(brain.commands[1], "/compact")

    async def test_run_clear_fails_on_checkpoint_error(self):
        h = self.make()
        brain = FakeBrain(responses=["error: the command timed out"])
        ok = await h.run_clear(brain)
        self.assertFalse(ok)
        # Must not have gone on to send /clear after a failed checkpoint.
        self.assertEqual(len(brain.commands), 1)

    async def test_run_compact_fails_on_command_error(self):
        h = self.make()
        brain = FakeBrain(responses=["ok", "error: the command timed out"])
        ok = await h.run_compact(brain)
        self.assertFalse(ok)
        self.assertEqual(len(brain.commands), 2)

    async def test_run_full_summary_and_clear_success(self):
        h = self.make()
        brain = FakeBrain()
        ok = await h.run_full_summary_and_clear(brain)
        self.assertTrue(ok)
        self.assertEqual(brain.commands[1], "/clear")
        self.assertIn("summary", brain.commands[0].lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backtalk && python -m pytest tests/test_hygiene_commands.py -v`
Expected: FAIL with `AttributeError: 'SessionHygiene' object has no attribute 'run_clear'`

- [ ] **Step 3: Write minimal implementation**

Add to `backtalk/backtalk/hygiene.py`, below the imports (module-level constants) and inside the class (methods):

```python
_CHECKPOINT_PROMPT = (
    "Per your memory discipline in CLAUDE.md, checkpoint current "
    "session state to the vault now -- today's daily note and any "
    "note whose contextual home this session touched. This is an "
    "automatic session-hygiene checkpoint, not a request from Sir.")

_FULL_SUMMARY_PROMPT = (
    "Per your memory discipline in CLAUDE.md, this session has hit "
    "its automatic compaction cap. Write a full session summary to "
    "today's daily note and every relevant vault note, the same way "
    "you would after a third manual compaction, then confirm when "
    "done. This is an automatic session-hygiene checkpoint, not a "
    "request from Sir.")
```

```python
    async def _checkpoint_then(self, brain, checkpoint_prompt: str,
                                slash_cmd: str) -> bool:
        await brain.reset_turn()
        resp = await brain.command(checkpoint_prompt)
        if resp.startswith("error:"):
            log(f"[hygiene] checkpoint failed before {slash_cmd}: {resp}")
            return False
        resp = await brain.command(slash_cmd)
        if resp.startswith("error:"):
            log(f"[hygiene] {slash_cmd} failed: {resp}")
            return False
        return True

    async def run_clear(self, brain) -> bool:
        return await self._checkpoint_then(brain, _CHECKPOINT_PROMPT,
                                           "/clear")

    async def run_compact(self, brain) -> bool:
        return await self._checkpoint_then(brain, _CHECKPOINT_PROMPT,
                                           "/compact")

    async def run_full_summary_and_clear(self, brain) -> bool:
        return await self._checkpoint_then(brain, _FULL_SUMMARY_PROMPT,
                                           "/clear")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backtalk && python -m pytest tests/test_hygiene_commands.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd backtalk
git add backtalk/hygiene.py tests/test_hygiene_commands.py
git commit -m "feat(hygiene): add checkpoint-then-command primitives"
```

---

## Task 6: `tick()` — the per-cycle decision and action

**Files:**
- Modify: `backtalk/backtalk/hygiene.py`
- Test: `backtalk/tests/test_hygiene_tick.py`

**Interfaces:**
- Consumes: `should_clear`/`should_compact`/`compaction_cap_reached` (Task 4), `run_clear`/`run_compact`/`run_full_summary_and_clear` (Task 5, mocked in tests via `unittest.mock.AsyncMock`), a `turn_lock`-shaped object exposing `is_active() -> bool` (matches `satellites.TurnLock`), an `is_online_fn: Callable[[], bool]`.
- Produces: `async SessionHygiene.tick(self, brain, turn_lock, is_online_fn)`, consumed by Task 7's `watch()`.

- [ ] **Step 1: Write the failing test**

```python
"""tick() owns all the gating: never fire mid-turn, never fire
offline, never a 4th compact once the cap is hit, and a failed
run_* must not increment the compaction counter."""
import unittest
from unittest.mock import AsyncMock

from backtalk.hygiene import SessionHygiene


class FakeLock:
    def __init__(self, active=False):
        self._active = active

    def is_active(self):
        return self._active


class HygieneTickTest(unittest.IsolatedAsyncioTestCase):
    def make(self, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        cfg.update(overrides)
        return SessionHygiene(cfg)

    async def test_skips_when_turn_active(self):
        h = self.make(idle_clear_minutes=0)  # idle threshold already met
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=True),
                     is_online_fn=lambda: True)
        h.run_clear.assert_not_called()

    async def test_skips_when_offline(self):
        h = self.make(idle_clear_minutes=0)
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: False)
        h.run_clear.assert_not_called()

    async def test_idle_triggers_clear_and_resets_clock(self):
        h = self.make(idle_clear_minutes=0)
        h.run_clear = AsyncMock(return_value=True)
        h.get_context_fraction = AsyncMock(return_value=0.1)
        idle_before = h.seconds_idle()
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        h.run_clear.assert_awaited_once()
        self.assertLess(h.seconds_idle(), idle_before)

    async def test_context_triggers_compact_and_increments_counter(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5)
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_compact = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        h.run_compact.assert_awaited_once()
        self.assertEqual(h._compactions_this_session, 1)

    async def test_failed_compact_does_not_increment_counter(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5)
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_compact = AsyncMock(return_value=False)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        self.assertEqual(h._compactions_this_session, 0)

    async def test_cap_reached_uses_heavier_path_not_a_4th_compact(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5,
                      max_compactions_per_session=3)
        h._compactions_this_session = 3
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_compact = AsyncMock(return_value=True)
        h.run_full_summary_and_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        h.run_compact.assert_not_called()
        h.run_full_summary_and_clear.assert_awaited_once()

    async def test_neither_threshold_met_does_nothing(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.99)
        h.get_context_fraction = AsyncMock(return_value=0.1)
        h.run_clear = AsyncMock(return_value=True)
        h.run_compact = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        h.run_clear.assert_not_called()
        h.run_compact.assert_not_called()

    async def test_a_tick_exception_is_swallowed_and_logged(self):
        h = self.make(idle_clear_minutes=0)
        h.run_clear = AsyncMock(side_effect=RuntimeError("boom"))
        # Must not raise out of tick().
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backtalk && python -m pytest tests/test_hygiene_tick.py -v`
Expected: FAIL with `AttributeError: 'SessionHygiene' object has no attribute 'tick'`

- [ ] **Step 3: Write minimal implementation**

Add to `backtalk/backtalk/hygiene.py`. First, a thin real `get_context_fraction` (tests above replace it with an `AsyncMock`, so its own body is exercised only in Task 9's integration check):

```python
    async def get_context_fraction(self, brain) -> float | None:
        ctx = await brain.context_usage()
        return context_occupied_fraction(ctx)

    async def tick(self, brain, turn_lock, is_online_fn):
        try:
            if turn_lock.is_active() or not is_online_fn():
                return
            idle_s = self.seconds_idle()
            if self.should_clear(idle_s):
                log(f"[hygiene] idle {idle_s / 60:.0f}min, no activity "
                    "-- checkpointing and clearing")
                if await self.run_clear(brain):
                    self.mark_activity()
                return
            fraction = await self.get_context_fraction(brain)
            if self.should_compact(fraction):
                if self.compaction_cap_reached():
                    log(f"[hygiene] context at {fraction:.0%}, compaction "
                        "cap hit -- full summary and clear instead")
                    await self.run_full_summary_and_clear(brain)
                    return
                log(f"[hygiene] context at {fraction:.0%} "
                    f"({self._compactions_this_session}/"
                    f"{self.cfg['max_compactions_per_session']} "
                    "compactions this session) -- checkpointing "
                    "and compacting")
                if await self.run_compact(brain):
                    self._compactions_this_session += 1
        except Exception as e:
            log(f"[hygiene] tick failed: {e!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backtalk && python -m pytest tests/test_hygiene_tick.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
cd backtalk
git add backtalk/hygiene.py tests/test_hygiene_tick.py
git commit -m "feat(hygiene): add tick() gating, decision and action logic"
```

---

## Task 7: `watch()` loop

**Files:**
- Modify: `backtalk/backtalk/hygiene.py`
- Test: `backtalk/tests/test_hygiene_watch.py`

**Interfaces:**
- Consumes: `tick()` (Task 6, mocked here), `self.cfg["check_interval_s"]`.
- Produces: `async SessionHygiene.watch(self, brain, turn_lock, is_online_fn)`, consumed by Task 9's `amain()` wiring. Runs until cancelled.

- [ ] **Step 1: Write the failing test**

```python
"""watch() is a thin loop: sleep check_interval_s, call tick(), repeat,
until cancelled. Uses a real short interval and real (tiny) sleeps --
matches this repo's existing testing style (see test_ptt_max_hold.py)
rather than a fake clock."""
import asyncio
import unittest
from unittest.mock import AsyncMock

from backtalk.hygiene import SessionHygiene


class HygieneWatchTest(unittest.IsolatedAsyncioTestCase):
    def make(self, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 0.05}
        cfg.update(overrides)
        return SessionHygiene(cfg)

    async def test_calls_tick_on_each_interval(self):
        h = self.make()
        h.tick = AsyncMock(return_value=None)
        task = asyncio.create_task(
            h.watch(brain=object(), turn_lock=object(),
                    is_online_fn=lambda: True))
        await asyncio.sleep(0.17)   # ~3 intervals
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self.assertGreaterEqual(h.tick.await_count, 2)

    async def test_cancellation_stops_the_loop_cleanly(self):
        h = self.make()
        h.tick = AsyncMock(return_value=None)
        task = asyncio.create_task(
            h.watch(brain=object(), turn_lock=object(),
                    is_online_fn=lambda: True))
        await asyncio.sleep(0.06)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backtalk && python -m pytest tests/test_hygiene_watch.py -v`
Expected: FAIL with `AttributeError: 'SessionHygiene' object has no attribute 'watch'`

- [ ] **Step 3: Write minimal implementation**

Add to `backtalk/backtalk/hygiene.py`:

```python
    async def watch(self, brain, turn_lock, is_online_fn):
        """Runs until cancelled -- amain() cancels this task in its
        existing shutdown finally: block (main.py:1644-1654)."""
        interval = self.cfg["check_interval_s"]
        while True:
            await asyncio.sleep(interval)
            await self.tick(brain, turn_lock, is_online_fn)
```

Add `import asyncio` to the top of `backtalk/backtalk/hygiene.py` alongside the existing `import time`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backtalk && python -m pytest tests/test_hygiene_watch.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd backtalk
git add backtalk/hygiene.py tests/test_hygiene_watch.py
git commit -m "feat(hygiene): add watch() loop"
```

---

## Task 8: Wire activity tracking into `handle()`

**Files:**
- Modify: `backtalk/backtalk/main.py:1299` (top of `handle()`)

**Interfaces:**
- Consumes: a `hygiene: SessionHygiene` instance already constructed in `amain()` (Task 9 — this task and Task 9 are ordered together deliberately: Task 8's edit is inert without Task 9's construction, but each is independently reviewable, and Task 9 depends on the `hygiene` name this task assumes).

No automated test: `handle()` is a large, already-tested-by-hand integration point with no existing unit-test scaffold (confirmed — `backtalk/tests/` and `backtalk/backtalk/tests/` have no test that constructs `handle()`'s full closure). Verified manually instead, in Task 10's end-to-end pass.

- [ ] **Step 1: Add the hook**

In `backtalk/backtalk/main.py`, inside `async def handle(text, spoke_from=None, source="local") -> bool:`, as the very first line of the function body (before `nonlocal speak_task, turn_epoch`):

```python
    async def handle(text: str, spoke_from: float | None = None, source="local") -> bool:
        """Process one utterance; returns False on quit. spoke_from is
        when the utterance STARTED (the PTT press), so an answer can be
        told apart from speech that began before the ask even existed."""
        hygiene.mark_activity()
        nonlocal speak_task, turn_epoch
```

- [ ] **Step 2: Commit**

```bash
cd backtalk
git add backtalk/main.py
git commit -m "feat(hygiene): reset the idle clock on every utterance"
```

---

## Task 9: Wire `SessionHygiene` construction and lifecycle into `amain()`

**Files:**
- Modify: `backtalk/backtalk/main.py` (near `amain()`'s setup, ~line 991, and its shutdown `finally:`, ~line 1644)

**Interfaces:**
- Produces: the `hygiene` name Task 8 assumes; a background task cancelled alongside `brain`/`connectivity` shutdown.

- [ ] **Step 1: Construct it and start the watcher**

In `backtalk/backtalk/main.py`, in `amain()`, immediately after the existing `n9_brain = N9Brain(...)` block (~line 995):

```python
    from backtalk.hygiene import SessionHygiene
    hygiene = SessionHygiene(CFG.get("session_hygiene", {}))
    hygiene_task = None
    if CFG.get("session_hygiene", {}).get("enabled"):
        hygiene_task = asyncio.create_task(
            hygiene.watch(brain, turn_lock,
                          lambda: connectivity.is_online))
        log("[backtalk] session hygiene watcher started")
```

- [ ] **Step 2: Stop it on shutdown**

In `backtalk/backtalk/main.py`, in `amain()`'s existing `finally:` block (~line 1644-1654), add the cancel alongside the existing cleanup, before `await brain.stop()`:

```python
    finally:
        _MIC["gen"] += 1     # abort any live open-mic capture promptly
        if speak_task and not speak_task.done():
            speak_task.cancel()
        if hygiene_task and not hygiene_task.done():
            hygiene_task.cancel()
        mouth.shutdown()  # restores the music on Ctrl-C / crash paths too
        signals.static_stop()
        signals.set_state("idle")
        await brain.stop()
        if lf_cfg.get("enabled"):
            await connectivity.stop()
        log("[backtalk] hung up")
```

- [ ] **Step 3: Manual smoke check**

With `session_hygiene.enabled: false` (the shipped default) in `backtalk.json`, run `uv run python -m backtalk.main` and confirm the log does **not** print `"session hygiene watcher started"` and the voice line otherwise behaves exactly as before (say a phrase, confirm a normal reply). This is the "off by default ships safe" check from Global Constraints.

- [ ] **Step 4: Commit**

```bash
cd backtalk
git add backtalk/main.py
git commit -m "feat(hygiene): wire SessionHygiene into amain(), off by default"
```

---

## Task 10: End-to-end manual verification with the feature turned on

**Files:** none (verification only — this task produces no diff unless a real bug is found, in which case fix it in the file it belongs to and fold that fix into the relevant earlier task's commit before moving on).

- [ ] **Step 1: Idle-clear fires and doesn't interrupt a live turn**

In a local, non-committed `backtalk.json` override, set `"session_hygiene": {"enabled": true, "idle_clear_minutes": 0.02, "check_interval_s": 5, ...other defaults}` (0.02 min ≈ 1.2s). Start the voice line, say nothing, and confirm within ~10s the log shows `[hygiene] idle 0min, no activity -- checkpointing and clearing`, followed by real checkpoint/clear activity. Then repeat with a real conversation running continuously across two check intervals and confirm it is *not* interrupted.

- [ ] **Step 2: Context-compact fires and respects the cap**

With the same fast `check_interval_s`, set `"compact_context_threshold": 0.01` so any real conversation trips it almost immediately. Have three short exchanges, confirm three `[hygiene] ... checkpointing and compacting` log lines with the counter incrementing (`1/3`, `2/3`, `3/3`), then a fourth exchange that should instead log `compaction cap hit -- full summary and clear instead`.

- [ ] **Step 3: Offline skip**

Disable the network adapter (or otherwise force `connectivity.is_online` false) during an idle window with the fast timings from Step 1; confirm no `[hygiene]` trigger fires while offline, and that it resumes correctly once reconnected.

- [ ] **Step 4: Manual verbs unaffected**

With `session_hygiene.enabled: true` and default (slow) thresholds, say "clear" and "compact" out loud and confirm both still work exactly as before (spoken confirmations, same behavior as pre-this-plan) — this plan adds a second caller of `brain.command()`, not a replacement.

- [ ] **Step 5: Restore defaults and commit the config file unchanged**

Revert any local `backtalk.json` threshold overrides used for testing — `session_hygiene.enabled` must ship `false` in the committed config. Confirm with `git diff backtalk.json` showing no changes (or only the intended default block from Task 1).

- [ ] **Step 6: Final commit**

```bash
cd backtalk
git add -A
git commit -m "test(hygiene): verify idle-clear, context-compact cap, offline skip end to end" --allow-empty
```
