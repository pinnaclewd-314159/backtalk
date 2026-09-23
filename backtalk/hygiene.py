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

Final-review fix pass (2026-09-23): a hygiene cycle runs as a
cancellable task (see preempt()) because WarmBrain has ONE shared SDK
message stream -- a real user turn arriving mid-cycle must interrupt
it before handle()'s own reset_turn()/command() calls touch that same
stream, or both sides read garbled, interleaved messages (the same
"off-by-one bug" class brain.reset_turn's own docstring describes).
"""
import asyncio
import time

from backtalk.vlog import log

_CHECKPOINT_PROMPT = (
    "Per your memory discipline in CLAUDE.md, checkpoint current "
    "session state to the vault now -- today's daily note and any "
    "note whose contextual home this session touched. This is an "
    "automatic session-hygiene checkpoint, not a request from Sir. "
    "This session has already been active -- skip your normal "
    "startup sequence (VAULT-INDEX.md, priorities, reminders) and go "
    "straight to writing the checkpoint.")

_FULL_SUMMARY_PROMPT = (
    "Per your memory discipline in CLAUDE.md, this session has hit "
    "its automatic compaction cap. Write a full session summary to "
    "today's daily note and every relevant vault note, the same way "
    "you would after a third manual compaction, then confirm when "
    "done. This is an automatic session-hygiene checkpoint, not a "
    "request from Sir. This session has already been active -- skip "
    "your normal startup sequence (VAULT-INDEX.md, priorities, "
    "reminders) and go straight to writing the summary.")

# Backoff after a failed cycle: doubles each consecutive failure,
# capped so a persistent problem still gets retried eventually rather
# than being abandoned forever.
_BACKOFF_BASE_S = 60.0
_BACKOFF_MAX_S = 1800.0


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
        try:
            tokens = int(c.get("tokens") or 0)
        except (TypeError, ValueError):
            continue
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


class SessionHygiene:
    def __init__(self, cfg: dict, now: float | None = None):
        self.cfg = cfg
        self._last_activity = now if now is not None else time.monotonic()
        self._compactions_this_session = 0
        # Important #3: nothing has happened since construction (which
        # counts as "since the last clear") -- an idle-clear on an
        # already-empty session has nothing to do.
        self._has_activity_since_clear = False
        # Important #4: backoff state after a failed cycle.
        self._backoff_until = 0.0
        self._consecutive_failures = 0
        # Critical #2: the currently in-flight checkpoint/command
        # cycle, if any -- so a real user turn can preempt() it.
        self._cycle_task: asyncio.Task | None = None

    def mark_activity(self, now: float | None = None):
        self._last_activity = now if now is not None else time.monotonic()
        self._has_activity_since_clear = True

    def _reset_after_clear(self, now: float | None = None):
        """Like mark_activity(), but for OUR OWN successful clear --
        must NOT re-arm _has_activity_since_clear, or an idle-clear
        would immediately look eligible to fire again next tick."""
        self._last_activity = now if now is not None else time.monotonic()
        self._has_activity_since_clear = False

    def _record_failure(self):
        self._consecutive_failures += 1
        delay = min(_BACKOFF_BASE_S * (2 ** (self._consecutive_failures - 1)),
                    _BACKOFF_MAX_S)
        self._backoff_until = time.monotonic() + delay

    def _record_success(self):
        self._consecutive_failures = 0
        self._backoff_until = 0.0

    def seconds_idle(self, now: float | None = None) -> float:
        now = now if now is not None else time.monotonic()
        return now - self._last_activity

    def should_clear(self, idle_s: float) -> bool:
        return idle_s >= self.cfg["idle_clear_minutes"] * 60

    def should_compact(self, fraction: float | None) -> bool:
        if fraction is None:
            return False
        return fraction >= self.cfg["compact_context_threshold"]

    def compaction_cap_reached(self) -> bool:
        return (self._compactions_this_session
                >= self.cfg["max_compactions_per_session"])

    async def _checkpoint_then(self, brain, checkpoint_prompt: str,
                                slash_cmd: str) -> bool:
        await brain.reset_turn()
        resp = await brain.command(checkpoint_prompt)
        if not resp or resp.startswith("error:"):
            log(f"[hygiene] checkpoint failed before {slash_cmd}: {resp!r}")
            return False
        resp = await brain.command(slash_cmd)
        if not resp or resp.startswith("error:"):
            log(f"[hygiene] {slash_cmd} failed: {resp!r}")
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

    async def get_context_fraction(self, brain) -> float | None:
        ctx = await brain.context_usage()
        return context_occupied_fraction(ctx)

    async def _run_cycle(self, coro) -> bool | None:
        """Run a checkpoint-then-command coroutine as a cancellable
        task, so preempt() can interrupt it before it shares the SDK's
        single message stream with a real user turn. Returns None if
        preempted (neither success nor failure -- retry later, don't
        count it as either), the coroutine's own bool result
        otherwise."""
        self._cycle_task = asyncio.create_task(coro)
        try:
            return await self._cycle_task
        except asyncio.CancelledError:
            log("[hygiene] cycle preempted by a real turn")
            return None
        finally:
            self._cycle_task = None

    async def preempt(self):
        """Cancel any in-flight hygiene cycle and wait for it to
        unwind. Call this from handle() before anything else touches
        the brain, so a hygiene checkpoint/command call is never still
        reading the shared SDK stream when a real turn starts reading
        it too. A no-op when no cycle is running."""
        if self._cycle_task and not self._cycle_task.done():
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass

    async def tick(self, brain, turn_lock, is_online_fn, is_autoapprove_fn):
        try:
            if (turn_lock.is_active() or not is_online_fn()
                    or not is_autoapprove_fn()):
                return
            now = time.monotonic()
            if now < self._backoff_until:
                return
            # Some brains (fakes, bare objects in gating tests) don't
            # define quota_exhausted() at all -- treat that as "not
            # exhausted" rather than crashing.
            if getattr(brain, "quota_exhausted", lambda: False)():
                return

            idle_s = self.seconds_idle()
            if self.should_clear(idle_s) and self._has_activity_since_clear:
                log(f"[hygiene] idle {idle_s / 60:.0f}min, no activity "
                    "-- checkpointing and clearing")
                result = await self._run_cycle(self.run_clear(brain))
                if result:
                    self._reset_after_clear()
                    self._compactions_this_session = 0
                    self._record_success()
                elif result is False:
                    self._record_failure()
                return

            fraction = await self.get_context_fraction(brain)
            if self.should_compact(fraction):
                if self.compaction_cap_reached():
                    log(f"[hygiene] context at {fraction:.0%}, compaction "
                        "cap hit -- full summary and clear instead")
                    result = await self._run_cycle(
                        self.run_full_summary_and_clear(brain))
                    if result:
                        self._reset_after_clear()
                        self._compactions_this_session = 0
                        self._record_success()
                    elif result is False:
                        self._record_failure()
                    return
                log(f"[hygiene] context at {fraction:.0%} "
                    f"({self._compactions_this_session}/"
                    f"{self.cfg['max_compactions_per_session']} "
                    "compactions this session) -- checkpointing "
                    "and compacting")
                result = await self._run_cycle(self.run_compact(brain))
                if result:
                    self._compactions_this_session += 1
                    self._record_success()
                elif result is False:
                    self._record_failure()
        except Exception as e:
            log(f"[hygiene] tick failed: {e!r}")

    async def watch(self, brain, turn_lock, is_online_fn, is_autoapprove_fn):
        """Runs until cancelled -- amain() cancels this task in its
        existing shutdown finally: block (main.py:1644-1654)."""
        interval = self.cfg["check_interval_s"]
        while True:
            await asyncio.sleep(interval)
            await self.tick(brain, turn_lock, is_online_fn, is_autoapprove_fn)
