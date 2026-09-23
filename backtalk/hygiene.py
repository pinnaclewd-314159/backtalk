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
import asyncio
import time

from backtalk.vlog import log

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


class SessionHygiene:
    def __init__(self, cfg: dict, now: float | None = None):
        self.cfg = cfg
        self._last_activity = now if now is not None else time.monotonic()
        self._compactions_this_session = 0

    def mark_activity(self, now: float | None = None):
        self._last_activity = now if now is not None else time.monotonic()

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

    async def watch(self, brain, turn_lock, is_online_fn):
        """Runs until cancelled -- amain() cancels this task in its
        existing shutdown finally: block (main.py:1644-1654)."""
        interval = self.cfg["check_interval_s"]
        while True:
            await asyncio.sleep(interval)
            await self.tick(brain, turn_lock, is_online_fn)
