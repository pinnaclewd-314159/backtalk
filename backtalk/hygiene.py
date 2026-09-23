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
