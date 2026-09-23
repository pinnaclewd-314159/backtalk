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
    def __init__(self, cfg: dict, now: float | None = None):
        self.cfg = cfg
        self._last_activity = now if now is not None else time.monotonic()
        self._compactions_this_session = 0

    def mark_activity(self, now: float | None = None):
        self._last_activity = now if now is not None else time.monotonic()

    def seconds_idle(self, now: float | None = None) -> float:
        now = now if now is not None else time.monotonic()
        return now - self._last_activity
