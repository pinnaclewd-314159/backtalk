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
"""The satellite ear+mouth — a Wyoming-protocol TCP listener that lets
jarvis-satellite hardware (or any Wyoming-speaking satellite) join the
same conversation PTT/typed input already use.

Design: docs/superpowers/specs/2026-09-03-wyoming-listener-design.md

Turn-lock rules (see TurnLock): local input always wins; a satellite can
only interrupt itself; anything else arriving mid-turn is dropped, never
queued — matching the firmware's own "drop, return to idle, never block"
principle on its side (jarvis-satellite/main/wyoming_client.c).
"""
import asyncio

import numpy as np

from backtalk.vlog import log

WIRE_RATE = 16000  # satellite mic/speaker rate, fixed by the firmware


class TurnLock:
    """Tracks which source (the string "local", or a SatelliteConnection)
    currently owns the active turn. Not thread-safe by design — every
    caller runs on the single asyncio event loop backtalk already uses,
    same as every other piece of shared state in main.py."""

    def __init__(self):
        self._owner = None

    def current_owner(self):
        return self._owner

    def is_active(self) -> bool:
        return self._owner is not None

    def try_acquire(self, owner) -> bool:
        """True and takes ownership if allowed to proceed: no active turn,
        local input (always wins), or this is the same owner re-triggering
        itself. False (and no state change) means: drop this utterance."""
        if owner == "local" or self._owner is None or self._owner == owner:
            self._owner = owner
            return True
        return False

    def release(self, owner) -> None:
        """Clears ownership if the caller actually holds it (a stale
        release from an already-superseded turn must not clobber a NEWER
        turn that has since acquired the lock)."""
        if self._owner == owner:
            self._owner = None


def resample_pcm(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """int16 mono PCM at from_rate -> int16 mono PCM at to_rate, via plain
    linear interpolation. No scipy/librosa in this project (ears.py and
    mouth.py don't use one either) -- this is voice-quality resampling,
    not hi-fi, and linear interpolation is more than sufficient for
    speech intelligibility over a small speaker.
    """
    if from_rate == to_rate or len(pcm) == 0:
        return pcm.astype(np.int16)
    duration_s = len(pcm) / from_rate
    n_out = max(1, int(round(duration_s * to_rate)))
    x_old = np.linspace(0.0, duration_s, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, duration_s, num=n_out, endpoint=False)
    resampled = np.interp(x_new, x_old, pcm.astype(np.float64))
    return np.clip(resampled, -32768, 32767).astype(np.int16)


if __name__ == "__main__":
    # Manual self-test, same convention as ears.py/mouth.py's own
    # __main__ blocks -- this project has no test framework.
    lock = TurnLock()
    assert lock.try_acquire("local") is True
    assert lock.current_owner() == "local"
    assert lock.try_acquire("sat_a") is False, "local turn must not be stolen"
    lock.release("local")
    assert lock.current_owner() is None
    assert lock.try_acquire("sat_a") is True
    assert lock.try_acquire("sat_b") is False, "a different satellite must be dropped"
    assert lock.try_acquire("sat_a") is True, "the SAME satellite may re-trigger itself"
    assert lock.try_acquire("local") is True, "local always wins, even over a satellite"
    lock.release("sat_a")  # stale release from the superseded turn
    assert lock.current_owner() == "local", "a stale release must not clobber a newer owner"
    lock.release("local")
    assert lock.current_owner() is None
    print("[satellites] TurnLock self-test: PASS")

    tone = (np.sin(2 * np.pi * 440 * np.arange(2400) / 24000) * 10000).astype(np.int16)
    down = resample_pcm(tone, 24000, 16000)
    assert len(down) == 1600, f"expected 1600 samples, got {len(down)}"
    assert down.dtype == np.int16
    same = resample_pcm(tone, 24000, 24000)
    assert np.array_equal(same, tone), "same-rate resample must be a no-op"
    print("[satellites] resample_pcm self-test: PASS")
