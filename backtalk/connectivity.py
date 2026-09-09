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
    already know the boot outcome (main.py's amain()) rely on this to
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


if __name__ == "__main__":
    # Quick self-test / smoke check, not a full test suite.
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
