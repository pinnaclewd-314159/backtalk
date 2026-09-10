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

The poll runs on its own OS thread, deliberately NOT as an asyncio task on
the main loop. The 2026-09-09 outage test showed why: a hung WarmBrain
call (one that goes silent instead of raising) can sit inside a single
await for minutes, and while it does, nothing else scheduled on that same
loop gets a turn — including an asyncio-based poller, which meant zero
offline detections fired for the entire nine-minute outage. A plain thread
with a blocking `httpx.Client` call keeps ticking regardless of what the
conversation's own coroutine is doing.

`force_offline()` is the reactive half: a real failed turn against
WarmBrain is stronger evidence than an indirect poll, so it flips
immediately and bypasses the failure side of the hysteresis. The success
side (switching back) is untouched — still needs `threshold` consecutive
clean polls, since a turn succeeding once during a flaky recovery isn't
the same guarantee.

See backtalk/docs/superpowers/specs/2026-09-09-offline-fallback-design.md.
"""
import asyncio
import threading

import httpx

from backtalk.vlog import log

_STATE = {"online": True, "consec_ok": 0, "consec_fail": 0}
_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None
_on_change = None
_loop: asyncio.AbstractEventLoop | None = None
_URL = ""
_INTERVAL_S = 20.0
_TIMEOUT_S = 4.0
_THRESHOLD = 2


def is_online() -> bool:
    return _STATE["online"]


def _probe() -> bool:
    try:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            client.get(_URL)
        return True
    except Exception:
        return False


def _flip(online: bool):
    """Updates state immediately — so is_online() reflects it right away
    even if the main event loop is currently stalled on something else —
    then hands the async on_change callback to that loop via
    run_coroutine_threadsafe. Safe to call from the poll thread or from
    the main loop's own thread (force_offline())."""
    _STATE["online"] = online
    log(f"[connectivity] {'back online' if online else 'offline'}")
    if _on_change and _loop:
        try:
            asyncio.run_coroutine_threadsafe(_on_change(online), _loop)
        except Exception as e:
            log(f"[connectivity] on_change scheduling failed: {e!r}")


def _poll_loop():
    while not _stop_event.wait(_INTERVAL_S):
        if _probe():
            _STATE["consec_fail"] = 0
            _STATE["consec_ok"] += 1
            if not _STATE["online"] and _STATE["consec_ok"] >= _THRESHOLD:
                _flip(True)
        else:
            _STATE["consec_ok"] = 0
            _STATE["consec_fail"] += 1
            if _STATE["online"] and _STATE["consec_fail"] >= _THRESHOLD:
                _flip(False)


def start(url: str, on_change=None, interval_s: float = 20.0,
          timeout_s: float = 4.0, threshold: int = 2,
          initial_online: bool = True,
          loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Begin the background poll thread. `initial_online` sets the
    starting state directly with NO on_change callback fired — on_change
    only fires for state changes the poll loop (or force_offline) detects
    going forward, never for this initial assignment. Callers that
    already know the boot outcome (main.py's amain()) rely on this to
    avoid a duplicate/racing notification right at startup.

    `loop` is the asyncio loop `on_change` (a coroutine function) gets
    scheduled onto — pass the running loop from amain() via
    asyncio.get_event_loop(). Required if `on_change` is given."""
    global _thread, _stop_event, _on_change, _loop, _URL
    global _INTERVAL_S, _TIMEOUT_S, _THRESHOLD
    _URL, _on_change, _loop = url, on_change, loop
    _INTERVAL_S, _TIMEOUT_S, _THRESHOLD = interval_s, timeout_s, threshold
    _STATE["online"] = initial_online
    _STATE["consec_ok"] = _STATE["consec_fail"] = 0
    _stop_event = threading.Event()
    _thread = threading.Thread(target=_poll_loop, daemon=True,
                                name="connectivity-poll")
    _thread.start()


async def stop() -> None:
    global _thread, _stop_event
    if _stop_event:
        _stop_event.set()
    if _thread:
        _thread.join(timeout=2)
        _thread = None
    _stop_event = None


def force_offline() -> None:
    """Reactive override — call this the moment a real turn against
    WarmBrain fails/times out. Safe to call synchronously from the main
    event loop's thread (the common case) or any other thread."""
    _STATE["consec_ok"] = 0
    _STATE["consec_fail"] = _THRESHOLD
    if _STATE["online"]:
        _flip(False)


if __name__ == "__main__":
    # Quick self-test / smoke check, not a full test suite.
    async def _run():
        calls = []

        async def on_change(online):
            calls.append(online)

        loop = asyncio.get_running_loop()

        global _probe
        real_probe = _probe

        _probe = lambda: False
        start("http://example.invalid", on_change=on_change,
              interval_s=0.02, timeout_s=0.02, threshold=2, loop=loop)
        assert is_online() is True, "should start online by default"

        await asyncio.sleep(0.03)   # 1 failed poll: not enough yet
        assert is_online() is True, "one failure must not flip it"
        await asyncio.sleep(0.05)   # 2nd failed poll: should flip
        assert is_online() is False, "two consecutive failures should flip offline"
        await asyncio.sleep(0.02)   # let the scheduled coroutine run
        assert calls == [False], f"on_change should have fired once with False, got {calls}"

        _probe = lambda: True
        await asyncio.sleep(0.03)
        assert is_online() is False, "one success must not flip it back yet"
        await asyncio.sleep(0.05)
        assert is_online() is True, "two consecutive successes should flip back online"
        await asyncio.sleep(0.02)
        assert calls == [False, True], f"on_change should have fired twice, got {calls}"

        await stop()
        _probe = real_probe

        # force_offline bypasses the failure hysteresis
        start("http://example.invalid", on_change=on_change, interval_s=999,
              loop=loop)
        calls.clear()
        force_offline()
        await asyncio.sleep(0.02)   # let the scheduled coroutine run
        assert is_online() is False, "force_offline should flip immediately"
        assert calls == [False]
        await stop()
        print("connectivity self-test: OK")

    asyncio.run(_run())
