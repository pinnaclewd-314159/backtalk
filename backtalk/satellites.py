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
import json
from dataclasses import dataclass, field

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


@dataclass
class SatelliteConnection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    name: str
    _buffer: bytearray = field(default_factory=bytearray)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


async def _read_message(reader: asyncio.StreamReader) -> dict | None:
    """One Wyoming JSON line -> dict, or None on a clean EOF/disconnect.
    A malformed line is logged and skipped (returns an empty dict, which
    callers treat as a no-op) rather than tearing down the connection --
    Wyoming is a simple line protocol and one bad message shouldn't be
    fatal."""
    line = await reader.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log(f"[satellites] malformed message, skipping: {e}")
        return {}


async def handle_connection(reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter,
                            on_utterance) -> SatelliteConnection:
    """Runs for the life of one satellite's TCP connection. Parses
    inbound audio-start/audio-chunk/audio-stop, and calls on_utterance
    once per completed utterance. Returns the SatelliteConnection so the
    caller (start_server, Task 4) can register/unregister it."""
    peer = writer.get_extra_info("peername")
    name = f"{peer[0]}:{peer[1]}" if peer else "unknown"
    conn = SatelliteConnection(reader=reader, writer=writer, name=name)
    log(f"[satellites] {name} connected")
    try:
        while True:
            msg = await _read_message(reader)
            if msg is None:
                break
            msg_type = msg.get("type")
            if msg_type == "detect":
                names = (msg.get("data") or {}).get("names")
                log(f"[satellites] {name} detect: {names}")
            elif msg_type == "audio-start":
                conn._buffer = bytearray()
            elif msg_type == "audio-chunk":
                try:
                    payload_len = int((msg.get("payload_length")) or 0)
                    payload = await reader.readexactly(payload_len) if payload_len else b""
                except (ValueError, TypeError) as e:
                    log(f"[satellites] {name} malformed message (skipped): {e}")
                    conn._buffer = bytearray()  # discard any partial utterance
                    continue
                conn._buffer.extend(payload)
            elif msg_type == "audio-stop":
                try:
                    pcm = np.frombuffer(bytes(conn._buffer), dtype=np.int16)
                except ValueError as e:
                    log(f"[satellites] {name} malformed message (skipped): {e}")
                    conn._buffer = bytearray()  # discard any partial utterance
                    continue
                conn._buffer = bytearray()
                # on_utterance callback exceptions are NOT caught here -- they
                # propagate normally, not mislabeled as protocol errors
                await on_utterance(conn, pcm)
            # unknown message types are silently ignored -- forward
            # compatible with future Wyoming message types this listener
            # doesn't need to act on
    except (asyncio.IncompleteReadError, ConnectionResetError) as e:
        log(f"[satellites] {name} disconnected mid-stream: {e}")
    finally:
        conn._buffer = bytearray()  # discard any partial utterance
        try:
            writer.close()
        except Exception:
            pass
        log(f"[satellites] {name} connection closed")
    return conn


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

    async def _test_inbound_parsing():
        import io

        received = []

        async def fake_on_utterance(conn, pcm):
            received.append((conn.name, pcm))

        # Build a fake Wyoming stream: audio-start, one 4-sample chunk,
        # audio-stop -- exactly what handle_connection must parse.
        chunk_pcm = np.array([100, -100, 200, -200], dtype=np.int16)
        wire = (
            b'{"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}\n'
            + ('{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": %d}\n' % (chunk_pcm.nbytes)).encode()
            + chunk_pcm.tobytes()
            + b'{"type": "audio-stop"}\n'
        )
        reader = asyncio.StreamReader()
        reader.feed_data(wire)
        reader.feed_eof()

        class _FakeWriter:
            def get_extra_info(self, _):
                return ("127.0.0.1", 12345)
            def close(self):
                pass

        await handle_connection(reader, _FakeWriter(), fake_on_utterance)
        assert len(received) == 1, f"expected 1 utterance, got {len(received)}"
        name, pcm = received[0]
        assert name == "127.0.0.1:12345"
        assert np.array_equal(pcm, chunk_pcm), "parsed PCM must match what was sent"
        print("[satellites] handle_connection self-test: PASS")

    asyncio.run(_test_inbound_parsing())

    async def _test_malformed_message_recovery():
        """Verify malformed audio-chunk messages don't crash the connection;
        they must be logged and skipped, and the connection must stay open."""
        received = []

        async def fake_on_utterance(conn, pcm):
            received.append((conn.name, pcm))

        # Test: non-numeric payload_length in audio-chunk (should be caught, logged, skipped)
        # Then a valid utterance to verify the connection stays open.
        chunk_pcm = np.array([100, -100], dtype=np.int16)
        wire = (
            b'{"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}\n'
            + b'{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": "not_a_number"}\n'
            # Start fresh without sending audio-stop (skip the empty utterance)
            + b'{"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}\n'
            + ('{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": %d}\n' % (chunk_pcm.nbytes)).encode()
            + chunk_pcm.tobytes()
            + b'{"type": "audio-stop"}\n'
        )
        reader = asyncio.StreamReader()
        reader.feed_data(wire)
        reader.feed_eof()

        class _FakeWriter:
            def get_extra_info(self, _):
                return ("127.0.0.1", 12346)
            def close(self):
                pass

        # This should NOT raise an exception; the malformed chunk should be skipped
        await handle_connection(reader, _FakeWriter(), fake_on_utterance)
        # After the malformed chunk was skipped and discarded, the connection
        # remained open and processed the subsequent valid utterance.
        assert len(received) == 1, f"expected 1 valid utterance, got {len(received)}"
        name, pcm = received[0]
        assert name == "127.0.0.1:12346"
        assert np.array_equal(pcm, chunk_pcm), "valid utterance must parse correctly after malformed chunk"
        print("[satellites] malformed message recovery self-test: PASS")

    asyncio.run(_test_malformed_message_recovery())

    async def _test_callback_exception_not_swallowed():
        """Verify that exceptions raised by the on_utterance callback are NOT
        caught and logged as malformed messages -- they should propagate normally."""

        async def failing_on_utterance(conn, pcm):
            # Simulate a real bug in the downstream handler
            raise ValueError("callback intentionally failed for testing")

        chunk_pcm = np.array([100, -100], dtype=np.int16)
        wire = (
            b'{"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}\n'
            + ('{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": %d}\n' % (chunk_pcm.nbytes)).encode()
            + chunk_pcm.tobytes()
            + b'{"type": "audio-stop"}\n'
        )
        reader = asyncio.StreamReader()
        reader.feed_data(wire)
        reader.feed_eof()

        class _FakeWriter:
            def get_extra_info(self, _):
                return ("127.0.0.1", 12347)
            def close(self):
                pass

        # The callback will raise ValueError; this should propagate, not be
        # caught and logged as "malformed message"
        caught_exception = False
        try:
            await handle_connection(reader, _FakeWriter(), failing_on_utterance)
        except ValueError as e:
            caught_exception = True
            # Verify it's the callback's error, not a protocol error
            assert "callback intentionally failed" in str(e), f"expected callback error, got {e}"

        assert caught_exception, "callback exception must propagate, not be swallowed"
        print("[satellites] callback exception propagation self-test: PASS")

    asyncio.run(_test_callback_exception_not_swallowed())
