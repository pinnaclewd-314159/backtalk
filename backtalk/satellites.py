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
import inspect
import json
from dataclasses import dataclass, field

import numpy as np

from backtalk.vlog import log

WIRE_RATE = 16000  # satellite mic/speaker rate, fixed by the firmware

# Wire-robustness ceilings. This listener binds 0.0.0.0 with no auth (the
# spec's trust model), so a single broken peer -- a firmware glitch is
# enough, no attacker required -- must never be able to make this process
# allocate without bound. Both caps are generous for any real utterance:
# a satellite chunk is a few kB, and 60s is far longer than the firmware's
# own capture window.
MAX_CHUNK_BYTES = 1024 * 1024              # 1 MB per audio-chunk payload
MAX_UTTERANCE_BYTES = 60 * WIRE_RATE * 2   # 60s of 16kHz mono int16 (~1.92 MB)

# 2026-09-15: a satellite that resets (power blip, or -- confirmed live --
# closing a debug serial connection to it) without sending a TCP FIN leaves
# reader.readline() awaiting bytes that are never coming. Nothing in this
# file's own logic ever revisits that await, so the connection stays
# ESTABLISHED on this end forever: never logged as closed, never removed
# from the registry, and if it happened to own the turn lock, that lock
# never releases either. A real satellite is never silent this long mid-
# connection -- it streams a chunk every ~20-50ms during an utterance and
# otherwise connects fresh per wake-word episode -- so any gap this long
# between messages means the peer is gone, not just slow.
IDLE_READ_TIMEOUT_S = 30.0


class TurnLock:
    """Tracks which source (the string "local", or a SatelliteConnection)
    currently owns the active turn. Not thread-safe by design — every
    caller runs on the single asyncio event loop backtalk already uses,
    same as every other piece of shared state in main.py."""

    def __init__(self):
        self._owner = None
        self._epoch = 0

    def current_owner(self):
        return self._owner

    def is_active(self) -> bool:
        return self._owner is not None

    def try_acquire(self, owner):
        """Returns the newly-acquired epoch (a positive int, always
        truthy) on success, or None if the utterance should be dropped.
        Every successful acquisition -- including a SAME-owner
        re-trigger -- starts a new epoch: a self-retrigger is a genuinely
        new turn, and this is what makes a stale release() from the turn
        it replaced unable to clobber it."""
        if owner == "local" or self._owner is None or self._owner == owner:
            self._owner = owner
            self._epoch += 1
            return self._epoch
        return None

    def release(self, owner, epoch) -> None:
        """Clears ownership only if the caller both still holds it AND is
        releasing the CURRENT epoch -- a release from a turn that's since
        been superseded (even by its own owner re-triggering) must not
        clobber the newer turn that replaced it."""
        if self._owner == owner and self._epoch == epoch:
            self._owner = None


# Windowed-sinc low-pass, numpy only -- see resample_pcm for why this
# exists. 63 taps with a Blackman window puts the stopband below -60dB,
# which is the figure that matters: linear interpolation on its own only
# attenuates by 3-5dB, and anything it leaves above the target Nyquist
# comes back as an in-band alias at very nearly full amplitude.
_LOWPASS_TAPS = 63


def _lowpass_taps(cutoff_hz: float, rate: int, n_taps: int = _LOWPASS_TAPS) -> np.ndarray:
    fc = cutoff_hz / rate                      # cycles per sample
    m = np.arange(n_taps) - (n_taps - 1) / 2.0
    h = 2 * fc * np.sinc(2 * fc * m) * np.blackman(n_taps)
    return h / h.sum()                         # unity gain at DC


def resample_pcm(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """int16 mono PCM at from_rate -> int16 mono PCM at to_rate, via a
    low-pass filter (downsampling only) followed by linear interpolation.
    No scipy/librosa in this project (ears.py and mouth.py don't use one
    either, and scipy is only a transitive dependency here -- `uv sync`
    strips anything pyproject.toml doesn't declare).

    HARD-WON AUDIO LAW #3 (2026-09-18) -- DOWNSAMPLING WITHOUT THE
    LOW-PASS FIRST IS BROKEN, however good the interpolator is. Dropping
    24kHz Kokoro to the satellites' 16kHz wire rate folds everything
    above 8kHz straight back into the speech band. Measured on this
    function before the filter existed: a 10kHz tone, which must vanish
    entirely, survived at -4.0dB aliased down to 6kHz; correct filtering
    puts it at -57dB. Sibilants are where a TTS voice keeps most of its
    8-12kHz energy, so the damage lands on consonants and reads as a
    fuzzy, distorted voice. ElevenLabs at 44.1kHz was far worse again.

    This went unheard for months because the ESP32-S3-BOX-3's onboard
    speaker barely reproduces 4-8kHz and was masking it. It was an
    external PCM5102A DAC into a powered speaker that finally exposed it.
    Diagnosis is in JarvisVault 07 - Resources/BOX-3 Hardware Facts.md.

    Cutoff sits at 45% of the target rate, leaving the filter's
    transition band room to land before the new Nyquist.
    """
    if from_rate == to_rate or len(pcm) == 0:
        return pcm.astype(np.int16)
    x = pcm.astype(np.float64)
    if to_rate < from_rate and x.size > _LOWPASS_TAPS:
        x = np.convolve(x, _lowpass_taps(0.45 * to_rate, from_rate), mode="same")
    duration_s = len(pcm) / from_rate
    n_out = max(1, int(round(duration_s * to_rate)))
    x_old = np.linspace(0.0, duration_s, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, duration_s, num=n_out, endpoint=False)
    resampled = np.interp(x_new, x_old, x)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


@dataclass
class SatelliteConnection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    name: str
    _buffer: bytearray = field(default_factory=bytearray)
    # True once this utterance blew a wire-robustness ceiling: the rest of
    # it is discarded (and never transcribed) until the next audio-start.
    _dropping: bool = False
    # True for the duration of send_reply() below. A satellite legitimately
    # sends nothing while it is only receiving/playing a reply -- exactly
    # the same silence IDLE_READ_TIMEOUT_S exists to catch on a genuinely
    # dead connection. Without this flag a reply whose total
    # generation+delivery time exceeds IDLE_READ_TIMEOUT_S gets its
    # connection torn down mid-stream (found 2026-09-15: a 6-sentence
    # reply cut off with reply audio-stop never sent).
    _replying: bool = False

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        # NEVER the dataclass default: that renders _buffer, which is raw
        # microphone PCM, straight into any log line that interpolates a
        # connection.
        return f"<SatelliteConnection {self.name}>"


async def _read_message(reader: asyncio.StreamReader) -> dict | None:
    """One Wyoming JSON line -> dict, or None on a clean EOF/disconnect.
    A malformed line is logged and skipped (returns an empty dict, which
    callers treat as a no-op) rather than tearing down the connection --
    Wyoming is a simple line protocol and one bad message shouldn't be
    fatal. That covers three separate ways a line can be bad: unparsable
    bytes, a line longer than the stream reader's own limit (readline
    raises ValueError and drops it, so the next readline resyncs), and
    VALID json that isn't an object at all (a bare number or list --
    msg.get() on one of those would raise AttributeError up in the
    parser, outside its own malformed-message handling, and tear the
    connection down)."""
    try:
        line = await reader.readline()
    except ValueError as e:      # line longer than the reader's limit
        log(f"[satellites] over-length message, skipping: {e}")
        return {}
    if not line:
        return None
    try:
        msg = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log(f"[satellites] malformed message, skipping: {e}")
        return {}
    if not isinstance(msg, dict):
        log(f"[satellites] malformed message (not a json object), "
            f"skipping: {type(msg).__name__}")
        return {}
    return msg


def make_connection(reader: asyncio.StreamReader,
                    writer: asyncio.StreamWriter) -> SatelliteConnection:
    """One accepted socket -> its SatelliteConnection, named for its peer.

    Split out of handle_connection so start_server can build (and
    register) the connection at ACCEPT time -- the spec's data flow says
    "satellite connects; registered in the connection dict", and a
    disconnect that happens before the satellite's first utterance has to
    be reportable too."""
    peer = writer.get_extra_info("peername")
    name = f"{peer[0]}:{peer[1]}" if peer else "unknown"
    return SatelliteConnection(reader=reader, writer=writer, name=name)


async def handle_connection(reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter,
                            on_utterance,
                            conn: SatelliteConnection | None = None
                            ) -> SatelliteConnection:
    """Runs for the life of one satellite's TCP connection. Parses
    inbound audio-start/audio-chunk/audio-stop, and calls on_utterance
    once per completed utterance. Returns the SatelliteConnection so the
    caller (start_server) can register/unregister it; pass `conn` when
    the caller already built one at accept time."""
    if conn is None:
        conn = make_connection(reader, writer)
    name = conn.name
    log(f"[satellites] {name} connected")
    try:
        while True:
            try:
                msg = await asyncio.wait_for(_read_message(reader),
                                              timeout=IDLE_READ_TIMEOUT_S)
            except asyncio.TimeoutError:
                if conn._replying:
                    # Expected silence, not a dead connection: the
                    # satellite is only receiving/playing send_reply()'s
                    # audio right now and has nothing of its own to send
                    # until that finishes. Re-arm the read instead of
                    # tearing the connection down under it -- see
                    # SatelliteConnection._replying's own comment.
                    continue
                log(f"[satellites] {name} idle timeout "
                    f"({IDLE_READ_TIMEOUT_S:.0f}s no data) -- "
                    f"dropping stale connection")
                break
            if msg is None:
                break
            msg_type = msg.get("type")
            if msg_type == "detect":
                names = (msg.get("data") or {}).get("names")
                log(f"[satellites] {name} detect: {names}")
            elif msg_type == "audio-start":
                conn._buffer = bytearray()
                conn._dropping = False
            elif msg_type == "audio-chunk":
                try:
                    payload_len = int((msg.get("payload_length")) or 0)
                    if payload_len > MAX_CHUNK_BYTES:
                        # Refuse to allocate it, and don't read it either:
                        # a length this wrong means the peer is broken, and
                        # readline resyncs on the next newline. Same
                        # log-and-skip contract as any malformed message.
                        raise ValueError(
                            f"payload_length {payload_len} over the "
                            f"{MAX_CHUNK_BYTES}-byte chunk ceiling")
                    payload = await reader.readexactly(payload_len) if payload_len else b""
                except (ValueError, TypeError) as e:
                    log(f"[satellites] {name} malformed message (skipped): {e}")
                    conn._buffer = bytearray()  # discard any partial utterance
                    conn._dropping = True
                    continue
                if conn._dropping:
                    continue     # payload read (stream stays in sync), discarded
                if len(conn._buffer) + len(payload) > MAX_UTTERANCE_BYTES:
                    log(f"[satellites] {name} utterance over the "
                        f"{MAX_UTTERANCE_BYTES}-byte ceiling (no audio-stop?) "
                        f"-- discarding it, connection stays open")
                    conn._buffer = bytearray()
                    conn._dropping = True
                    continue
                conn._buffer.extend(payload)
            elif msg_type == "audio-stop":
                if conn._dropping:
                    # this utterance already blew a ceiling: it is partial
                    # garbage, so it is discarded rather than transcribed
                    conn._buffer = bytearray()
                    conn._dropping = False
                    continue
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
        conn._dropping = False
        try:
            writer.close()
        except Exception:
            pass
        log(f"[satellites] {name} connection closed")
    return conn


async def send_reply(conn: SatelliteConnection, rated_pcm_chunks) -> bool:
    """Streams ONE WHOLE REPLY to a satellite as exactly one envelope:
    audio-start, then an audio-chunk per PCM chunk, then one audio-stop
    when the reply is done (the spec's wire format -- one envelope per
    reply, however many sentences it contains, so this must be called
    once per turn and never once per sentence).

    `rated_pcm_chunks` is an iterable OR async iterable of
    `(sample_rate, pcm)` tuples -- exactly the shape
    mouth.synth_stream() already yields, so the caller can hand its
    generator straight through. Each chunk is resampled from ITS OWN
    rate (Kokoro=24000, ElevenLabs=44100) down to the satellite's fixed
    WIRE_RATE; carrying the rate per chunk is what makes a mid-reply
    engine fallback correct, and removes any need for the caller to know
    the rate before synthesis has started.

    Returns False on any write failure -- the caller (main.py's
    speak_reply, Task 5) must stop synthesizing further sentences and
    clean the connection out of the registry when this happens, per the
    spec: "one satellite's failure never takes the shared backtalk
    process down."

    Sets conn._replying for the whole call (cleared in `finally`, so it
    always clears on any exit path) -- see the field's own comment for
    why: handle_connection()'s idle-read loop needs to know a reply is
    actively streaming so it doesn't mistake the satellite's expected
    silence during that window for a dead connection.
    """
    conn._replying = True
    try:
        conn.writer.write(
            b'{"type": "audio-start", "data": {"rate": %d, "width": 2, "channels": 1}}\n'
            % WIRE_RATE)
        await conn.writer.drain()

        async def _chunks():
            if hasattr(rated_pcm_chunks, "__aiter__"):
                async for c in rated_pcm_chunks:
                    yield c
            else:
                for c in rated_pcm_chunks:
                    yield c

        async for source_rate, pcm in _chunks():
            resampled = resample_pcm(pcm, source_rate, WIRE_RATE)
            if resampled.size == 0:
                continue
            payload = resampled.tobytes()
            header = (
                '{"type": "audio-chunk", "data": {"rate": %d, "width": 2, '
                '"channels": 1}, "payload_length": %d}\n'
                % (WIRE_RATE, len(payload))
            ).encode()
            conn.writer.write(header + payload)
            await conn.writer.drain()

        conn.writer.write(b'{"type": "audio-stop"}\n')
        await conn.writer.drain()
        return True
    except (ConnectionError, OSError) as e:
        log(f"[satellites] {conn.name} write failed mid-reply: {e}")
        return False
    finally:
        conn._replying = False


async def send_stop(conn: SatelliteConnection) -> None:
    """Just audio-stop, no preceding audio -- used when a satellite-owned
    turn is interrupted by local input, so its firmware isn't left
    waiting for audio that's never coming (spec, turn-lock section)."""
    try:
        conn.writer.write(b'{"type": "audio-stop"}\n')
        await conn.writer.drain()
    except (ConnectionError, OSError) as e:
        log(f"[satellites] {conn.name} send_stop failed: {e}")


class SatelliteRegistry:
    def __init__(self):
        self.connections: set[SatelliteConnection] = set()

    def add(self, conn: SatelliteConnection) -> None:
        self.connections.add(conn)

    def remove(self, conn: SatelliteConnection) -> None:
        self.connections.discard(conn)


async def start_server(host: str, port: int, on_utterance,
                       registry: SatelliteRegistry, on_disconnect=None):
    """Binds the Wyoming TCP listener. Each accepted connection runs
    handle_connection() as its own task; the registry is updated on
    connect/disconnect so main.py's turn-lock interrupt logic (Task 5)
    can find and message any connected satellite.

    A connection is registered at ACCEPT time, not at its first
    utterance -- that's what the spec's data flow says ("satellite
    connects; registered in the connection dict"), and it's what makes a
    drop before the first utterance visible at all.

    `on_disconnect`, when given, is called with the SatelliteConnection
    once its connection has torn down, for any reason. It may be sync or
    async. This module stays self-contained -- it knows nothing about
    turn locks -- so main.py uses this hook to cancel a turn whose owner
    just vanished (spec: "a dropped connection ... cancels that turn and
    clears the owner, same as any other interrupt"). Its exceptions are
    logged and swallowed: a callback bug must not take the listener down.
    """

    async def _on_client(reader, writer):
        conn = make_connection(reader, writer)
        registry.add(conn)
        try:
            await handle_connection(reader, writer, on_utterance, conn=conn)
        finally:
            registry.remove(conn)
            if on_disconnect is not None:
                try:
                    result = on_disconnect(conn)
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    log(f"[satellites] {conn.name} on_disconnect failed: {e!r}")

    server = await asyncio.start_server(_on_client, host, port)
    log(f"[satellites] Wyoming listener on {host}:{port}")
    return server


if __name__ == "__main__":
    # Manual self-test, same convention as ears.py/mouth.py's own
    # __main__ blocks -- this project has no test framework.
    lock = TurnLock()
    ep_local = lock.try_acquire("local")
    assert ep_local, "local must always be able to acquire"
    assert lock.current_owner() == "local"
    assert lock.try_acquire("sat_a") is None, "local turn must not be stolen"
    lock.release("local", ep_local)
    assert lock.current_owner() is None
    ep_a = lock.try_acquire("sat_a")
    assert ep_a, "an unowned turn must be acquirable"
    assert lock.try_acquire("sat_b") is None, "a different satellite must be dropped"
    ep_a2 = lock.try_acquire("sat_a")
    assert ep_a2, "the SAME satellite may re-trigger itself"
    ep_local2 = lock.try_acquire("local")
    assert ep_local2, "local always wins, even over a satellite"
    lock.release("sat_a", ep_a2)  # stale release from the superseded turn
    assert lock.current_owner() == "local", "a stale release must not clobber a newer owner"
    lock.release("local", ep_local2)
    assert lock.current_owner() is None
    print("[satellites] TurnLock self-test: PASS")

    # THE SELF-INTERRUPT REGRESSION. Owner-matching alone was not enough:
    # when a source interrupted ITSELF, the superseded turn's release
    # matched the (unchanged) owner and cleared a lock the replacement
    # turn was still holding, leaving the turn unowned for its whole
    # duration -- so a DIFFERENT satellite could take it out from under
    # the one that was actually speaking. The epoch is what closes that.
    lock = TurnLock()
    old_epoch = lock.try_acquire("sat_a")
    new_epoch = lock.try_acquire("sat_a")     # the same satellite re-triggers
    assert old_epoch != new_epoch, "a self-retrigger must start a NEW epoch"
    lock.release("sat_a", old_epoch)          # the superseded turn unwinds
    assert lock.current_owner() == "sat_a", \
        "a stale release from a self-superseded turn must be a no-op"
    assert lock.try_acquire("sat_b") is None, \
        "the lock must still genuinely be HELD, not just report an owner"
    lock.release("sat_a", new_epoch)          # the live turn finishes
    assert lock.current_owner() is None, "the current epoch's release must work"
    assert lock.try_acquire("sat_b"), "and the turn is free again afterwards"
    print("[satellites] TurnLock self-interrupt epoch self-test: PASS")

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

    async def _test_outbound_reply():
        written = bytearray()

        class _FakeWriter:
            def write(self, data):
                written.extend(data)
            async def drain(self):
                pass
            def get_extra_info(self, _):
                return ("127.0.0.1", 9999)
            def close(self):
                pass

        conn = SatelliteConnection(reader=None, writer=_FakeWriter(), name="test")
        tone = (np.sin(2 * np.pi * 440 * np.arange(2400) / 24000) * 10000).astype(np.int16)
        # Several chunks at DIFFERENT rates in one call: one envelope out,
        # and each chunk resampled from its own tuple's rate -- the whole
        # point of the (rate, pcm) contract.
        el_tone = (np.sin(2 * np.pi * 440 * np.arange(4410) / 44100) * 10000).astype(np.int16)
        ok = await send_reply(conn, [(24000, tone), (44100, el_tone)])
        assert ok is True
        text = bytes(written)
        assert text.startswith(b'{"type": "audio-start"'), "must start with audio-start"
        assert b'"type": "audio-chunk"' in text, "must contain an audio-chunk"
        assert text.rstrip().endswith(b'{"type": "audio-stop"}'), "must end with audio-stop"
        # Parse the wire for real (payload bytes can contain newlines, so
        # a naive split/count would lie).
        types, lens, pos = [], [], 0
        while pos < len(text):
            nl = text.find(b"\n", pos)
            if nl < 0:
                break
            msg = json.loads(text[pos:nl])
            pos = nl + 1
            types.append(msg.get("type"))
            if msg.get("type") == "audio-chunk":
                lens.append(msg["payload_length"])
                pos += msg["payload_length"]
        assert types.count("audio-start") == 1, f"exactly ONE audio-start: {types}"
        assert types.count("audio-stop") == 1, f"exactly ONE audio-stop: {types}"
        assert types == ["audio-start", "audio-chunk", "audio-chunk",
                         "audio-stop"], f"one envelope per reply, got {types}"
        # 2400 @24k -> 1600 @16k (3200 bytes); 4410 @44.1k -> 1600 @16k (3200 bytes)
        assert lens == [3200, 3200], f"per-chunk rate must be honoured, got {lens}"
        print("[satellites] send_reply self-test: PASS")

    asyncio.run(_test_outbound_reply())

    async def _test_outbound_reply_write_failure():
        """Verify send_reply returns False on a write failure (drain raises),
        and does not propagate the exception."""
        written = bytearray()

        class _FakeWriterWithFailure:
            def __init__(self):
                self.drain_call_count = 0

            def write(self, data):
                written.extend(data)

            async def drain(self):
                self.drain_call_count += 1
                if self.drain_call_count == 1:
                    raise ConnectionResetError("simulated reset")

            def get_extra_info(self, _):
                return ("127.0.0.1", 9999)

            def close(self):
                pass

        conn = SatelliteConnection(reader=None, writer=_FakeWriterWithFailure(), name="test")
        tone = (np.sin(2 * np.pi * 440 * np.arange(2400) / 24000) * 10000).astype(np.int16)
        ok = await send_reply(conn, [(24000, tone)])
        assert ok is False, "send_reply must return False on write failure"
        print("[satellites] send_reply write-failure self-test: PASS")

    asyncio.run(_test_outbound_reply_write_failure())

    async def _test_send_stop():
        """Verify send_stop works in both success and failure cases,
        and never propagates exceptions."""
        # Success case: normal drain
        written = bytearray()

        class _FakeWriter:
            def write(self, data):
                written.extend(data)

            async def drain(self):
                pass

            def get_extra_info(self, _):
                return ("127.0.0.1", 9999)

            def close(self):
                pass

        conn = SatelliteConnection(reader=None, writer=_FakeWriter(), name="test")
        await send_stop(conn)
        assert bytes(written) == b'{"type": "audio-stop"}\n', "send_stop must write audio-stop message"

        # Failure case: drain raises exception
        written_fail = bytearray()

        class _FakeWriterWithFailure:
            def write(self, data):
                written_fail.extend(data)

            async def drain(self):
                raise ConnectionResetError("simulated reset")

            def get_extra_info(self, _):
                return ("127.0.0.1", 9999)

            def close(self):
                pass

        conn_fail = SatelliteConnection(reader=None, writer=_FakeWriterWithFailure(), name="test")
        # This must not raise -- send_stop must swallow the exception
        await send_stop(conn_fail)
        assert bytes(written_fail) == b'{"type": "audio-stop"}\n', "send_stop must still write before drain fails"

        print("[satellites] send_stop self-test: PASS")

    asyncio.run(_test_send_stop())

    async def _test_server_roundtrip():
        registry = SatelliteRegistry()
        got = []

        async def on_utterance(conn, pcm):
            got.append(pcm)
            await send_reply(conn, [(WIRE_RATE, pcm)])

        dropped = []

        async def on_disconnect(conn):
            dropped.append(conn)

        server = await start_server("127.0.0.1", 17700, on_utterance, registry,
                                    on_disconnect=on_disconnect)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 17700)
            tone = (np.sin(2 * np.pi * 440 * np.arange(320) / 16000) * 5000).astype(np.int16)
            payload = tone.tobytes()
            writer.write(b'{"type": "detect", "data": {"names": ["hey_jarvis"]}}\n')
            writer.write(b'{"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}\n')
            writer.write(
                ('{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": %d}\n' % len(payload)).encode()
                + payload)
            writer.write(b'{"type": "audio-stop"}\n')
            await writer.drain()

            header_line = await reader.readline()
            assert header_line.startswith(b'{"type": "audio-start"'), header_line
            chunk_line = await reader.readline()
            assert b'"type": "audio-chunk"' in chunk_line, chunk_line
            plen = json.loads(chunk_line)["payload_length"]
            await reader.readexactly(plen)
            stop_line = await reader.readline()
            assert stop_line.startswith(b'{"type": "audio-stop"'), stop_line

            assert len(registry.connections) == 1, \
                "the connection must be registered at ACCEPT time"

            writer.close()
            await asyncio.sleep(0.1)  # let the server-side handler unwind
            assert len(got) == 1
            assert np.array_equal(got[0], tone)
            assert len(dropped) == 1, "on_disconnect must fire on teardown"
            assert not registry.connections, "registry must be empty again"
            print("[satellites] start_server round-trip self-test: PASS")
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test_server_roundtrip())

    async def _test_disconnect_before_any_utterance():
        """A satellite that connects and drops WITHOUT ever speaking must
        still be registered and still fire on_disconnect -- the whole point
        of registering at accept time."""
        registry = SatelliteRegistry()
        dropped = []
        seen_registered = []

        async def on_utterance(conn, pcm):
            pass

        def on_disconnect(conn):          # sync callback, deliberately
            dropped.append(conn)

        server = await start_server("127.0.0.1", 17701, on_utterance, registry,
                                    on_disconnect=on_disconnect)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 17701)
            await asyncio.sleep(0.1)
            seen_registered.append(len(registry.connections))
            writer.close()
            await asyncio.sleep(0.1)
            assert seen_registered == [1], \
                f"connect-time registration expected, got {seen_registered}"
            assert len(dropped) == 1, "on_disconnect must fire with no utterance"
            assert not registry.connections
            print("[satellites] connect-time registration / on_disconnect "
                  "self-test: PASS")
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test_disconnect_before_any_utterance())

    async def _test_wire_robustness_caps():
        """A broken peer must never make this process allocate without
        bound: an over-size payload_length is refused (and never read),
        an unbounded run of audio-chunks with no audio-stop is capped,
        a non-dict json line is skipped, and the connection survives all
        three."""
        received = []

        async def fake_on_utterance(conn, pcm):
            received.append(pcm)

        class _FakeWriter:
            def get_extra_info(self, _):
                return ("127.0.0.1", 12348)
            def close(self):
                pass

        good = np.array([1, 2, 3, 4], dtype=np.int16)
        # a "chunk" claiming 4 GB, a bare-number json line, then a real
        # utterance -- the connection must still be parsing normally.
        wire = (
            b'{"type": "audio-start", "data": {"rate": 16000}}\n'
            + b'{"type": "audio-chunk", "payload_length": 4294967296}\n'
            + b'42\n'
            + b'["not", "an", "object"]\n'
            + b'{"type": "audio-start", "data": {"rate": 16000}}\n'
            + ('{"type": "audio-chunk", "payload_length": %d}\n' % good.nbytes).encode()
            + good.tobytes()
            + b'{"type": "audio-stop"}\n'
        )
        reader = asyncio.StreamReader()
        reader.feed_data(wire)
        reader.feed_eof()
        await handle_connection(reader, _FakeWriter(), fake_on_utterance)
        assert len(received) == 1, f"expected 1 utterance, got {len(received)}"
        assert np.array_equal(received[0], good)

        # Now the total-buffer ceiling: chunks that never stop.
        received.clear()
        blob = np.zeros(200_000, dtype=np.int16)   # 400 kB per chunk
        header = ('{"type": "audio-chunk", "payload_length": %d}\n'
                  % blob.nbytes).encode()
        wire = b'{"type": "audio-start", "data": {"rate": 16000}}\n'
        for _ in range(12):                        # 4.8 MB, well over the cap
            wire += header + blob.tobytes()
        wire += b'{"type": "audio-stop"}\n'
        # ...and a clean utterance afterwards, proving it stayed open.
        wire += (b'{"type": "audio-start", "data": {"rate": 16000}}\n'
                 + ('{"type": "audio-chunk", "payload_length": %d}\n' % good.nbytes).encode()
                 + good.tobytes()
                 + b'{"type": "audio-stop"}\n')
        reader = asyncio.StreamReader(limit=8 * 1024 * 1024)
        reader.feed_data(wire)
        reader.feed_eof()
        w = _FakeWriter()
        conn = SatelliteConnection(reader=reader, writer=w, name="cap-test")
        await handle_connection(reader, w, fake_on_utterance, conn=conn)
        assert len(received) == 1, \
            f"the over-cap utterance must be dropped, not transcribed: {len(received)}"
        assert np.array_equal(received[0], good), "the later clean utterance must parse"
        assert len(conn._buffer) == 0, "the buffer must not survive the connection"
        print("[satellites] wire-robustness caps self-test: PASS")

    asyncio.run(_test_wire_robustness_caps())
