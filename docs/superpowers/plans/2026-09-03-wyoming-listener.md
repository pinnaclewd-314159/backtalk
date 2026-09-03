# backtalk-side Wyoming Listener Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give backtalk a networked ear and mouth — a Wyoming-protocol TCP listener that lets `jarvis-satellite` hardware (or any Wyoming-speaking satellite) join the same conversation PTT and typed input already use, hearing an utterance and speaking the reply back in whatever room the satellite sits in.

**Architecture:** One new module, `backtalk/backtalk/satellites.py`, holds a self-contained TCP server (connection registry, Wyoming JSONL+binary protocol parsing both directions, turn-lock rules) with no dependency on `main.py`'s internals — it exposes a callback interface. `main.py` is extended minimally: `handle()` and `speak_reply()` gain an optional `source` parameter (`"local"` or a satellite connection object) so the exact same pipeline PTT/typed input already uses now also serves satellite-originated turns, routing the reply to the right place.

**Tech Stack:** Python 3.11+, `asyncio` (stdlib, already the whole of `main.py`), `numpy` (already a dependency, used for the resampler — no new audio libraries added), no test framework (this codebase has none; tests are plain assert-based scripts run directly, matching `ears.py`/`mouth.py`'s existing `if __name__ == "__main__":` convention).

**Spec:** `backtalk/docs/superpowers/specs/2026-09-03-wyoming-listener-design.md`

## Global Constraints

- Wire format is fixed, not a free choice — confirmed against `jarvis-satellite/main/wyoming_client.c`: one JSON line per message (newline-terminated), `audio-chunk`'s JSON header immediately followed by exactly `payload_length` raw bytes of int16 mono 16kHz PCM (no base64). Both directions use this identical shape.
- Default port: **10700** (Wyoming convention, and `jarvis-satellite`'s `main/config.h` already hardcodes `BACKTALK_PORT 10700` — this is not configurable on the satellite side without a firmware change, so backtalk's default must match).
- Multi-satellite support from the start — a `dict`-based connection registry, not a single-connection global.
- Turn-lock rules (exact, from the spec — every task touching turn ownership must match this precisely):
  - Local input (PTT/typed/open-mic) always proceeds and interrupts anything active, including a satellite-owned turn.
  - A satellite's utterance proceeds only if there's no active turn, or the active turn is already owned by that *same* satellite connection.
  - Otherwise (a different satellite or local owns the active turn): the new utterance is **dropped silently** — no queueing, ever.
  - Interrupting a satellite-owned turn sends that satellite `audio-stop` immediately (never leave it waiting for audio that isn't coming).
- No new dependencies. Resampling is implemented with plain `numpy` (linear interpolation) — this project has never used `scipy`/`librosa` and the existing DSP in `ears.py`/`mouth.py` doesn't either; don't introduce one for this.
- No TLS/auth on the TCP listener — matches the trust model of every other local service in this environment.
- Room identity is explicitly out of scope (see spec's "Future extension") — but `SatelliteConnection.name` must exist as a field (defaulting to the peer address) so it's a one-line addition later, not a restructure.

---

## Task 1: Turn-lock and resampling utilities

**Files:**
- Create: `backtalk/backtalk/satellites.py`
- Test: manual — run the file directly (`python -m backtalk.satellites`), see Step 4

**Interfaces:**
- Produces: `TurnLock` class with `try_acquire(self, owner) -> bool`, `release(self, owner) -> None`, `is_active(self) -> bool`, `current_owner(self)` (returns the current owner, or `None`). `owner` is always either the literal string `"local"` or a `SatelliteConnection` instance (defined in Task 2 — for this task, any hashable placeholder object stands in for it in tests).
- Produces: `resample_pcm(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray` — int16 in, int16 out.

This is the file's first content — later tasks append to it, never replace it.

- [ ] **Step 1: Write `satellites.py` with the turn-lock and resampler**

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
```

- [ ] **Step 2: Run the self-test**

Run: `cd backtalk && .venv/Scripts/python.exe -m backtalk.satellites` (or `.venv/bin/python` on macOS/Linux)
Expected output:
```
[satellites] TurnLock self-test: PASS
[satellites] resample_pcm self-test: PASS
```

- [ ] **Step 3: If either assertion fails, fix the implementation, not the test**

The test asserts match the spec's turn-lock rules exactly (Global Constraints above) — a failure here means the `TurnLock` logic is wrong, not the test.

- [ ] **Step 4: Commit**

```bash
git add backtalk/satellites.py
git commit -m "Add TurnLock and resample_pcm for the satellite listener"
```

---

## Task 2: Wyoming protocol — inbound connection handler

**Files:**
- Modify: `backtalk/backtalk/satellites.py`

**Interfaces:**
- Consumes: nothing new from Task 1 directly (this task's logic is independent of `TurnLock`/`resample_pcm`, added to the same file).
- Produces: `class SatelliteConnection` with fields `reader: asyncio.StreamReader`, `writer: asyncio.StreamWriter`, `name: str`. `async def handle_connection(reader, writer, on_utterance) -> None`, where `on_utterance: Callable[[SatelliteConnection, np.ndarray], Awaitable[None]]` is called once per completed utterance with 16kHz mono int16 PCM. Later tasks (registry, server) rely on this exact signature.

- [ ] **Step 1: Append the connection dataclass and inbound parser to `satellites.py`**

```python
import json
from dataclasses import dataclass, field


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
                payload_len = int((msg.get("payload_length")) or 0)
                payload = await reader.readexactly(payload_len) if payload_len else b""
                conn._buffer.extend(payload)
            elif msg_type == "audio-stop":
                pcm = np.frombuffer(bytes(conn._buffer), dtype=np.int16)
                conn._buffer = bytearray()
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
```

- [ ] **Step 2: Write a manual protocol test**

Append to the `if __name__ == "__main__":` block in `satellites.py`, **before** the two `print(...)` self-test lines already there (so all self-tests run in one invocation):

```python
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
```

- [ ] **Step 3: Run it**

Run: `cd backtalk && .venv/Scripts/python.exe -m backtalk.satellites`
Expected: all three self-test lines print PASS (`TurnLock`, `resample_pcm`, `handle_connection`), no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add backtalk/satellites.py
git commit -m "Add Wyoming inbound connection parser to the satellite listener"
```

---

## Task 3: Wyoming protocol — outbound reply sender

**Files:**
- Modify: `backtalk/backtalk/satellites.py`

**Interfaces:**
- Consumes: `SatelliteConnection` (Task 2), `resample_pcm` (Task 1), `WIRE_RATE` (Task 1).
- Produces: `async def send_reply(conn: SatelliteConnection, sentence_pcm_stream, source_rate: int) -> bool`. `sentence_pcm_stream` is an iterable (sync or async — see implementation) of `np.ndarray` int16 PCM chunks at `source_rate`. Returns `True` if the whole reply was sent, `False` if the connection failed partway (caller must treat `False` as "stop synthesizing more, clean up this connection" — this is the signal Task 6 uses to abort a satellite-bound reply on a write failure, per the spec's error handling section).

- [ ] **Step 1: Append the outbound sender to `satellites.py`**

```python
async def send_reply(conn: SatelliteConnection, pcm_chunks, source_rate: int) -> bool:
    """Streams one reply to a satellite as audio-start / audio-chunk(s) /
    audio-stop, resampling each chunk from source_rate (Kokoro=24000,
    ElevenLabs=44100) down to the satellite's fixed WIRE_RATE. pcm_chunks
    may be a plain iterable or an async iterable of int16 np.ndarrays.

    Returns False on any write failure -- the caller (Task 6) must stop
    synthesizing further sentences and clean the connection out of the
    registry when this happens, per the spec: "one satellite's failure
    never takes the shared backtalk process down."
    """
    try:
        conn.writer.write(
            b'{"type": "audio-start", "data": {"rate": %d, "width": 2, "channels": 1}}\n'
            % WIRE_RATE)
        await conn.writer.drain()

        async def _chunks():
            if hasattr(pcm_chunks, "__aiter__"):
                async for c in pcm_chunks:
                    yield c
            else:
                for c in pcm_chunks:
                    yield c

        async for pcm in _chunks():
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


async def send_stop(conn: SatelliteConnection) -> None:
    """Just audio-stop, no preceding audio -- used when a satellite-owned
    turn is interrupted by local input, so its firmware isn't left
    waiting for audio that's never coming (spec, turn-lock section)."""
    try:
        conn.writer.write(b'{"type": "audio-stop"}\n')
        await conn.writer.drain()
    except (ConnectionError, OSError) as e:
        log(f"[satellites] {conn.name} send_stop failed: {e}")
```

- [ ] **Step 2: Write a manual protocol test**

Append to the `_test_inbound_parsing` async test function's caller section (add a second async test and a second `asyncio.run` call right after the first one in `if __name__ == "__main__":`):

```python
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
        ok = await send_reply(conn, [tone], source_rate=24000)
        assert ok is True
        text = bytes(written)
        assert text.startswith(b'{"type": "audio-start"'), "must start with audio-start"
        assert b'"type": "audio-chunk"' in text, "must contain an audio-chunk"
        assert text.rstrip().endswith(b'{"type": "audio-stop"}'), "must end with audio-stop"
        print("[satellites] send_reply self-test: PASS")

    asyncio.run(_test_outbound_reply())
```

- [ ] **Step 3: Run it**

Run: `cd backtalk && .venv/Scripts/python.exe -m backtalk.satellites`
Expected: four self-test PASS lines now (`TurnLock`, `resample_pcm`, `handle_connection`, `send_reply`).

- [ ] **Step 4: Commit**

```bash
git add backtalk/satellites.py
git commit -m "Add Wyoming outbound reply sender to the satellite listener"
```

---

## Task 4: Connection registry and TCP server

**Files:**
- Modify: `backtalk/backtalk/satellites.py`
- Modify: `backtalk/backtalk/config.py`

**Interfaces:**
- Consumes: `handle_connection` (Task 2), `SatelliteConnection` (Task 2).
- Produces: `class SatelliteRegistry` with `add(self, conn)`, `remove(self, conn)`, `connections: set[SatelliteConnection]` (a plain attribute, read directly by Task 6 for logging/introspection — no accessor method needed for a single attribute). `async def start_server(host: str, port: int, on_utterance, registry: SatelliteRegistry) -> asyncio.base_events.Server`.

- [ ] **Step 1: Add the `wyoming_port` config default**

In `backtalk/backtalk/config.py`, inside the `DEFAULTS` dict, add this key near `signals_dir` (both are "where backtalk listens/writes for other things to connect to" — group them):

```python
    # TCP port the satellite Wyoming listener binds to. 10700 is the
    # Wyoming protocol's own convention, and jarvis-satellite/main/config.h
    # hardcodes this exact port on the firmware side -- changing it here
    # without a matching firmware change breaks every satellite.
    "wyoming_port": 10700,
```

- [ ] **Step 2: Append the registry and server to `satellites.py`**

```python
class SatelliteRegistry:
    def __init__(self):
        self.connections: set[SatelliteConnection] = set()

    def add(self, conn: SatelliteConnection) -> None:
        self.connections.add(conn)

    def remove(self, conn: SatelliteConnection) -> None:
        self.connections.discard(conn)


async def start_server(host: str, port: int, on_utterance,
                       registry: SatelliteRegistry):
    """Binds the Wyoming TCP listener. Each accepted connection runs
    handle_connection() as its own task; the registry is updated on
    connect/disconnect so main.py's turn-lock interrupt logic (Task 6)
    can find and message any connected satellite."""

    async def _on_client(reader, writer):
        conn_ref = [None]  # populated once the first utterance names the connection

        async def _wrapped_on_utterance(conn, pcm):
            conn_ref[0] = conn
            registry.add(conn)
            await on_utterance(conn, pcm)

        try:
            await handle_connection(reader, writer, _wrapped_on_utterance)
        finally:
            if conn_ref[0] is not None:
                registry.remove(conn_ref[0])

    server = await asyncio.start_server(_on_client, host, port)
    log(f"[satellites] Wyoming listener on {host}:{port}")
    return server
```

Note: the connection is only added to the registry once it's produced its first real utterance (inside `_wrapped_on_utterance`), not at raw TCP accept time — `handle_connection` doesn't hand back a usable `SatelliteConnection` reference until it exists inside the coroutine's own scope. This is fine: the registry's only real consumer (Task 6's turn-lock interrupt path) only needs to reach a connection that's *actively owning a turn*, which by definition means it has already produced an utterance.

- [ ] **Step 3: Write a manual server smoke test**

Append a third async test to `if __name__ == "__main__":`:

```python
    async def _test_server_roundtrip():
        registry = SatelliteRegistry()
        got = []

        async def on_utterance(conn, pcm):
            got.append(pcm)
            await send_reply(conn, [pcm], source_rate=WIRE_RATE)

        server = await start_server("127.0.0.1", 17700, on_utterance, registry)
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

            writer.close()
            await asyncio.sleep(0.1)  # let the server-side handler unwind
            assert len(got) == 1
            assert np.array_equal(got[0], tone)
            print("[satellites] start_server round-trip self-test: PASS")
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test_server_roundtrip())
```

- [ ] **Step 4: Run it**

Run: `cd backtalk && .venv/Scripts/python.exe -m backtalk.satellites`
Expected: five self-test PASS lines, ending with `start_server round-trip self-test: PASS`.

- [ ] **Step 5: Commit**

```bash
git add backtalk/satellites.py backtalk/config.py
git commit -m "Add satellite connection registry and Wyoming TCP server"
```

---

## Task 5: main.py integration — wire the listener into the running session

**Files:**
- Modify: `backtalk/backtalk/main.py`

**Interfaces:**
- Consumes: `satellites.SatelliteRegistry`, `satellites.TurnLock`, `satellites.start_server`, `satellites.send_reply`, `satellites.send_stop` (all from Tasks 1-4), `ears.transcribe` (existing), `mouth.synth_stream` (existing).
- Produces: `handle(text, spoke_from=None, source="local")` and `speak_reply(brain, mouth, text, source="local")` — both existing functions, extended with a new optional parameter each. No caller of the *old* signatures needs to change (both new parameters default to `"local"`, matching every existing call site's implicit behavior exactly).

This is the task that actually makes a satellite able to talk to Jarvis. Read it fully before editing — the changes are small in line count but touch several places in an already-dense file.

- [ ] **Step 1: Import the new module and create the shared registry/lock at the top of `amain()`**

In `main.py`, add to the imports near the top of the file:

```python
from backtalk import satellites
```

Inside `amain()`, right after the existing `mouth = Mouth()` / `ears = Ears(...)` / `brain = WarmBrain(...)` block, add:

```python
    turn_lock = satellites.TurnLock()
    sat_registry = satellites.SatelliteRegistry()
```

- [ ] **Step 2: Extend `speak_reply()` to route satellite-bound replies**

Replace the existing `speak_reply` function with this version (the only changes: a new `source` parameter, and `emit` becomes `async` and branches on `source`):

```python
async def speak_reply(brain: WarmBrain, mouth: Mouth, text: str, source="local"):
    """First sentence ships alone (fast start); the rest go in
    2-sentence breaths — fuller chunks get livelier prosody (single
    short sentences come out flat). When source is a satellite
    connection, sentences are synthesized directly (mouth.synth_stream)
    and streamed to that connection instead of the local speaker queue —
    the whole point of a satellite is being heard in its own room."""
    from backtalk.mouth import synth_stream  # local import: see note below
    t0 = time.time()
    first = True
    batch: list[str] = []

    async def emit(raw: str):
        nonlocal first, batch
        s = raw.replace("`", "").replace("<<", "").replace(">>", "").strip()
        if not s:
            return
        if source == "local":
            if first:
                log(f"[{NAME}] ({time.time()-t0:.1f}s to first) {s}")
                mouth.say_chunk(s)
                first = False
            else:
                log(f"[{NAME}] {s}")
                batch.append(s)
                if len(batch) >= 2:
                    mouth.say_chunk(" ".join(batch))
                    batch = []
            return
        # Satellite-bound: synthesize this sentence directly and stream
        # it out over the connection. No batching -- unlike local
        # playback there's no shared queue to smooth over, and sending
        # sentence-by-sentence keeps latency down for the person waiting
        # in another room.
        log(f"[{NAME}->{source.name}] ({time.time()-t0:.1f}s) {s}")
        rate_holder = {}

        async def _pcm_iter():
            for rate, pcm in synth_stream(s):
                rate_holder["rate"] = rate
                yield pcm
        ok = await satellites.send_reply(source, _pcm_iter(),
                                         source_rate=rate_holder.get("rate", 24000))
        if not ok:
            sat_registry.remove(source)

    try:
        async for sentence in brain.ask_stream(text):
            await emit(sentence)
        if source == "local" and batch:
            mouth.say_chunk(" ".join(batch))
        if source == "local" and first:
            signals.static_stop()
            signals.set_state("idle")
        if source != "local":
            await satellites.send_stop(source)
    except asyncio.CancelledError:
        try:
            await brain.interrupt()
        except Exception:
            pass
        if source != "local":
            await satellites.send_stop(source)
        raise
    except Exception as e:
        # A genuine mid-turn failure (not an interrupt). The spec requires
        # a satellite-bound turn to SAY that something broke -- otherwise
        # the person in the other room just hears silence forever with no
        # idea why. (Local turns keep their pre-existing behavior: this
        # exception type was already unhandled before this feature, and
        # fixing that is outside this spec's scope.)
        log(f"[{NAME}] speak_reply failed: {e!r}")
        if source != "local":
            try:
                async def _err_pcm():
                    for rate, pcm in synth_stream(
                            "Sorry, something went wrong on my end."):
                        yield pcm
                await satellites.send_reply(source, _err_pcm(), source_rate=24000)
            except Exception:
                pass
            await satellites.send_stop(source)
        else:
            raise
    finally:
        turn_lock.release(source)
```

The `finally` block releases `turn_lock` unconditionally, for `"local"` too — `TurnLock.release()` (Task 1) only clears ownership when the given owner matches the current one, so this is always safe to call. **This matters more than it looks:** without releasing `"local"` here, the lock would stay owned by `"local"` forever after the very first local utterance, since nothing else ever resets it back to `None` — permanently blocking every satellite from acquiring the turn lock again afterward. Do not narrow this back to `if source != "local":` — that reintroduces exactly this bug.

Note the `synth_stream` import (used both in the normal per-sentence path above and the error path here) is deliberately local to the function (matching the pattern of a few other lazy imports already in this file, e.g. inside `make_permission_gate`) — it avoids a module-level import cycle risk between `main.py` and `mouth.py` at startup, since `main.py` already imports `Mouth` from `mouth.py` at the top.

There's a `rate_holder.get("rate", 24000)` default because `synth_stream`'s generator hasn't necessarily yielded yet by the time `send_reply` is called with the async iterator — the real rate is captured on first yield, before any resampling happens (the `_pcm_iter` generator always yields before `send_reply` reads `rate_holder`, since Python generators don't run ahead of their consumer; this ordering is safe by construction, not a race).

- [ ] **Step 3: Extend `handle()` with the turn-lock check and `source` passthrough**

In `handle()`, the signature changes from:
```python
    async def handle(text: str, spoke_from: float | None = None) -> bool:
```
to:
```python
    async def handle(text: str, spoke_from: float | None = None, source="local") -> bool:
```

Immediately after the `nonlocal speak_task` line (before the existing permission-ask / confirm / quit-phrase checks), insert the turn-lock gate:

```python
        if source != "local" and not turn_lock.try_acquire(source):
            log(f"[satellites] dropped utterance from {source.name} "
               f"(turn owned by {turn_lock.current_owner()})")
            return True
        if source == "local":
            turn_lock.try_acquire("local")  # always succeeds; may steal from a satellite
```

Find the existing block (already present, unchanged) that cancels an in-flight `speak_task`:
```python
        if speak_task and not speak_task.done():
            log("[turn] interrupted mid-reply by new input")
            _deny_pending()          # an ask never outlives its turn
            speak_task.cancel()
            mouth.shut_up()
```
Immediately after `mouth.shut_up()` in that same block, add the satellite-interrupt notification (only fires when the turn being cut off belonged to a satellite — sending `audio-stop` to a `source == "local"` interrupt would be meaningless since there's no satellite connection involved there):

```python
            prev_owner = turn_lock.current_owner()
            if prev_owner not in ("local", None) and prev_owner != source:
                await satellites.send_stop(prev_owner)
```

Finally, find the line near the end of `handle()` that creates the reply task:
```python
        speak_task = asyncio.create_task(
            speak_reply(brain, mouth, text))
```
Change it to pass `source` through:
```python
        speak_task = asyncio.create_task(
            speak_reply(brain, mouth, text, source=source))
```

- [ ] **Step 4: Start the satellite server alongside the existing PTT/typed-input setup**

In `amain()`, find the block that starts the typed-input reader thread:
```python
    typed_q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_typed_reader, args=(typed_q,), daemon=True).start()
```
Immediately after it, add:

```python
    async def _on_satellite_utterance(conn, pcm):
        if pcm.size == 0:
            return
        text = await loop.run_in_executor(None, transcribe, pcm)
        if text:
            await handle(text, source=conn)

    from backtalk.ears import transcribe
    await satellites.start_server(
        "0.0.0.0", CFG["wyoming_port"], _on_satellite_utterance, sat_registry)
    log(f"[backtalk] satellite listener on port {CFG['wyoming_port']}")
```

(`from backtalk.ears import transcribe` is placed here rather than at the top of the file alongside the existing `from backtalk.ears import Ears, record_held, warm as warm_ears` import for locality with its only use — but either placement works; if a reviewer prefers it grouped with the other `ears` imports at the top of the file, that's an equally correct alternative and not worth a second round-trip over.)

- [ ] **Step 5: Manual integration test**

There's no way to unit-test `amain()` itself (it's the live event loop) — this step is a real run.

Run: `cd backtalk && .venv/Scripts/python.exe -m backtalk.main` (or however this project is normally launched — check `README.md` if unsure of the exact invocation)

Expected: the existing startup log lines appear as before, PLUS a new line:
```
[satellites] Wyoming listener on 0.0.0.0:10700
[backtalk] satellite listener on port 10700
```

Then, in a second terminal, confirm the port is actually listening:

Run: `netstat -an | grep 10700` (or `Get-NetTCPConnection -LocalPort 10700` in PowerShell)
Expected: shows `LISTENING` on `10700`.

Ctrl-C the backtalk process to stop it before moving on.

- [ ] **Step 6: Commit**

```bash
git add backtalk/main.py
git commit -m "Wire the Wyoming satellite listener into backtalk's main loop"
```

---

## Task 6: Fake-satellite test client — full round-trip verification

**Files:**
- Create: `backtalk/tests/fake_satellite_client.py`

**Interfaces:**
- Consumes: nothing from `backtalk`'s own package (this is a standalone client script, deliberately independent of `satellites.py`'s internals — it only needs to speak the wire protocol, the same constraint a real satellite has).
- Produces: nothing consumed by later tasks — this is the spec's "layer 1" test tool, used here and kept for future regression checks.

- [ ] **Step 1: Create the test client**

```python
# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fake satellite: speaks Wyoming exactly like jarvis-satellite's own
firmware (main/wyoming_client.c) does, without needing real hardware.

Design: docs/superpowers/specs/2026-09-03-wyoming-listener-design.md,
"Testing" section, layer 1.

Two modes:
  python -m tests.fake_satellite_client
      Sends one real synthesized utterance ("what is two plus two"),
      via backtalk's own Kokoro TTS round-tripped through itself -- no
      external audio file needed -- and prints the reply's framing.
  python -m tests.fake_satellite_client --turnlock-test
      Opens TWO connections, triggers an utterance on both at nearly the
      same instant, and verifies exactly one of them gets a reply while
      the other is silently dropped -- exercises the turn-lock rule from
      the spec without needing two physical satellites.
"""
import asyncio
import json
import sys

import numpy as np

HOST = "127.0.0.1"
PORT = 10700
WIRE_RATE = 16000


def _tts_to_pcm16k(text: str) -> np.ndarray:
    """Uses backtalk's OWN Kokoro engine to generate a real, recognizable
    utterance -- no prerecorded WAV asset needed, and round-tripping
    synthesized speech through STT is a legitimate way to exercise this
    protocol integration without a human in the loop."""
    from backtalk.mouth import synth_stream
    from backtalk.satellites import resample_pcm
    chunks = []
    rate = WIRE_RATE
    for r, pcm in synth_stream(text):
        rate = r
        chunks.append(pcm)
    full = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    return resample_pcm(full, rate, WIRE_RATE)


async def _send_utterance(writer: asyncio.StreamWriter, pcm: np.ndarray):
    payload = pcm.tobytes()
    writer.write(b'{"type": "detect", "data": {"names": ["hey_jarvis"]}}\n')
    writer.write(
        b'{"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}\n')
    writer.write(
        ('{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, '
         '"channels": 1}, "payload_length": %d}\n' % len(payload)).encode()
        + payload)
    writer.write(b'{"type": "audio-stop"}\n')
    await writer.drain()


async def _read_reply(reader: asyncio.StreamReader, timeout: float = 60.0) -> int:
    """Reads until audio-stop or timeout; returns total reply bytes
    received (0 means no reply arrived -- the "dropped" case)."""
    total = 0
    try:
        async with asyncio.timeout(timeout):
            line = await reader.readline()
            if not line.startswith(b'{"type": "audio-start"'):
                return 0
            while True:
                line = await reader.readline()
                if not line:
                    break
                msg = json.loads(line)
                if msg.get("type") == "audio-stop":
                    break
                if msg.get("type") == "audio-chunk":
                    plen = msg["payload_length"]
                    await reader.readexactly(plen)
                    total += plen
    except (TimeoutError, asyncio.TimeoutError):
        pass
    return total


async def basic_test():
    pcm = _tts_to_pcm16k("what is two plus two")
    reader, writer = await asyncio.open_connection(HOST, PORT)
    await _send_utterance(writer, pcm)
    total = await _read_reply(reader)
    writer.close()
    if total > 0:
        print(f"[fake_satellite] PASS: got a reply, {total} bytes of 16kHz PCM")
    else:
        print("[fake_satellite] FAIL: no reply received")
        sys.exit(1)


async def turnlock_test():
    pcm = _tts_to_pcm16k("tell me a very long story about your day")
    reader_a, writer_a = await asyncio.open_connection(HOST, PORT)
    reader_b, writer_b = await asyncio.open_connection(HOST, PORT)
    await _send_utterance(writer_a, pcm)
    await asyncio.sleep(0.3)  # let A's turn actually start before B tries
    await _send_utterance(writer_b, _tts_to_pcm16k("hello"))

    total_a = await _read_reply(reader_a)
    total_b = await _read_reply(reader_b, timeout=5.0)
    writer_a.close()
    writer_b.close()

    if total_a > 0 and total_b == 0:
        print("[fake_satellite] PASS: A got a reply, B was correctly dropped")
    else:
        print(f"[fake_satellite] FAIL: expected A>0,B==0, got A={total_a},B={total_b}")
        sys.exit(1)


if __name__ == "__main__":
    if "--turnlock-test" in sys.argv:
        asyncio.run(turnlock_test())
    else:
        asyncio.run(basic_test())
```

- [ ] **Step 2: Run the basic round-trip test against a live backtalk instance**

In one terminal: `cd backtalk && .venv/Scripts/python.exe -m backtalk.main`
Wait for `[satellites] Wyoming listener on 0.0.0.0:10700` in its log.

In a second terminal: `cd backtalk && .venv/Scripts/python.exe -m tests.fake_satellite_client`

Expected: `[fake_satellite] PASS: got a reply, N bytes of 16kHz PCM` (N > 0). This means the full pipeline actually worked: fake audio → real transcription → a real Claude Code turn → a real synthesized reply → correctly-framed Wyoming audio back. This is the first point in the whole plan where a real (small, cheap) Claude Code turn actually runs — that's inherent to testing this integration, not avoidable.

- [ ] **Step 3: Run the turn-lock test against the same live instance**

Run: `cd backtalk && .venv/Scripts/python.exe -m tests.fake_satellite_client --turnlock-test`

Expected: `[fake_satellite] PASS: A got a reply, B was correctly dropped`

If this fails, re-check Task 5 Step 3's turn-lock gate is actually being reached before `speak_task` creation — a common mistake is placing the gate after the existing `if speak_task and not speak_task.done():` interrupt block instead of before it, which would let B interrupt A instead of being dropped.

- [ ] **Step 4: Commit**

```bash
git add backtalk/tests/fake_satellite_client.py
git commit -m "Add fake-satellite test client for the Wyoming listener"
```

---

## Known limitation, deliberately out of scope

The core conversational loop (satellite hears you, gets a real reply spoken back through its own speaker) is fully covered by Tasks 1-6. Two adjacent things are **not** made satellite-aware by this plan, found during self-review against the spec:

- **Permission-gate questions** (`make_permission_gate` in `main.py`) call `mouth.say(ask)` directly — a satellite-triggered turn that needs a gated tool will still ask out loud through the *local* speaker, not the satellite, even though the person may be in another room.
- **Console-verb confirmations** (`_run_console_inner`, e.g. "cleared, fresh slate" after "clear the session") likewise always call `mouth.say(...)` locally.

Both work *correctly* today (the action happens, e.g. the session really does clear) — only the *spoken confirmation's location* doesn't yet follow the satellite. Fixing this means threading `source` through the tool-permission callback chain and every `run_console` branch, not just `handle()`/`speak_reply()` — a distinct, sizeable change best done as its own follow-up plan once the core loop above is verified working, rather than expanding this one further.

- **Quit phrases** ("goodbye Jarvis", etc.) inside `handle()` return `False` to tell `amain()`'s own loop to hang up the whole session — but Task 5's `_on_satellite_utterance` callback (a different code path from that loop) never inspects `handle()`'s return value, so this signal goes nowhere. A satellite saying a quit phrase today would run `handle()`'s quit branch internally (cancel the in-flight turn, speak the signoff — locally, same local-only gap as above) without actually ending the backtalk process. Whether a satellite *should* be able to hang up the entire multi-room session at all is itself a real product question (probably not — "goodbye" from the bedroom killing the Kitchen satellite too seems wrong) that wasn't covered in the approved spec. Left unresolved here deliberately rather than guessed at; worth a real answer before the same follow-up plan above touches it.

## Final step: real hardware acceptance test (manual, not code)

Per the spec's "layer 2" — once all six tasks above are committed and both fake-satellite tests pass:

1. Make sure `jarvis-satellite/main/config.h` has real WiFi credentials and `BACKTALK_HOST` pointing at this machine's LAN IP (both were already confirmed working the night the firmware itself was brought up — see `jarvis-satellite/README.md`'s status section for the current values).
2. Power on the XIAO satellite (USB power is enough — it doesn't need to be connected to a PC for normal operation, per the note in that project's own README).
3. Start backtalk on this machine (`python -m backtalk.main`), confirm the `[satellites] Wyoming listener` log line.
4. Say "Hey Jarvis" to the satellite, then a real question.
5. Confirm the reply plays back through the **satellite's own speaker**, not this PC's — that's the actual point of the whole project ("so I can communicate with you from another room").

This step isn't a "task" in the checkbox sense because it can't be driven by a Bash command in this session — it needs a human in a physically different room from the PC to confirm. Report back what happened (worked / didn't / partially) rather than assuming success.
