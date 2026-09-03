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
  python -m tests.fake_satellite_client [words of a question]
      Sends one real synthesized utterance (default: "what is two plus
      two"), via backtalk's own Kokoro TTS round-tripped through itself
      -- no external audio file needed -- and checks the reply's framing
      is exactly ONE audio-start ... chunks ... audio-stop envelope.
      Passing a question that provokes several sentences is the sharpest
      way to see that: the framing must not change with sentence count.
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


async def _read_reply(reader: asyncio.StreamReader, timeout: float = 60.0,
                      grace: float = 2.0) -> tuple[int, int, int]:
    """Reads the WHOLE reply, not just up to the first audio-stop, and
    returns (total_payload_bytes, audio_start_count, audio_stop_count).

    Reading only until the first audio-stop is what let a real framing
    bug pass this test: backtalk used to open a fresh
    audio-start/audio-stop envelope per SENTENCE, so a stop-and-return
    reader saw a correct-looking reply that was actually just sentence
    one. The spec's contract is ONE envelope per reply, so this keeps
    reading for `grace` seconds after the first audio-stop -- anything
    that arrives in that window is a second envelope and a failure.

    (0, 0, 0) means no reply arrived at all -- the "dropped" case."""
    total = starts = stops = 0
    try:
        async with asyncio.timeout(timeout):
            while True:
                if stops:
                    # the reply claims to be over: only wait a moment to
                    # catch a second envelope trailing behind it
                    try:
                        async with asyncio.timeout(grace):
                            line = await reader.readline()
                    except (TimeoutError, asyncio.TimeoutError):
                        break
                else:
                    line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                kind = msg.get("type")
                if kind == "audio-start":
                    starts += 1
                elif kind == "audio-stop":
                    stops += 1
                elif kind == "audio-chunk":
                    plen = msg["payload_length"]
                    await reader.readexactly(plen)
                    total += plen
    except (TimeoutError, asyncio.TimeoutError):
        pass
    return total, starts, stops


def _shape(total: int, starts: int, stops: int) -> str:
    return f"{starts} audio-start / {stops} audio-stop / {total} bytes"


async def basic_test():
    prompt = " ".join(sys.argv[1:]).strip() or "what is two plus two"
    pcm = _tts_to_pcm16k(prompt)
    reader, writer = await asyncio.open_connection(HOST, PORT)
    await _send_utterance(writer, pcm)
    total, starts, stops = await _read_reply(reader)
    writer.close()
    if total <= 0:
        print(f"[fake_satellite] FAIL: no reply received ({_shape(total, starts, stops)})")
        sys.exit(1)
    # ONE envelope for the whole reply, however many sentences it has.
    if starts != 1 or stops != 1:
        print(f"[fake_satellite] FAIL: framing — expected exactly one "
              f"audio-start ... chunks ... audio-stop, got "
              f"{_shape(total, starts, stops)}")
        sys.exit(1)
    secs = total / (WIRE_RATE * 2)
    print(f"[fake_satellite] PASS: got a reply, {total} bytes of 16kHz PCM "
          f"({secs:.2f}s), framing OK ({_shape(total, starts, stops)})")


async def turnlock_test():
    pcm = _tts_to_pcm16k("tell me a very long story about your day")
    reader_a, writer_a = await asyncio.open_connection(HOST, PORT)
    reader_b, writer_b = await asyncio.open_connection(HOST, PORT)
    await _send_utterance(writer_a, pcm)
    await asyncio.sleep(0.3)  # let A's turn actually start before B tries
    await _send_utterance(writer_b, _tts_to_pcm16k("hello"))

    total_a, starts_a, stops_a = await _read_reply(reader_a)
    total_b, starts_b, stops_b = await _read_reply(reader_b, timeout=5.0)
    writer_a.close()
    writer_b.close()

    if total_a > 0 and starts_a == 1 and stops_a == 1 and total_b == 0 and starts_b == 0:
        print(f"[fake_satellite] PASS: A got a reply "
              f"({_shape(total_a, starts_a, stops_a)}), B was correctly dropped")
    else:
        print(f"[fake_satellite] FAIL: expected A = one envelope with audio, "
              f"B = nothing; got A={_shape(total_a, starts_a, stops_a)}, "
              f"B={_shape(total_b, starts_b, stops_b)}")
        sys.exit(1)


if __name__ == "__main__":
    if "--turnlock-test" in sys.argv:
        asyncio.run(turnlock_test())
    else:
        asyncio.run(basic_test())
