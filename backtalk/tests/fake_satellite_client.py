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
