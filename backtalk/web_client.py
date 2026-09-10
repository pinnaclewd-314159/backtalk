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
"""Browser push-to-talk client — a second, WebSocket-native remote-input
listener alongside satellites.py's Wyoming one, sharing the same
TurnLock/SatelliteRegistry pattern. See docs/superpowers/specs/
2026-09-10-web-ptt-client-design.md."""
import inspect
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import websockets
import websockets.asyncio.server
from websockets.datastructures import Headers
from websockets.http11 import Response

from backtalk.satellites import resample_pcm
from backtalk.vlog import log

WIRE_RATE = 16000


@dataclass
class WebConnection:
    ws: object
    name: str
    _buffer: bytearray = field(default_factory=bytearray)
    _capture_rate: int = WIRE_RATE

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"<WebConnection {self.name}>"


async def send_reply(conn: WebConnection, rated_pcm_chunks) -> bool:
    """Same one-envelope-per-reply shape as satellites.send_reply, but
    WITHOUT resampling -- the browser plays back whatever rate Kokoro/
    ElevenLabs actually produced, chunk by chunk (mixed rates within one
    reply are fine; each chunk carries its own rate)."""
    try:
        async def _chunks():
            if hasattr(rated_pcm_chunks, "__aiter__"):
                async for c in rated_pcm_chunks:
                    yield c
            else:
                for c in rated_pcm_chunks:
                    yield c

        async for rate, pcm in _chunks():
            if pcm.size == 0:
                continue
            await conn.ws.send(json.dumps({"type": "audio-chunk", "rate": rate}))
            await conn.ws.send(pcm.astype(np.int16).tobytes())

        await conn.ws.send(json.dumps({"type": "audio-stop"}))
        return True
    except (websockets.exceptions.ConnectionClosed, OSError) as e:
        log(f"[web_client] {conn.name} write failed mid-reply: {e}")
        return False


async def send_stop(conn: WebConnection) -> None:
    try:
        await conn.ws.send(json.dumps({"type": "audio-stop"}))
    except (websockets.exceptions.ConnectionClosed, OSError) as e:
        log(f"[web_client] {conn.name} send_stop failed: {e}")


async def push_state(conn: WebConnection, state: str) -> None:
    """state: 'listening' | 'thinking' | 'speaking' | 'busy'. Best-effort
    -- a failed push here isn't the reply itself, so it's logged and
    swallowed rather than raised."""
    try:
        await conn.ws.send(json.dumps({"type": "state", "state": state}))
    except (websockets.exceptions.ConnectionClosed, OSError) as e:
        log(f"[web_client] {conn.name} push_state failed: {e}")


# Path-traversal-safe static file responder for process_request's plain-
# HTTP branch. A GET for "/" serves index.html; anything else resolves
# relative to static_dir and 404s if it escapes that directory or
# doesn't exist -- this is the one place a remote LAN device's request
# path reaches the filesystem, so the containment check is not optional
# even under the LAN-only trust model.
def _static_response(static_dir: Path, path: str) -> Response:
    rel = "index.html" if path in ("/", "") else path.lstrip("/")
    target = (static_dir / rel).resolve()
    try:
        target.relative_to(static_dir.resolve())
    except ValueError:
        return Response(403, "Forbidden", Headers(), b"")
    if not target.is_file():
        return Response(404, "Not Found", Headers(), b"Not found")
    body = target.read_bytes()
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    headers = Headers()
    headers["Content-Type"] = ctype
    headers["Content-Length"] = str(len(body))
    return Response(200, "OK", headers, body)


def _make_process_request(static_dir: Path):
    async def process_request(connection, request):
        if "Upgrade" in request.headers and request.headers["Upgrade"].lower() == "websocket":
            return None       # let the WS handshake proceed
        return _static_response(static_dir, request.path.split("?", 1)[0])
    return process_request


async def _handle_connection(ws, on_utterance, registry, on_disconnect):
    peer = ws.remote_address
    name = f"{peer[0]}:{peer[1]}" if peer else "unknown"
    conn = WebConnection(ws=ws, name=name)
    registry.add(conn)
    log(f"[web_client] {name} connected")
    try:
        async for message in ws:
            if isinstance(message, (bytes, bytearray)):
                conn._buffer.extend(message)
                continue
            try:
                msg = json.loads(message)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                log(f"[web_client] {name} malformed message, skipping: {e}")
                continue
            mtype = msg.get("type")
            if mtype == "utterance_start":
                conn._buffer = bytearray()
                conn._capture_rate = int(msg.get("rate", WIRE_RATE))
            elif mtype == "utterance_end":
                pcm = np.frombuffer(bytes(conn._buffer), dtype=np.int16)
                conn._buffer = bytearray()
                if pcm.size:
                    resampled = resample_pcm(pcm, conn._capture_rate, WIRE_RATE)
                    await on_utterance(conn, resampled)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        registry.remove(conn)
        log(f"[web_client] {name} disconnected")
        if on_disconnect is not None:
            try:
                result = on_disconnect(conn)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                log(f"[web_client] {name} on_disconnect failed: {e!r}")


async def start_server(host: str, port: int, on_utterance, registry,
                       static_dir: Path, on_disconnect=None):
    async def handler(ws):
        await _handle_connection(ws, on_utterance, registry, on_disconnect)

    server = await websockets.asyncio.server.serve(
        handler, host, port,
        process_request=_make_process_request(Path(static_dir)))
    log(f"[web_client] PTT web server on {host}:{port}")
    return server


if __name__ == "__main__":
    # Quick self-test / smoke check, not a full test suite -- same
    # convention as satellites.py's own __main__ block.
    import asyncio

    def _run():
        # A fake connection recording every .send() call, so send_reply's
        # framing can be checked without a real socket.
        sent = []

        class FakeWS:
            async def send(self, data):
                sent.append(data)

        conn = WebConnection(ws=FakeWS(), name="test")

        async def _chunks():
            yield (24000, np.array([100, 200, -100], dtype=np.int16))
            yield (24000, np.array([300], dtype=np.int16))

        ok = asyncio.run(send_reply(conn, _chunks()))
        assert ok is True, "send_reply should report success"
        # 2 chunks -> (audio-chunk + binary) * 2 + one final audio-stop = 5
        assert len(sent) == 5, f"expected 5 sends, got {len(sent)}"
        assert json.loads(sent[0])["type"] == "audio-chunk"
        assert json.loads(sent[0])["rate"] == 24000, "no resampling on the reply path"
        assert isinstance(sent[1], (bytes, bytearray)), "second send must be binary PCM"
        assert json.loads(sent[-1])["type"] == "audio-stop"
        print("web_client send_reply self-test: OK")

    _run()
