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
import asyncio
import inspect
import json
import mimetypes
import ssl
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import websockets
import websockets.asyncio.server
from websockets.datastructures import Headers
from websockets.http11 import Response

from backtalk.config import CFG
from backtalk.satellites import resample_pcm
from backtalk.vlog import log

WIRE_RATE = 16000
# websockets.serve()'s own default max_size (1 MiB) rejects a real
# utterance outright -- found live: a 24s/24kHz test clip (~1.16 MB)
# got hard-disconnected with a 1009 "message too big" close. Same
# wire-robustness reasoning as satellites.py's MAX_UTTERANCE_BYTES:
# generous enough for any real utterance (60s at up to 48kHz mono
# int16, the highest rate real browsers have reported live tonight,
# plus headroom), but never unbounded.
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


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


class CertMissing(Exception):
    """No cert/key pair on disk for the PTT server to serve."""


class _CertWatcher:
    """Holds the TLS context, and reloads it when the cert changes on disk.

    backtalk does NOT mint certificates any more. It used to generate a
    self-signed one covering every local IPv4 address, and regenerate it
    -- new private key included -- whenever any of those addresses was
    missing from the old cert. A Hyper-V/WSL virtual adapter changes its
    address across reboots, so this fired on nearly every boot, and each
    regeneration silently voided the trust exception every phone had
    granted. A device that worked last night was untrusted this morning,
    which is why the browser PTT client never once got confirmed working.

    The cert now comes from `tailscale cert`: signed by Let's Encrypt,
    valid for the machine's MagicDNS name, and trusted by every device
    with nothing installed on any of them. Renewal is
    tools/renew_tailscale_cert.ps1.
    """

    RELOAD_EVERY_SEC = 3600

    def __init__(self, cert_path: Path, key_path: Path):
        self.cert_path = cert_path
        self.key_path = key_path
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._stamp = None
        self._task = None
        self._load()

    def _read_stamp(self):
        return (self.cert_path.stat().st_mtime, self.key_path.stat().st_mtime)

    def _load(self):
        self.context.load_cert_chain(str(self.cert_path), str(self.key_path))
        self._stamp = self._read_stamp()

    async def watch(self):
        """A renewal lands as a new file on disk. load_cert_chain() on the
        LIVE context applies it to every connection opened after this
        point, so the 90-day renewal never costs Sir a restart in the
        middle of a conversation. Every failure is caught and logged: the
        old cert stays loaded and working, and a broken reload must never
        take the voice line down."""
        while True:
            await asyncio.sleep(self.RELOAD_EVERY_SEC)
            try:
                if self._read_stamp() != self._stamp:
                    self._load()
                    log("[web_client] TLS cert changed on disk -- reloaded")
            except Exception as e:
                log(f"[web_client] TLS reload failed, keeping the loaded cert: {e!r}")


def load_tls(cert_dir: Path) -> _CertWatcher:
    """Load cert.pem/key.pem from cert_dir. Raises CertMissing if either
    is absent -- deliberately NOT falling back to a self-signed pair,
    because that fallback is what produced the trust churn above and it
    would hide the real problem until a phone failed weeks later."""
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"
    if not (cert_path.is_file() and key_path.is_file()):
        raise CertMissing(f"no cert.pem/key.pem in {cert_dir}")
    return _CertWatcher(cert_path, key_path)


# Path-traversal-safe static file responder for process_request's plain-
# HTTP branch. A GET for "/" serves index.html; anything else resolves
# relative to static_dir and 404s if it escapes that directory or
# doesn't exist -- this is the one place a remote LAN device's request
# path reaches the filesystem, so the containment check is not optional
# even under the LAN-only trust model.
def _static_response(static_dir: Path, path: str) -> Response:
    rel = "ptt.html" if path in ("/", "") else path.lstrip("/")
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


def _usage_payload():
    """Plan usage for the page's readout, in the same shape ai-visualizer's
    server.py serves at its own /rate_limit -- deliberately identical, so
    the widget is a port of that one rather than a second implementation
    that can drift away from it.

    Returns None when the feature is switched off or the file cannot be
    read, which the caller turns into a 404 so the widget hides itself
    for good instead of polling a dead endpoint every three seconds.

    `captured_at_epoch` is passed through on purpose: nothing writes that
    file unless a Claude Code session is live, so a reading can age, and
    showing a stale percentage as if it were current would be a lie. The
    browser decides what counts as stale; it is not filtered out here."""
    raw = (CFG.get("usage_file") or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(Path(raw).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not payload.get("rate_limits_present"):
        # Genuinely absent (before the first response, or a plan without
        # published limits). A real state, not an error.
        return {"present": False,
                "captured_at_epoch": payload.get("captured_at_epoch")}
    rl = payload.get("rate_limits") or {}
    ctx = payload.get("context_window") or {}
    return {
        "present": True,
        "captured_at_epoch": payload.get("captured_at_epoch"),
        "model": payload.get("model"),
        "context_pct": ctx.get("used_percentage"),
        "five_hour": rl.get("five_hour"),
        "seven_day": rl.get("seven_day"),
    }


def _json_response(obj) -> Response:
    body = json.dumps(obj).encode("utf-8")
    headers = Headers()
    headers["Content-Type"] = "application/json"
    headers["Content-Length"] = str(len(body))
    headers["Cache-Control"] = "no-store"
    return Response(200, "OK", headers, body)


def _make_process_request(static_dir: Path):
    async def process_request(connection, request):
        if "Upgrade" in request.headers and request.headers["Upgrade"].lower() == "websocket":
            return None       # let the WS handshake proceed
        path = request.path.split("?", 1)[0]
        if path == "/rate_limit":
            usage = _usage_payload()
            if usage is None:
                return Response(404, "Not Found", Headers(), b"")
            return _json_response(usage)
        return _static_response(static_dir, path)
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
            if mtype == "client_log":
                # Diagnostic forwarding from ptt.html's on-page #dlog panel --
                # added 2026-09-11 after a night of fixing the iOS mic-unlock
                # flow blind, going only on Sir relaying what the phone showed.
                # This is the exact same text, straight into backtalk.log.
                level = "ERR" if msg.get("isErr") else "log"
                log(f"[web_client:client] {name} {level}: {msg.get('msg')}")
                continue
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
                       static_dir: Path, cert_dir: Path, on_disconnect=None):
    """HTTPS/WSS, not plain HTTP/WS -- getUserMedia() (the mic capture
    the whole page exists for) only works in a secure context, and a
    LAN IP over plain HTTP never qualifies, on any browser. See
    docs/superpowers/specs/2026-09-10-web-ptt-client-design.md's
    revision note on this."""
    async def handler(ws):
        await _handle_connection(ws, on_utterance, registry, on_disconnect)

    try:
        watcher = load_tls(Path(cert_dir))
    except (CertMissing, OSError, ssl.SSLError) as e:
        # Loud, and only fatal to THIS server. The mic cannot work over
        # plain HTTP in any browser, so there is no degraded mode worth
        # starting -- but the local voice line, the satellites and the
        # face have nothing to do with this and must stay up.
        log(f"[web_client] NOT STARTING the PTT web server: {e}")
        log("[web_client] fix: tools/renew_tailscale_cert.ps1 (or run "
            "`tailscale cert` by hand), then restart the voice line")
        return None

    server = await websockets.asyncio.server.serve(
        handler, host, port, ssl=watcher.context, max_size=MAX_MESSAGE_BYTES,
        process_request=_make_process_request(Path(static_dir)))
    # Held on the watcher, not dropped: a bare create_task() reference can
    # be garbage-collected mid-flight, and its exception would vanish into
    # stderr rather than backtalk.log.
    watcher._task = asyncio.create_task(watcher.watch())
    log(f"[web_client] PTT web server on https://{host}:{port}")
    return server


if __name__ == "__main__":
    # Quick self-test / smoke check, not a full test suite -- same
    # convention as satellites.py's own __main__ block.

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
