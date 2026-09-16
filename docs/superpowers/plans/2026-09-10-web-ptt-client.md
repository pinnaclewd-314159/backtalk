# Browser Push-to-Talk Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let any phone/tablet/PC on the LAN talk to Jarvis via a webpage with a push-to-talk button and the same visual face `ai-visualizer` shows locally.

**Architecture:** A new `backtalk/backtalk/web_client.py` module runs a `websockets`-based server inside backtalk's own asyncio process, sharing the existing `turn_lock`/callback pattern the hardware-satellite listener (`satellites.py`) already uses. One refinement found during planning, not in the original spec wording: rather than a separate stdlib `http.server` thread alongside the WebSocket server, **one `websockets.serve()` instance serves both** — its `process_request` hook (confirmed against the installed `websockets` 17.1 API) returns a static-file `Response` for plain HTTP GETs and falls through to the normal WebSocket handshake otherwise. Same single-port outcome the spec calls for (`web_ptt_port`), simpler than running two servers.

**Tech Stack:** `websockets` 17.1 (already a dependency), Web Audio API + `AudioWorklet` (raw PCM capture, not `MediaRecorder` — the spec's wire protocol needs raw int16 PCM, and `MediaRecorder`'s compressed WebM/Opus output would need a decode step nothing in this stack does), Pointer Events API.

**Spec:** `backtalk/docs/superpowers/specs/2026-09-10-web-ptt-client-design.md`

## Global Constraints

- LAN-only trust model, no auth — matches `satellites.py`'s existing posture exactly.
- Push-to-talk only; no wake-word/VAD on the browser side.
- `ears.py`, `satellites.py`'s Wyoming wire format, and the PC's local PTT/hands-free modes are not modified.
- Inbound audio is resampled server-side to 16kHz via `satellites.resample_pcm` before `ears.transcribe()`; outbound audio is **not** resampled (browser plays native Kokoro/ElevenLabs rate).
- The `TurnLock` instance is shared with the hardware-satellite listener — never a second lock.

---

## Task 1: `web_client.py` — connection handling, turn integration, static+WS server

**Files:**
- Create: `backtalk/backtalk/web_client.py`
- Modify: `backtalk/backtalk/config.py:108` (DEFAULTS, alongside the `wake_word` block added earlier today — add after it, or after `wyoming_port` at line ~234-238 for thematic grouping with the other listener port)

**Interfaces:**
- Consumes: `satellites.SatelliteRegistry`, `satellites.resample_pcm(pcm, from_rate, to_rate)`, `satellites.TurnLock` (type only, an existing instance is passed in), `ears.transcribe` (imported by caller, not this module — matches `satellites.py`'s own pattern of staying agnostic of transcription).
- Produces: `web_client.WebConnection` (dataclass: `ws`, `name`, `_buffer: bytearray`, `_capture_rate: int`), `web_client.start_server(host, port, on_utterance, registry, static_dir, on_disconnect=None) -> websockets.asyncio.server.Server`, `web_client.send_reply(conn, rated_pcm_chunks) -> bool`, `web_client.send_stop(conn) -> None`, `web_client.push_state(conn, state: str) -> None`.

- [ ] **Step 1: Add the config key**

In `backtalk/backtalk/config.py`, after the `"wyoming_port": 10700,` line and its comment, add:

```python
    # TCP port for the browser push-to-talk client (web_client.py) —
    # serves both the static face/button page and the WebSocket PTT
    # protocol on one port via websockets' process_request hook. See
    # docs/superpowers/specs/2026-09-10-web-ptt-client-design.md.
    "web_ptt_port": 8795,
```

- [ ] **Step 2: Write the failing self-test**

Create `backtalk/backtalk/web_client.py` with just this content first:

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
        assert len(sent) == 4, f"expected 4 sends (2 text + 2 binary), got {len(sent)}"
        assert json.loads(sent[0])["type"] == "audio-chunk"
        assert json.loads(sent[0])["rate"] == 24000, "no resampling on the reply path"
        assert isinstance(sent[1], (bytes, bytearray)), "second send must be binary PCM"
        assert json.loads(sent[3])["type"] == "audio-stop"
        print("web_client send_reply self-test: OK")

    _run()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd backtalk && .venv\Scripts\python.exe -m backtalk.web_client`
Expected: `NameError: name 'send_reply' is not defined`.

- [ ] **Step 4: Implement `send_reply`, `send_stop`, `push_state`**

Insert above the `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 5: Run it to verify it passes**

Run: `cd backtalk && .venv\Scripts\python.exe -m backtalk.web_client`
Expected: `web_client send_reply self-test: OK`.

- [ ] **Step 6: Add the static-file responder and the WebSocket connection handler**

Insert above the self-test block (after `push_state`):

```python
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
```

Also add `import websockets.asyncio.server` near the top imports (the `import websockets` line alone doesn't pull in the `asyncio.server` submodule's `serve` under all import orders — explicit is safer than relying on a side-effect import elsewhere in the process).

- [ ] **Step 7: Verify the module still imports and self-test still passes**

Run: `cd backtalk && .venv\Scripts\python.exe -m backtalk.web_client`
Expected: same `OK` line as Step 5 — this step only added new functions, didn't touch `send_reply`.

- [ ] **Step 8: Commit**

```bash
cd backtalk
git add backtalk/web_client.py backtalk/config.py
git commit -m "feat: add web_client.py -- browser PTT WebSocket server

New backtalk/web_client.py: WebConnection, send_reply/send_stop/
push_state (WebSocket-native, no resampling on the reply path),
inbound utterance buffering + resample-to-16kHz via
satellites.resample_pcm, and start_server() which serves both the
static frontend and the WS protocol on one port via websockets'
process_request hook. web_ptt_port config key added. Not yet wired
into main.py -- see the design spec.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01APHWCYtdJ2sz3fEGTeEgSX"
```

---

## Task 2: Frontend — PTT audio capture and WebSocket client

**Files:**
- Create: `backtalk/web/pcm-processor.js`
- Create: `backtalk/web/ptt.html`

**Interfaces:**
- Consumes: `web_client`'s wire protocol (Task 1) — `utterance_start`/`utterance_end` text messages, one binary PCM message per utterance, `audio-chunk`/`audio-stop`/`state` messages inbound.
- Produces: nothing consumed by a later task directly — Task 3 edits this same file to add the face.

- [ ] **Step 1: Write the AudioWorklet processor**

`backtalk/web/pcm-processor.js`:

```javascript
// Runs off the main thread. Buffers nothing itself -- just forwards
// each 128-sample Float32 render quantum to the main thread, which
// accumulates and converts to int16 on release. Keeping this processor
// dumb avoids any GC pressure on the audio thread.
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      this.port.postMessage(input[0].slice());
    }
    return true;
  }
}
registerProcessor("pcm-processor", PCMProcessor);
```

- [ ] **Step 2: Write the PTT page (capture + WebSocket, no face yet)**

`backtalk/web/ptt.html`:

```html
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Jarvis</title>
<style>
  html, body { margin: 0; height: 100dvh; background: #050509; color: #fff;
               font-family: system-ui, sans-serif; }
  #app { display: flex; flex-direction: column; height: 100dvh; }
  #face { flex: 1 1 auto; display: flex; align-items: center; justify-content: center; }
  #talk {
    flex: 0 0 auto;
    margin: 16px;
    margin-bottom: calc(16px + env(safe-area-inset-bottom));
    padding: 24px;
    font-size: 1.4rem;
    border-radius: 16px;
    border: none;
    background: #2b6cff;
    color: #fff;
    touch-action: none;
    -webkit-touch-callout: none;
    user-select: none;
  }
  #talk[data-state="listening"] { background: #ff3b30; }
  #talk[data-state="busy"] { background: #555; }
</style>
</head>
<body>
<div id="app">
  <div id="face"><div id="status">Hold to talk</div></div>
  <button id="talk" data-state="idle">Hold to talk</button>
</div>
<script>
"use strict";
const talkBtn = document.getElementById("talk");
const statusEl = document.getElementById("status");
const WIRE_RATE = 16000;

let ws = null;
let audioCtx = null;
let workletNode = null;
let micStream = null;
let capturing = false;
let capturedChunks = [];
let playCtx = null;

function connect() {
  ws = new WebSocket(`ws://${location.host}/`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = onWsMessage;
  ws.onclose = () => setTimeout(connect, 2000);
}

async function onWsMessage(ev) {
  if (typeof ev.data === "string") {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") {
      talkBtn.dataset.state = msg.state;
      statusEl.textContent = msg.state === "busy" ? "Jarvis is busy"
        : msg.state[0].toUpperCase() + msg.state.slice(1) + "...";
    } else if (msg.type === "audio-chunk") {
      pendingRate = msg.rate;
    } else if (msg.type === "audio-stop") {
      talkBtn.dataset.state = "idle";
      statusEl.textContent = "Hold to talk";
    }
  } else {
    await playChunk(ev.data, pendingRate);
  }
}

let pendingRate = WIRE_RATE;

async function playChunk(arrayBuffer, rate) {
  if (!playCtx) playCtx = new AudioContext();
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
  const buf = playCtx.createBuffer(1, float32.length, rate);
  buf.copyToChannel(float32, 0);
  const src = playCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playCtx.destination);
  src.start();
}

async function startCapture() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule("pcm-processor.js");
  }
  if (audioCtx.state === "suspended") await audioCtx.resume();
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioCtx.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(audioCtx, "pcm-processor");
  capturedChunks = [];
  workletNode.port.onmessage = (e) => { if (capturing) capturedChunks.push(e.data); };
  source.connect(workletNode);
  capturing = true;
  ws.send(JSON.stringify({ type: "utterance_start", rate: audioCtx.sampleRate }));
  talkBtn.dataset.state = "listening";
  statusEl.textContent = "Listening...";
}

function stopCapture() {
  if (!capturing) return;
  capturing = false;
  micStream.getTracks().forEach((t) => t.stop());
  workletNode.disconnect();

  let total = 0;
  for (const c of capturedChunks) total += c.length;
  const int16 = new Int16Array(total);
  let offset = 0;
  for (const c of capturedChunks) {
    for (let i = 0; i < c.length; i++) {
      const s = Math.max(-1, Math.min(1, c[i]));
      int16[offset++] = s < 0 ? s * 32768 : s * 32767;
    }
  }
  ws.send(int16.buffer);
  ws.send(JSON.stringify({ type: "utterance_end" }));
  statusEl.textContent = "Sending...";
}

talkBtn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  if (talkBtn.dataset.state === "busy") return;
  startCapture();
});
talkBtn.addEventListener("pointerup", (e) => { e.preventDefault(); stopCapture(); });
talkBtn.addEventListener("pointercancel", (e) => { e.preventDefault(); stopCapture(); });
talkBtn.addEventListener("pointerleave", (e) => { if (capturing) stopCapture(); });
talkBtn.addEventListener("contextmenu", (e) => e.preventDefault());

connect();
</script>
</body>
</html>
```

- [ ] **Step 3: Verify it manually**

Run backtalk with `web_client.start_server` reachable (Task 4 wires this into `main.py`; until then, run a throwaway local server pointed at `backtalk/web/` to check the page loads and the button responds):

```bash
cd backtalk && .venv\Scripts\python.exe -c "
import asyncio
from pathlib import Path
from backtalk.web_client import start_server

async def main():
    async def noop_utterance(conn, pcm): print('utterance', len(pcm))
    await start_server('0.0.0.0', 8795, noop_utterance, set(), Path('web'))
    await asyncio.Event().wait()

asyncio.run(main())
"
```

Open `http://localhost:8795/` (or `http://<this-PC's-LAN-IP>:8795/` from a phone on the same network) in a browser. Expected: the page loads, holding the button shows "Listening...", releasing shows "Sending...", and the terminal prints `utterance <N>` with a real sample count. This is a genuine manual check — no automated harness exists for real browser mic capture in this stack.

- [ ] **Step 4: Commit**

```bash
cd backtalk
git add web/ptt.html web/pcm-processor.js
git commit -m "feat: add PTT capture frontend (no face yet)

backtalk/web/ptt.html + pcm-processor.js: AudioWorklet-based raw PCM
capture (not MediaRecorder -- the wire protocol needs raw int16, not
compressed WebM/Opus), Pointer Events for hold-to-talk with the
touch-callout/context-menu fixes, WebSocket client speaking
web_client.py's protocol. Manually verified against a throwaway
local server. No face/visuals yet -- see Task 3.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01APHWCYtdJ2sz3fEGTeEgSX"
```

---

## Task 3: Port the radial face visuals

**Files:**
- Create: `backtalk/web/core.js` (adapted copy of `ai-visualizer/core.js`)
- Modify: `backtalk/web/ptt.html` (add the face markup/canvas + adapted rendering script from `ai-visualizer/faces/radial/index.html`)

**Interfaces:**
- Consumes: `ai-visualizer/core.js`'s `AV` object shape (`AV.state`, `AV.level`, `AV.env`, `AV.samples`, `AV.alert`, `AV.name`, `AV.label`) — the exported fields the radial face's render loop already reads, unchanged.
- Produces: `AV.setRaw(obj)` — new, the hook `ptt.html`'s WebSocket handler calls instead of `core.js`'s own `/state` polling.

- [ ] **Step 1: Copy and adapt `core.js`**

Copy `ai-visualizer/core.js` to `backtalk/web/core.js`. This project scopes to state display only (no transcript panel, no rate-limit widget, no idle-dim burn-in guard, no local-mic-level visualization — all of those read local files or local hardware that don't apply to a remote page, and none are asked for in the spec). Make these edits to the copy:

Replace the bus-polling block:
```javascript
  let raw = { state: "idle", level: 0, samples: null, alert: false,
              loading: false };
  if (!DEMO) {
    setInterval(async () => {
      try {
        const r = await fetch("/state", { cache: "no-store" });
        raw = await r.json();
      } catch (e) { /* server gone: hold last state */ }
    }, 120);
  }
```
with:
```javascript
  let raw = { state: "idle", level: 0, samples: null, alert: false,
              loading: false };
  A.setRaw = (obj) => { raw = { ...raw, ...obj }; };
```

Delete the calls to `transcriptInit(...)` and `rateLimitInit()` inside `applyConfig` (both read `/log` and `/rate_limit`, neither of which `web_client.py` serves — out of scope per the spec) and delete the `transcriptInit`/`transcriptPoll`/`rateLimitInit`/`rateLimitPoll` function bodies themselves. Also delete the `dimInit`/`dimUpdate` burn-in-guard functions and their call sites — that guard is for the always-on PC display, not a phone someone picks up and puts down.

- [ ] **Step 2: Copy the radial face markup/script into `ptt.html`**

Copy the `<canvas>` markup and the rendering `<script>` block (the `spectrum`/`orbBuild`/`orbRender`/`drawParticles`/`frame`/`resize` functions etc.) from `ai-visualizer/faces/radial/index.html` into `ptt.html`'s `#face` div, replacing the placeholder `<div id="status">Hold to talk</div>` — keep a small status text overlay alongside the canvas for the state label, reusing the existing `#status` id and styling. Add `<script src="core.js"></script>` before it, matching the original's own load order (`core.js` must define `AV` before the face script references it).

Call `AV.init({ mic: false })` (no local-mic waveform on a remote page — audio capture there is for sending, not for driving the idle-breathing visual) near the top of the face script, matching whatever init call the original `index.html` makes (check its exact call and argument shape when copying — do not guess it, read the real line).

**Third gap caught in self-review:** the spec requires `window.visualViewport`, not just plain `resize`, for layout changes — CSS `100dvh` alone keeps the *button* on-screen (Task 2), but the ported face's own `resize()` function (which resizes the `<canvas>`'s backing pixel buffer, copied from the original `index.html`) still needs to be told *when* to re-run, and mobile browsers are unreliable firing plain `resize` for chrome/orientation changes. After copying the original's own `window.addEventListener("resize", resize)` call (or whatever its real equivalent is — read the actual line when copying, same rule as above), add:

```javascript
if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", resize);
}
window.addEventListener("orientationchange", resize);
```

Both are additive fallbacks alongside the original's own listener, not replacements — covers the mobile-specific gaps without removing whatever already worked in the kiosk-display context.

- [ ] **Step 3: Wire the WebSocket state pushes into `AV.setRaw`**

In `ptt.html`'s WebSocket handler (added in Task 2), change the `state` message branch from directly setting `talkBtn.dataset.state`/`statusEl.textContent` to:

```javascript
    if (msg.type === "state") {
      AV.setRaw({ state: msg.state === "busy" ? "idle" : msg.state,
                  alert: msg.state === "busy" });
      talkBtn.dataset.state = msg.state;
      statusEl.textContent = msg.state === "busy" ? "Jarvis is busy"
        : msg.state[0].toUpperCase() + msg.state.slice(1) + "...";
    }
```

`"busy"` maps to the face's existing `alert` flag (already a supported visual state per `AV`'s documented fields) rather than a fifth face state that doesn't exist in the ported rendering code — reuses what's there instead of extending the face's own state machine.

- [ ] **Step 4: Verify it manually**

Re-run the Task 2 Step 3 throwaway-server check. Expected: the radial orb renders and idles, hold-to-talk shows the listening visual (via `AV.setRaw`), and the busy/alert state is visible when a second connection (or a quick manual `ws.send('{"type":"state","state":"busy"}')` from the browser devtools console against the test server) is simulated.

- [ ] **Step 5: Commit**

```bash
cd backtalk
git add web/core.js web/ptt.html
git commit -m "feat: port the radial face visuals into the PTT page

backtalk/web/core.js: adapted copy of ai-visualizer's shared module,
bus-file polling replaced with AV.setRaw() pushed from the WebSocket
handler; transcript/rate-limit/burn-in-guard features dropped (out
of scope -- they read local files a remote page has no access to
and nothing in the spec asks for them here). ptt.html now renders
the same orb/particle face ai-visualizer shows locally, state driven
by web_client.py's WS pushes. 'busy' maps onto the face's existing
alert flag rather than a new state.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01APHWCYtdJ2sz3fEGTeEgSX"
```

---

## Task 4: Wire into `main.py`

**Files:**
- Modify: `backtalk/backtalk/main.py:72-82` (imports)
- Modify: `backtalk/backtalk/main.py:477-478` (registries)
- Modify: `backtalk/backtalk/main.py:846` (reply dispatch)
- Modify: `backtalk/backtalk/main.py:1268` (interrupt-stop dispatch)
- Modify: `backtalk/backtalk/main.py:1037-1039` (server startup)

**Interfaces:**
- Consumes: `web_client.start_server`, `web_client.send_reply`, `web_client.send_stop`, `web_client.WebConnection` (Task 1); the existing `_on_satellite_utterance(conn, pcm)` and `_on_satellite_disconnect(conn)` functions (already fully generic over `conn` — confirmed by reading them: neither assumes a `SatelliteConnection` specifically, both just check identity/type-agnostic turn-lock state and call `handle(text, source=conn)`), reused **unchanged** for the web listener rather than duplicated.

- [ ] **Step 1: Add the import**

In `backtalk/backtalk/main.py`, add alongside the existing `from backtalk import satellites` line (near line 72-73):

```python
from backtalk import web_client
```

- [ ] **Step 2: Add the registry**

Near line 477-478 (`turn_lock = satellites.TurnLock()` / `sat_registry = satellites.SatelliteRegistry()`), add:

```python
web_registry = satellites.SatelliteRegistry()   # generic, reused as-is
```

- [ ] **Step 3: Add the reply-dispatch helpers**

Near the top of the file (module scope, alongside other small helpers — placement doesn't matter functionally, keep it near `satellites`/`web_client` imports for discoverability), add:

```python
async def _send_reply(conn, gen):
    if isinstance(conn, web_client.WebConnection):
        return await web_client.send_reply(conn, gen)
    return await satellites.send_reply(conn, gen)


async def _send_stop(conn):
    if isinstance(conn, web_client.WebConnection):
        await web_client.send_stop(conn)
    else:
        await satellites.send_stop(conn)
```

- [ ] **Step 4: Replace the two direct dispatch call sites**

At line 846, change:
```python
        ok = await satellites.send_reply(conn, gen)
```
to:
```python
        ok = await _send_reply(conn, gen)
```
And the matching cleanup line just below it (`sat_registry.remove(conn)` on failure) stays as-is if `conn` is a satellite, but needs the same dispatch for a web connection — change:
```python
            sat_registry.remove(conn)
```
to:
```python
            (web_registry if isinstance(conn, web_client.WebConnection)
             else sat_registry).remove(conn)
```

At line 860 (`await satellites.send_stop(conn)`, inside the `CancelledError` handler), change to:
```python
        await _send_stop(conn)
```

At line 1268 (`await satellites.send_stop(prev_owner)`), change to:
```python
        await _send_stop(prev_owner)
```

**Real gap caught in self-review:** `web_client.push_state` (Task 1) was never actually called anywhere — satellites don't need a "busy" signal (no screen to show it on), but the spec explicitly calls for one on a web connection, and nothing wired it up. The drop happens at line 1191-1197:
```python
        if source != "local" and not epoch:
            owner = turn_lock.current_owner()
            # .name, never the object: a SatelliteConnection's repr would
            # otherwise be free to carry its raw mic PCM into the log file.
            log(f"[satellites] dropped utterance from {source.name} "
               f"(turn owned by {getattr(owner, 'name', owner)})")
            return True
```
Change to:
```python
        if source != "local" and not epoch:
            owner = turn_lock.current_owner()
            # .name, never the object: a SatelliteConnection's repr would
            # otherwise be free to carry its raw mic PCM into the log file.
            log(f"[satellites] dropped utterance from {source.name} "
               f"(turn owned by {getattr(owner, 'name', owner)})")
            if isinstance(source, web_client.WebConnection):
                await web_client.push_state(source, "busy")
            return True
```

- [ ] **Step 4b: Push "thinking"/"speaking" state to web connections**

**Second gap caught in self-review:** `push_state` now fires on "busy" (Step 4 above) but nothing yet pushes "thinking" or "speaking" — the spec calls for both. The exact hook already exists and is idempotent: `speak_reply`'s `_mark_speaking()` (lines 792-802) is called exactly once, right when the first real reply audio is ready — the same moment it flips the global bus to `"speaking"` via `signals.set_state`. Change:

```python
    speaking = False

    def _mark_speaking():
        # The first byte of real audio is where a satellite turn stops
        # "thinking" and starts "speaking" — the same moment mouth.py's
        # worker publishes it for a local turn.
        nonlocal speaking
        if not speaking:
            speaking = True
            signals.static_stop()
            signals.set_state("speaking")
```
to:
```python
    speaking = False
    if isinstance(source, web_client.WebConnection):
        asyncio.create_task(web_client.push_state(source, "thinking"))

    def _mark_speaking():
        # The first byte of real audio is where a satellite turn stops
        # "thinking" and starts "speaking" — the same moment mouth.py's
        # worker publishes it for a local turn.
        nonlocal speaking
        if not speaking:
            speaking = True
            signals.static_stop()
            signals.set_state("speaking")
            if isinstance(source, web_client.WebConnection):
                asyncio.create_task(web_client.push_state(source, "speaking"))
```

Fire-and-forget (`asyncio.create_task`, not `await`) deliberately — `_mark_speaking` is a sync function called from inside an async generator's hot path, and `push_state` is already designed as best-effort (Task 1: failures are logged and swallowed), so blocking reply synthesis on a state-push round-trip would be the wrong trade-off.

- [ ] **Step 5: Start the server alongside the satellite listener**

At line 1037-1039 (the existing `await satellites.start_server(...)` call), add immediately after:

```python
    await web_client.start_server(
        "0.0.0.0", CFG["web_ptt_port"], _on_satellite_utterance, web_registry,
        static_dir=Path(__file__).resolve().parent.parent / "web",
        on_disconnect=_on_satellite_disconnect)
    log(f"[backtalk] PTT web server on port {CFG['web_ptt_port']}")
```

`_on_satellite_utterance` and `_on_satellite_disconnect` are passed **unchanged** — both are already generic over `conn` (verified in Task's Interfaces note above), so this is genuine reuse, not a look-alike duplicate. Confirm `Path` is already imported in `main.py` (it is — `from pathlib import Path` near the top) before relying on it here.

- [ ] **Step 6: Verify it starts cleanly**

Run: `cd backtalk && .venv\Scripts\python.exe -m backtalk.main` (or however it's normally launched), watch the log for both `[satellites] Wyoming listener on 0.0.0.0:10700` and `[backtalk] PTT web server on port 8795` with no traceback. Then open `http://<this-PC's-LAN-IP>:8795/` from another device on the LAN and confirm the face renders (full PTT round-trip is Task 5's acceptance test, but the server coming up cleanly and the page loading over the network is real, checkable evidence now).

- [ ] **Step 7: Commit**

```bash
cd backtalk
git add backtalk/main.py
git commit -m "feat: wire web_client into main.py alongside satellites

New web_registry (reuses satellites.SatelliteRegistry as-is), shared
turn_lock with the hardware-satellite listener. _send_reply/_send_stop
dispatch helpers pick web_client vs satellites based on connection
type, replacing the two direct satellites.send_reply/send_stop call
sites. _on_satellite_utterance/_on_satellite_disconnect are reused
UNCHANGED for the web listener -- both were already generic over the
connection object, confirmed by reading them, not assumed. Verified:
both listeners start cleanly, page loads from another LAN device.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01APHWCYtdJ2sz3fEGTeEgSX"
```

---

## Task 5: End-to-end test, live acceptance, final commit

**Files:**
- Create: `backtalk/tools/web_ptt_test_client.py` (throwaway-adjacent but kept, mirrors `satellites.py`'s own testing philosophy of a standalone protocol client)

**Interfaces:** none — this task only exercises what Tasks 1-4 built.

- [ ] **Step 1: Write the standalone WebSocket test client**

`backtalk/tools/web_ptt_test_client.py`:

```python
"""Standalone test client for web_client.py's WebSocket PTT protocol --
speaks the wire format like a real browser would, without needing one.
Usage: python web_ptt_test_client.py <host> <port> <path/to/test.wav>
"""
import asyncio
import json
import sys
import wave

import numpy as np
import websockets


async def main(host, port, wav_path):
    with wave.open(wav_path, "rb") as wf:
        rate = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    async with websockets.connect(f"ws://{host}:{port}/") as ws:
        await ws.send(json.dumps({"type": "utterance_start", "rate": rate}))
        await ws.send(pcm.tobytes())
        await ws.send(json.dumps({"type": "utterance_end"}))
        print(f"sent {len(pcm)} samples at {rate}Hz, waiting for reply...")

        saw_state = False
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                print("<-", data)
                if data.get("type") == "state":
                    saw_state = True
                if data.get("type") == "audio-stop":
                    break
            else:
                print(f"<- binary chunk, {len(msg)} bytes")
        assert saw_state, "expected at least one state push before audio-stop"
        print("web_ptt_test_client: OK")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
```

- [ ] **Step 2: Run it against the live server**

With backtalk running (Task 4's server up), from `backtalk/`:

```bash
.venv\Scripts\python.exe tools\web_ptt_test_client.py localhost 8795 gpu_test.wav
```

Expected: `sent N samples...`, a `state` message, one or more `audio-chunk`/binary pairs, `audio-stop`, then `web_ptt_test_client: OK`. This exercises the full real pipeline (resample → transcribe → `handle()` → brain → TTS → reply framing) without a browser.

- [ ] **Step 3: Turn-lock sharing check**

With the test client from Step 2 and a real local PTT/typed turn both available, start a turn via the test client (don't let it finish — or use a longer WAV) and, while it's active, try a local typed turn in the console. Expected: local wins per the existing turn-lock rule (already true for satellites, now confirmed true for a web connection too) — the web client's connection should receive an `audio-stop` cutting its in-flight reply short, matching `_send_stop`'s dispatch from Task 4.

- [ ] **Step 4: Real device acceptance test**

From an actual phone or tablet on the LAN, open `http://<PC-LAN-IP>:8795/`:
- Full PTT round-trip with an audible spoken reply.
- Long-press the button: confirm no system context menu, no text-selection callout, no copy/share popup appears.
- Rotate the device (portrait ↔ landscape) both while idle and mid-recording: confirm the button stays visible and usable throughout, and a mid-recording rotation doesn't abort or corrupt the capture.
- With the phone actively holding a turn, try local PTT on the PC: confirm the phone shows `"busy"`/gets cut off correctly per the turn-lock rule, matching the design.
- With the internet disconnected (same style of test as yesterday's PC-side offline-fallback verification): confirm a phone-originated turn is answered by `LocalBrain`, not silence — this specific path was flagged in the spec as needing its own real test, not assumed identical just because the code path is shared.

- [ ] **Step 5: Commit the spec, plan, and test client**

```bash
cd backtalk
git add docs/superpowers/specs/2026-09-10-web-ptt-client-design.md \
        docs/superpowers/plans/2026-09-10-web-ptt-client.md \
        tools/web_ptt_test_client.py
git commit -m "docs: web PTT client design spec, plan, and test client

See docs/superpowers/specs/2026-09-10-web-ptt-client-design.md.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01APHWCYtdJ2sz3fEGTeEgSX"
```

- [ ] **Step 6: Vault checkpoint**

Update `Active Priorities.md` and the daily note with the live-confirmed status (or the honest partial status if Step 4's real-device pass surfaced anything not yet resolved) — per the "document the moment it ships" rule, record what actually happened, not the intended outcome.
