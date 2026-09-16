# Browser push-to-talk client — design

**Status:** approved, ready for implementation plan
**Date:** 2026-09-10
**Companion project:** `jarvis-satellite`/`backtalk/backtalk/satellites.py` (the Wyoming-protocol hardware satellite listener) — this spec deliberately reuses its `TurnLock` and `SatelliteRegistry` rather than inventing parallel turn semantics.

## Purpose

Sir wants to talk to Jarvis from any phone, tablet, or PC on the LAN — no dedicated hardware, no app install, just a webpage with a push-to-talk button and the same visual face `ai-visualizer` already shows on the main display. This is a second, complementary "remote input" channel alongside the existing hardware satellites, not a replacement for either the PC's own mic or `jarvis-satellite`.

## Non-goals

- Wake-word / hands-free listening on the browser side. Push-to-talk only — the button press *is* the trigger, same role the PC's `ptt_key` already plays.
- Authentication or access control. Trust model is explicitly LAN-only (Sir's own call, matching every other local service in this stack — `satellites.py`'s Wyoming listener has zero auth by the same design).
- Any change to `ears.py`, `satellites.py`'s Wyoming wire format, or the PC's own local PTT/hands-free modes. This is a new, additive channel.
- Room identity / per-device naming. Out of scope here, same as it was for the original Wyoming listener spec — a future extension if it's ever needed.

## Architecture

**A second listener running inside backtalk's own asyncio process, sharing the existing `TurnLock` instance** (`turn_lock = satellites.TurnLock()`, `main.py` line 477) with the hardware-satellite listener — a phone and a physical satellite compete fairly for the turn under one rule, not two independent locks. Rejected alternatives, both considered and dropped:

- *A separate bridge process translating WebSocket↔Wyoming TCP* — adds an extra hop and a second thing to keep alive for no real benefit; `satellites.py`'s core primitives (`TurnLock`, `SatelliteRegistry`, `send_reply`/`send_stop`) are already protocol-agnostic in everything but their wire-level read/write functions, so a sibling module can reuse the architecture directly, in-process.
- *Extending `ai-visualizer`'s `server.py` instead of backtalk itself* — rejected per Sir's own call: `ai-visualizer` is more of an open-source/upstream-flavored project here (local edits already need care to survive a future `./update.sh`), and this feature needs deep access to backtalk's mic-turn internals that `ai-visualizer` has zero access to today. Backtalk owns its own remote-input surface, same as it already owns the Wyoming listener.

**Multiple simultaneous devices are supported** (Sir's call) — `TurnLock`/`SatelliteRegistry` already generalize to N connections for free: local input always wins, a same-connection re-trigger interrupts its own prior turn, any other connection's utterance while a turn is active is dropped silently. The one addition needed beyond what satellites already get for free: **a visible "Jarvis is busy" state on a phone that tries to press the button while another device or local holds the turn** — a hardware satellite has no screen to show this on, but a person staring at a phone waiting for a tap to register does need the feedback.

## Components

- **`backtalk/backtalk/web_client.py`** (new module, mirrors `satellites.py`'s shape):
  - `WebConnection` — dataclass wrapping a `websockets` connection object + a peer-derived name, same role `SatelliteConnection` plays for TCP.
  - Reuses `satellites.SatelliteRegistry` as-is (already generic: `set[connection]`, nothing Wyoming-specific in it).
  - `start_server(host, port, on_utterance, registry, on_disconnect=None)` — same signature shape as `satellites.start_server`, so `main.py` wires it up identically.
  - `send_reply(conn, rated_pcm_chunks)` / `send_stop(conn)` — WebSocket-native equivalents of `satellites.py`'s functions of the same name, same semantics (return `False` on write failure, caller removes the connection from the registry), but **skip the down-to-16kHz resampling** `satellites.py` needs for the ESP32's fixed I2S bus — a browser plays back whatever rate Kokoro/ElevenLabs natively produced, so `resample_pcm` isn't called here at all.
- **`backtalk/web/`** (new static asset folder): `ptt.html` + JS/CSS — a copy of the radial face's visual assets from `ai-visualizer/faces/radial/`, with its local-bus-file polling stripped out and replaced by state pushed live over the WebSocket (lower latency than polling, and the only option anyway since a remote device has no local files to read).
- **A plain stdlib HTTP server** (`http.server`, threaded — matching the framework-free style already used across this stack, no new dependency) serving the static folder above. Runs alongside the WebSocket listener, both started from `main.py`.
- **`main.py` changes:**
  - New module-level `web_registry = web_client.SatelliteRegistry()` alongside the existing `sat_registry`, both sharing the one `turn_lock`.
  - `web_client.start_server(...)` launched alongside the existing `satellites.start_server(...)` call (~line 1037).
  - The reply-dispatch point (~line 846, currently calls `satellites.send_reply`/`send_stop` directly) gains **one** dispatch check — `isinstance(conn, web_client.WebConnection)` picks the web module's functions instead — kept to a single point, not scattered `isinstance` checks through the file.
- **Config** (`config.py` DEFAULTS): new `web_ptt_port` key, following the exact existing pattern of `wyoming_port`.

## Wire protocol

WebSocket, not Wyoming JSON-lines — push-to-talk's clean start/end means the browser can buffer the whole utterance client-side rather than streaming live chunks the way RAM-constrained firmware has to:

**Inbound (browser → backtalk):** on `pointerdown` the browser starts local recording and sends `{"type": "utterance_start", "rate": N}` (text), where `N` is whatever sample rate the browser's `AudioContext` actually captured at — browsers don't reliably honor a requested rate, so this is read from the real context rather than assumed. On `pointerup`/`pointercancel` it sends **one binary WebSocket message** containing the whole utterance's int16 mono PCM at that rate, followed by `{"type": "utterance_end"}` (text). No chunked `audio-chunk` framing needed on this side at all — genuinely simpler than the Wyoming protocol it's inspired by. Backend resamples the buffered PCM down to 16kHz using `satellites.resample_pcm` (already rate-agnostic, already a dependency) **before** calling `ears.transcribe()`, which requires exactly int16 mono 16kHz — same function already used for the *outbound* direction in `satellites.py`, just applied the other way here.

**Outbound (backtalk → browser), per turn:** `{"type": "audio-chunk", "rate": N}` (text) immediately followed by one binary PCM frame, repeated per synthesized sentence (same one-envelope-per-reply shape as `satellites.send_reply`, just without the resampling step), ending with `{"type": "audio-stop"}` (text).

**State pushes**, interleaved on the same connection whenever that connection's own turn changes phase: `{"type": "state", "state": "listening"|"thinking"|"speaking"|"busy"}` — `"busy"` is the one state with no hardware-satellite equivalent, sent when a `pointerdown` arrives while `turn_lock.try_acquire()` returns `None` (someone else already owns the turn).

## Data flow

1. Browser opens the WebSocket on page load; backtalk accepts and registers the `WebConnection` in `web_registry` at accept time (same "registered on connect, not on first utterance" rule `satellites.py` already follows, for the same reason — a drop before any utterance still needs to be visible).
2. `pointerdown` → if `turn_lock.try_acquire(conn)` fails, send `{"type": "state", "state": "busy"}` and do nothing further; otherwise begin local recording and send `utterance_start` with the real capture rate.
3. `pointerup`/`pointercancel` → stop recording, send the buffered PCM + `utterance_end`.
4. Backtalk's `web_client` module buffers the incoming binary message, and on `utterance_end` resamples it to 16kHz (`satellites.resample_pcm`) and calls `ears.transcribe()` directly (no VAD/endpointing needed — the browser already decided the utterance's bounds), then the shared `on_utterance` callback, identical in shape to the satellite path's `_on_satellite_utterance`.
5. `handle(text, source=conn)` proceeds exactly as it already does for every other input source — **including the offline-fallback brain selection** (`active_brain = brain if connectivity.is_online() else local_brain`, `main.py` line 1293), which is keyed purely on global connectivity state with no branching on where the turn came from. A web-client turn during an outage gets the same `LocalBrain` treatment (HA intents + canned utility replies) as local or satellite input, for free — confirmed by reading the actual dispatch code, not assumed.
6. Reply streams back per the outbound wire protocol above; the page's face renders the pushed state changes live.

## Frontend (touch UI requirements)

- **Hold-to-talk via the Pointer Events API** (`pointerdown`/`pointerup`/`pointercancel`), not separate touch/mouse handlers — the modern unified input API, broadly supported.
- **`touch-action: none`** CSS on the button, plus `-webkit-touch-callout: none` and `user-select: none`, plus `preventDefault()` in `pointerdown` and on `contextmenu` — together these suppress the long-press context-menu/text-selection/"Save Image" callout behavior touchscreens default to, which would otherwise fight with a hold-to-talk gesture. `pointercancel` (fired when the OS interrupts the gesture — an incoming call, a notification swipe, the finger sliding off the button) **must** be handled as a release, not ignored, or the mic can get stuck believing it's still recording.
- **Layout:** face and button in a flex column, button pinned as a fixed-height flex item at the bottom — never absolutely positioned at a fixed pixel coordinate that could land off-screen on a different aspect ratio.
- **`100dvh`** (dynamic viewport height), not `100vh`, on the outer container — avoids the well-known mobile bug where `100vh` over-counts space hidden behind a browser's dynamic address bar, which is exactly what could push the button off-screen on an orientation change or chrome show/hide.
- **`env(safe-area-inset-bottom)`** padding under the button, for notched/gesture-nav phones.
- **`window.visualViewport`**, not just the plain `resize` event, for layout-affecting size changes — more reliable on mobile for chrome/orientation changes.
- **Button states**, driven by the pushed `state` messages plus local recording/sending status: idle ("Hold to talk") → held/recording ("Listening…") → sent, waiting for transcription+reply ("Sending…" then the pushed `"thinking"` state) → `"speaking"` while the reply plays. `"busy"` overrides all of the above when another connection holds the turn.

## Error handling

Mirrors `satellites.py`'s existing philosophy directly, extended to the new connection type:
- A WebSocket write failure mid-reply aborts that reply and removes the connection from `web_registry` — one device's failure never takes the shared process down.
- A dropped connection mid-turn (browser closed, phone locked, network hiccup) releases `turn_lock` via the same `on_disconnect` hook pattern `satellites.py` already uses.
- A malformed/unexpected message is logged and the connection stays open, matching `satellites.py`'s `_read_message` tolerance for one bad line.
- The interrupt path (a different source taking over an active web-client turn) sends `{"type": "audio-stop"}` immediately, mirroring `satellites.send_stop`'s purpose — the page shouldn't sit showing "thinking"/"speaking" forever for a turn that's been superseded.

## Testing

1. **A standalone Python WebSocket test client**, mirroring `satellites.py`'s own self-test pattern — sends `utterance_start` with a known rate, a known WAV's PCM as one binary message, then `utterance_end`, and verifies the transcript and the reply's framing come back correctly. Also exercises turn-lock sharing directly: a simulated web client and a simulated satellite connection contending for the same turn, without needing real hardware or a real browser for either.
2. **Real acceptance test from an actual phone/tablet on the LAN**, covering: a full PTT round-trip with audible reply; the pointer-event/context-menu fix (long-press does not bring up a system menu or text selection); orientation change both while idle and mid-recording; two devices (or a device plus local PTT) contending for a turn at the same time, confirming the loser sees `"busy"` rather than silent nothing; and a real internet-outage test from the phone specifically (not assumed identical to yesterday's PC-side test just because the code path is shared) confirming `LocalBrain` picks up the turn correctly.

## Future extension (explicitly out of scope now)

Room/device identity, the same deferred item the original Wyoming listener spec carries — useful once there's a reason to give the agent per-device context (e.g. "the Kitchen tablet" implying the Kitchen's lights), not needed for this spec's actual ask.
