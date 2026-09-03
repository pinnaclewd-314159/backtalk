# backtalk-side Wyoming listener — design

**Status:** approved, ready for implementation plan
**Date:** 2026-09-03
**Companion project:** `jarvis-satellite` (the ESP32-S3 firmware this talks to) — see its own `docs/superpowers/specs/2026-09-02-satellite-firmware-design.md`. That firmware is confirmed working end to end as of last night (real "Hey Jarvis" detection on real hardware) and already speaks the exact wire format this spec implements against.

## Purpose

Sir wants to talk to Jarvis from any room, not just at this PC — that's the entire point of the `jarvis-satellite` hardware project. The satellite firmware already wake-words, captures an utterance, and streams it out over the Wyoming protocol; nothing on backtalk's side has ever listened for it. This spec adds that listener: backtalk gains a network ear and a network mouth, in addition to (not instead of) its existing local PTT/open-mic ears and local speaker mouth.

One backtalk instance, one ongoing conversation — a satellite is a new **input and output channel** into the exact same session, not a separate assistant. Anything you can do today from PTT or typed input (ask a question, say "clear the session," answer a permission prompt) works identically from a satellite.

## Non-goals (this spec)

- Room identity / naming satellites. Deferred — see "Future extension" below.
- Wake-word tuning, VAD threshold tuning, or anything on the firmware side. That's `jarvis-satellite`'s own concern; this listener trusts that by the time `audio-start` arrives, the satellite has already decided this is a real utterance.
- TLS/authentication on the TCP listener. Matches the trust model of the rest of this environment (LAN-only, no auth on any existing local service).

## Architecture

A new module, `backtalk/backtalk/satellites.py`, runs one `asyncio` TCP server (default port 10700, matching Wyoming convention and `jarvis-satellite`'s hardcoded `BACKTALK_PORT`) as a task inside backtalk's existing event loop in `main.py` — not a separate process. Each accepted connection is registered in an in-memory dict keyed by the connection object itself; multiple satellites can be connected simultaneously from the start (per Sir's explicit call — the eventual plan is three, in Bedroom/Kitchen/Lounge, all talking to this same instance).

Each connection's state is a small object holding: the stream reader/writer, an audio-accumulation buffer (bytes, reset on every `audio-start`), and nothing else — no room name today (see Future extension), but the field exists and defaults to the connection's peer address, so adding real names later is a one-line change, not a restructure.

## Wire format

Confirmed directly against the firmware source (`jarvis-satellite/main/wyoming_client.c`), not assumed — this is what's actually implemented and already sending on the satellite side:

```
{"type": "detect", "data": {"names": ["hey_jarvis"]}}\n
{"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}}\n
{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": N}\n<N raw PCM bytes>
  ... (repeated for the length of the utterance)
{"type": "audio-stop"}\n
```

One JSON line per message (newline-terminated), `audio-chunk` immediately followed by exactly `payload_length` raw bytes of int16 mono 16kHz PCM — no base64, no framing beyond the length prefix. The reply direction (backtalk → satellite) uses the identical shape, symmetric: `audio-start` once, then `audio-chunk` messages as each sentence synthesizes, then `audio-stop` when the full reply is done. The firmware's own `wyoming_client.c` already has a TODO stub expecting exactly this on the read side.

`detect`'s `data.names` is logged, not acted on — only one wake word exists right now, so there's nothing to gate on yet.

## Data flow

**Inbound (satellite → text → reply):**
1. Satellite connects; registered in the connection dict.
2. `audio-start` resets that connection's buffer.
3. Each `audio-chunk`'s raw payload is appended to the buffer.
4. `audio-stop` closes the utterance: buffer → int16 numpy array → `ears.transcribe()` (called directly — no local VAD/endpointing needed or wanted; the satellite already did that) → text.
5. Empty/silence-only transcription: no reply, no turn started, matches `Ears.listen_once`'s existing convention.
6. Non-empty text is handed to `main.py`'s existing `handle(text)`, tagged with the originating connection. `handle()` itself needs no changes — every input source already flows through it uniformly (console verbs, permission-gate answers, everything works identically from a satellite).

**Outbound (reply → satellite):**
1. When `handle()` starts a turn for a satellite-sourced utterance, that connection becomes the turn owner (see concurrency model below).
2. Per sentence, instead of `mouth.say_chunk()` (which queues for local playback), call `mouth.synth_stream(sentence)` directly, resample each PCM chunk to 16kHz mono (Kokoro renders at 24kHz, ElevenLabs at 44.1kHz — the satellite is fixed at 16kHz), and write it to the owning connection as `audio-chunk` messages, bracketed by one `audio-start` and a final `audio-stop`.
3. `signals.py` state (thinking/speaking/idle) updates normally regardless of audio routing, so the PC-side face/visualizer stays accurate even when the reply is heard elsewhere.

## Concurrency and turn-lock model

One shared "active turn owner": `None`, `"local"` (PTT/typed/open-mic — unchanged from today), or a specific satellite connection.

- **Local input always wins.** It proceeds and interrupts whatever's active — including a satellite-owned turn — exactly matching existing PTT/typed behavior today, just extended to also override satellites.
- **A satellite's utterance proceeds only if:** there's no active turn, or the active turn is already owned by that *same* satellite (re-triggering interrupts its own prior turn, mirroring how local input can already interrupt itself).
- **Otherwise (a different satellite, or local, owns the active turn): the new utterance is dropped silently.** No queueing — matches the firmware's own "drop, return to idle, never block" principle on `wyoming_client.c`'s side; this listener follows the same philosophy rather than inventing a different one.
- When a satellite-owned turn is interrupted by local input, that satellite is sent `audio-stop` immediately so its firmware doesn't sit waiting for audio that isn't coming.
- A dropped connection (satellite reboots, network hiccup) while it owns the active turn cancels that turn and clears the owner, same as any other interrupt.

## Error handling

- Malformed/unexpected messages: logged, that line skipped, connection stays open.
- Connection drops mid-utterance (before `audio-stop`): buffer discarded, never transcribed.
- **The agent errors mid-turn while answering a satellite:** the error message is synthesized and sent to *that satellite*, not local speakers — otherwise Sir would never hear that something broke while away from the PC.
- TTS failure: `mouth.synth_stream()` already falls back ElevenLabs→Kokoro automatically; total failure sends the satellite a graceful `audio-stop` rather than hanging it.
- A satellite's socket write failing mid-reply aborts that reply and cleans up the connection — one satellite's failure never takes the shared backtalk process down.

## Testing

Two layers:
1. **A standalone Python Wyoming-client script** that speaks the protocol like a real satellite — sends a known WAV as `audio-start`/`audio-chunk`(s)/`audio-stop`, verifies the transcript and the reply's framing come back correctly. Fast, repeatable, no hardware needed for most iteration — and lets the turn-lock/drop rules be tested with two simulated connections without needing two physical satellites.
2. **Real hardware acceptance test**, once the above passes: say "Hey Jarvis" to the actual `jarvis-satellite` unit and confirm the reply plays back through its own speaker.

## Future extension (explicitly out of scope now)

Room identity: once real per-room satellites exist, give each connection a real name (config-driven, matched by IP or a device-reported ID) so the agent can receive room context with an utterance (e.g. "turn off the lights" from the Kitchen satellite implying the Kitchen's lights). The connection object already has the field reserved; this needs agent-prompt/tool-calling changes too, not just the listener, so it's a separate spec when it's actually needed.
