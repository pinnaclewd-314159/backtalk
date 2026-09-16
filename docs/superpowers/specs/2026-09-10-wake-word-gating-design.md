# Local hands-free wake-word gating — design

**Status:** approved, ready for implementation plan
**Date:** 2026-09-10

## Purpose

Sir noticed a real inconsistency: the `jarvis-satellite` hardware requires "Hey Jarvis" before it'll send anything to backtalk, but backtalk's own local hands-free ("open mic") mode has no such gate — plain VAD (voice activity detection) treats every utterance in the room that clears the noise floor as a real command. He wants the two aligned: the local PC mic should require the same wake word before a turn starts.

This spec covers only backtalk's local mic path (`ears.py` / `main.py`'s hands-free loop). Push-to-talk is untouched by design — holding the key is already its own trigger and needs no wake word.

## Non-goals

- Any change to push-to-talk mode.
- Any change to satellite wake-word behavior — that's `jarvis-satellite` firmware's own concern, already working.
- Training a custom wake-word model. openWakeWord's pretrained `hey_jarvis` model is used as-is, the same phrase the satellites already use.
- A conversational-memory/context feature. The grace window (below) only changes *when* a wake word is required, not how turns are handled once started.

## Architecture

**Approach chosen: single persistent mic stream, state-machine gate**, over two alternatives considered:
- *Two parallel streams* (openWakeWord on its own continuous stream, VAD capture on a separate one) — rejected: relies on unverified Windows/WASAPI support for two simultaneous opens of the same input device, and adds a second failure mode to the already-fragile device-hotplug handling in `ears.py` (`_reopen_after_device_change`).
- *Text-gate hybrid* (transcribe everything, discard non-wake-word turns after the fact) — rejected: still burns STT on every utterance in the room, which defeats the actual point (nothing should be transcribed or acted on until triggered).

The chosen design adds a new phase in front of the *existing, unchanged* `Ears.listen_once()` VAD capture rather than rewriting it:

1. Hands-free mode calls a new `ears.wait_for_wake(gate, abort, detector)`. This reads frames from the mic and feeds them to openWakeWord until it fires or `abort()` returns true (same abort/gate contract `listen_once()` already uses, so mode switches and barge-in suppression work identically).
2. On trigger: play a short chime (synthesized tone, no audio asset needed), then call `ears.listen_once()` exactly as today for the actual command capture.
3. After the reply finishes, instead of returning to `wait_for_wake()`, the loop calls `listen_once(gate=mic_gate, timeout_s=grace_window_s, abort=...)` directly — reusing `timeout_s`, which already exists on `listen_once()`, no new capability needed there.
   - A transcript within the window: handle it, then open another `grace_window_s` window.
   - A timeout (returns `None`): fall back to `wait_for_wake()`, re-arming the wake requirement.

This means `listen_once()` and its VAD state machine are not modified at all — the only new code is the wake-detection phase and the orchestration in main.py's loop deciding which phase to call next.

## Components

- **`backtalk/wakeword.py`** (new module): thin wrapper around openWakeWord.
  - Loads the pretrained `hey_jarvis` model via the `onnxruntime` backend (CPU-only, no GPU/tflite-runtime dependency needed).
  - `WakeDetector.predict(frame: np.ndarray) -> float` — score for one audio chunk.
  - Threshold check against `wake_word.threshold` (config).
  - **Open technical question, to confirm during implementation, not assumed here:** openWakeWord's documented chunk size is ~80ms (1280 samples); `ears.py`'s existing capture loop reads 30ms frames (`FRAME_LEN`, sized for webrtcvad's fixed 10/20/30ms requirement). Whether openWakeWord's `predict()` accepts arbitrary chunk sizes with internal buffering, or needs frames pre-batched to its own window size, needs a direct check against the library during implementation rather than a guess baked into this spec.
  - Chime synthesis (short sine sweep or two-tone beep, generated with numpy — no new asset file) lives here or in `mouth.py`; final placement is an implementation-time call, not a design-level one.

- **`ears.wait_for_wake(gate, abort, detector)`** (new function in `ears.py`): mirrors `listen_once()`'s existing `gate`/`abort` pattern exactly — same barge-in suppression (ignores the mic while Jarvis is speaking unless barge-in is on), same abort-on-mode-switch behavior via `_MIC["gen"]`.

- **`main.py`'s hands-free loop**: the phase-selection logic described in Architecture above replaces the current single `ears.listen_once(gate=mic_gate, abort=...)` call. The existing `_MIC` mode/gen/abort machinery, mic-failure fallback-to-PTT logic, and typed/PTT-key waiters are all unchanged — only what gets called when `_MIC["mode"] == "open"` changes.

- **Config** (`config.py` DEFAULTS + `backtalk.json`):
  - `wake_word.enabled` — default `false` until confirmed working live (matches how offline-fallback and other recent features shipped: built and self-tested first, flipped on only after Sir confirms). When `false`, hands-free mode falls back to today's plain-VAD behavior unchanged — a real escape hatch, not just a placeholder.
  - `wake_word.threshold` — openWakeWord's own recommended default, tunable if false triggers/misses show up in real use.
  - `wake_word.grace_window_s` — `9` (middle of the 8-10s range discussed).
  - `wake_word.chime` — `true`.

- **Dependency**: `openwakeword` added to `pyproject.toml`.

## Data flow summary

```
hands-free mode active
  -> wait_for_wake()            [openWakeWord scores every frame, silent otherwise]
       -> wake detected -> chime
       -> listen_once()         [existing VAD capture, unchanged]
            -> transcript -> handle() -> reply
            -> reply done -> listen_once(timeout_s=grace_window_s)  [grace window]
                 -> transcript within window -> handle() -> reply -> loop grace window again
                 -> timeout -> back to wait_for_wake()
```

## Error handling

- openWakeWord model load failure at startup: logged clearly, `wake_word.enabled` effectively forced off for the session (falls back to plain VAD) rather than crashing hands-free mode entirely — mirrors the existing STT GPU-fallback philosophy in `ears.py::warm()` (degrade, never go silent).
- Mic device errors during `wait_for_wake()`: same handling `listen_once()` already has via `explain_audio_failure()` and the existing mic-failure counter that falls back to push-to-talk after repeated failures.
- A mode switch away from hands-free (e.g. "push to talk mode") while inside `wait_for_wake()` aborts it promptly via the existing `abort` callable, identical to how it already interrupts `listen_once()` today.

## Testing

1. **Unit-level self-test**, no live mic: openWakeWord ships example positive/negative sample clips for `hey_jarvis` — run `WakeDetector.predict()` against them and assert scores land on the correct side of the threshold. Also a synthetic silence/noise run to check for spurious triggers.
2. **Real live acceptance test**, same shape as the satellite's own: say "Hey Jarvis" for real, confirm the chime and command capture fire correctly; speak in the room without the wake word and confirm nothing is transcribed or acted on; test the grace window by asking a quick follow-up within and after the window.

## Future extension (explicitly out of scope now)

- Per-utterance sensitivity auto-tuning based on false-trigger history.
- Sharing one wake-word detector instance across a future multi-mic setup, if the mobile comms lab or another local input source ever needs its own hands-free mode.
