# Local Hands-Free Wake-Word Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make backtalk's local hands-free ("open mic") mode require "Hey Jarvis" before a turn starts, matching the satellites, instead of today's plain-VAD-triggers-everything behavior.

**Architecture:** A new `wait_for_wake()` phase (openWakeWord's pretrained `hey_jarvis` model, its own short-lived mic stream) runs before the *existing, unmodified* `Ears.listen_once()` VAD capture. `main.py`'s hands-free loop gains a three-state cycle (`wake` → `capture` → `grace` → back to `wake`) that decides which function to call next; `listen_once()`'s existing `timeout_s` parameter implements the post-reply grace window with no changes to that function itself.

**Tech Stack:** `openwakeword` (onnxruntime backend), existing `sounddevice`/`numpy` already in the project.

**Spec:** `backtalk/docs/superpowers/specs/2026-09-10-wake-word-gating-design.md`

## Global Constraints

- `wake_word.enabled` defaults to `false` in `config.py` DEFAULTS — ships inert, matching how offline-fallback and other recent features shipped (built + self-tested first, flipped on only after a live confirm).
- Push-to-talk mode is untouched — no wake-word logic may run on that path.
- `Ears.listen_once()` in `ears.py` is not modified — the spec chose this design specifically to avoid touching its VAD state machine.
- openWakeWord's native chunk size (1280 samples / 80ms @ 16kHz) is used for the wake-detection phase's own stream; it is never reconciled with `listen_once()`'s 30ms VAD frames, because the two phases use separate stream opens (per the spec's rejected-alternatives reasoning).
- All new self-tests follow this repo's existing convention (`if __name__ == "__main__":` block with plain `assert` statements, e.g. `backtalk/connectivity.py`) — no pytest suite exists here and none should be introduced.

---

## Task 1: Dependency, config, and the `wakeword` module

**Files:**
- Modify: `backtalk/pyproject.toml` (dependencies list)
- Modify: `backtalk/backtalk/config.py:108` (DEFAULTS, after `"silence_ms": 480,`)
- Create: `backtalk/backtalk/wakeword.py`

**Interfaces:**
- Produces: `wakeword.RATE` (`int`, `16000`), `wakeword.FRAME_LEN` (`int`, `1280`), `wakeword.get_detector() -> WakeDetector` (lazy singleton, mirrors `ears.py`'s `_model`/`warm()` pattern), `WakeDetector.score(frame: np.ndarray) -> float`, `wakeword.chime() -> None`.

- [ ] **Step 1: Add the dependency**

In `backtalk/pyproject.toml`, add to the `dependencies` list (alphabetical position, after `"numpy>=1.26.0",`):

```toml
    # Wake-word gating for local hands-free mode: pretrained "hey_jarvis"
    # model, CPU-only onnxruntime backend, no GPU/tflite dependency.
    # See docs/superpowers/specs/2026-09-10-wake-word-gating-design.md.
    "openwakeword>=0.6.0",
```

- [ ] **Step 2: Install it**

Run: `cd backtalk && uv sync`
Expected: resolves and installs `openwakeword` (and its `onnxruntime` dependency) into `backtalk/.venv` with no errors.

- [ ] **Step 3: Add config defaults**

In `backtalk/backtalk/config.py`, immediately after the `"silence_ms": 480,` line (currently line 108) and its existing comment block, insert:

```python
    # Wake-word gating for hands-free listening: require "Hey Jarvis"
    # before a turn starts, same trigger the satellites already use.
    # OFF by default until confirmed working live — with this false,
    # hands-free mode behaves exactly as it does today (plain VAD,
    # no wake word). See docs/superpowers/specs/
    # 2026-09-10-wake-word-gating-design.md.
    "wake_word": {
        "enabled": False,
        # openWakeWord's own recommended default for "hey_jarvis".
        "threshold": 0.5,
        # After a reply, how long (seconds) to keep listening for a
        # follow-up before requiring the wake word again.
        "grace_window_s": 9,
        "chime": True,
    },
```

- [ ] **Step 4: Write the failing self-test**

Create `backtalk/backtalk/wakeword.py` with just this content first (the self-test, referencing code that doesn't exist yet):

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
"""Wake-word detection for local hands-free mode — openWakeWord's
pretrained "hey_jarvis" model, gating entry into ears.py's existing
VAD capture instead of replacing it. See docs/superpowers/specs/
2026-09-10-wake-word-gating-design.md."""
import threading

import numpy as np
import openwakeword
import sounddevice as sd
from openwakeword.model import Model

from backtalk.vlog import log

RATE = 16000
# openWakeWord's native chunk size: 80ms @ 16kHz. Deliberately NOT
# ears.py's 30ms VAD frame size — wait_for_wake() opens its own
# short-lived stream at this size, separate from listen_once()'s,
# so the two frame sizes never need to be reconciled.
FRAME_LEN = 1280
MODEL_NAME = "hey_jarvis"

if __name__ == "__main__":
    # Quick self-test / smoke check, not a full test suite.
    def _run():
        detector = get_detector()
        # Silence should never trigger: feed several seconds of digital
        # silence through the real detector and confirm the score never
        # crosses even a generous 0.3 (well below the 0.5 default), so
        # a genuinely quiet room can never false-trigger.
        silence = np.zeros(FRAME_LEN, dtype=np.int16)
        max_score = 0.0
        for _ in range(50):     # 50 * 80ms = 4s of silence
            score = detector.score(silence)
            max_score = max(max_score, score)
        assert max_score < 0.3, (
            f"silence scored {max_score:.3f} -- should be near zero")
        chime()   # smoke test: must not raise
        print(f"wakeword self-test: OK (max silence score {max_score:.3f})")

    _run()
```

- [ ] **Step 5: Run it to verify it fails**

Run: `cd backtalk && .venv\Scripts\python.exe -m backtalk.wakeword`
Expected: `NameError: name 'get_detector' is not defined` (or similar) — `get_detector` and `chime` don't exist yet.

- [ ] **Step 6: Implement `WakeDetector`, `get_detector()`, and `chime()`**

Insert this above the `if __name__ == "__main__":` block (the `import openwakeword`/`from openwakeword.model import Model` lines are already up top from Step 4's initial content):

```python
_detector = None
_detector_lock = threading.Lock()


class WakeDetector:
    def __init__(self):
        # Idempotent per openWakeWord's own on-disk caching -- safe to
        # call every process start, not just the first ever run.
        openwakeword.utils.download_models(model_names=[MODEL_NAME])
        self.model = Model(wakeword_models=[MODEL_NAME])

    def score(self, frame: np.ndarray) -> float:
        """frame: int16 mono 16kHz PCM, ideally FRAME_LEN samples."""
        prediction = self.model.predict(frame)
        return float(prediction.get(MODEL_NAME, 0.0))


def get_detector() -> WakeDetector:
    """Lazy singleton -- mirrors ears.py's _model/warm() pattern so the
    (one-time) model load can happen during the startup greeting rather
    than blocking the first real wake-word wait."""
    global _detector
    with _detector_lock:
        if _detector is None:
            log("[wakeword] loading hey_jarvis model...")
            _detector = WakeDetector()
            log("[wakeword] model ready")
    return _detector


def chime():
    """A short two-tone beep confirming the wake word was heard.
    Synthesized rather than a bundled asset file -- one less thing to
    ship or go missing."""
    t = np.linspace(0, 0.12, int(RATE * 0.12), endpoint=False)
    tone1 = 0.2 * np.sin(2 * np.pi * 880 * t)
    tone2 = 0.2 * np.sin(2 * np.pi * 1320 * t)
    gap = np.zeros(int(RATE * 0.02))
    audio = np.concatenate([tone1, gap, tone2]).astype(np.float32)
    sd.play(audio, RATE)
    sd.wait()
```

- [ ] **Step 7: Run it to verify it passes**

Run: `cd backtalk && .venv\Scripts\python.exe -m backtalk.wakeword`
Expected: downloads the `hey_jarvis` model on first run (one-time, needs internet), plays an audible chime, then prints `wakeword self-test: OK (max silence score 0.0XX)` with no assertion failure. **If the printed max silence score is at or above 0.3, do not proceed to Task 2** — that's a real signal the threshold or model load is wrong, not something to paper over.

- [ ] **Step 8: Commit**

```bash
cd backtalk
git add pyproject.toml backtalk/config.py backtalk/wakeword.py
git commit -m "feat: add openWakeWord hey_jarvis detector module

New backtalk/wakeword.py wraps openWakeWord's pretrained hey_jarvis
model behind a lazy singleton (get_detector()) plus a synthesized
chime helper. wake_word config block added, defaulting to disabled.
Not yet wired into the hands-free loop -- see the design spec.

See docs/superpowers/specs/2026-09-10-wake-word-gating-design.md."
```

---

## Task 2: `wait_for_wake()` in `ears.py`

**Files:**
- Modify: `backtalk/backtalk/ears.py:140-160` (`_open_mic`)
- Modify: `backtalk/backtalk/ears.py` (add `wait_for_wake`, after `listen_once`'s class, before `record_held` — i.e. after the current line 457/458 gap, before line 460)

**Interfaces:**
- Consumes: `wakeword.FRAME_LEN` (`int`), `wakeword.get_detector() -> WakeDetector`, `WakeDetector.score(frame) -> float`, `CFG["wake_word"]["threshold"]` (`float`).
- Produces: `ears.wait_for_wake(gate=None, abort=None, detector=None) -> bool` — `True` on a real detection, `False` if `abort()` fired first.

- [ ] **Step 1: Write the failing verification script**

This is a smoke check, not a pytest file (matches the repo's convention) — run it directly rather than adding it as permanent code, since `ears.py`'s existing `__main__` block already serves a different manual-demo purpose and shouldn't be disturbed:

```bash
cd backtalk && .venv\Scripts\python.exe -c "
from backtalk.ears import _open_mic
from backtalk import wakeword
with _open_mic(blocksize=wakeword.FRAME_LEN) as stream:
    assert stream.blocksize == wakeword.FRAME_LEN, stream.blocksize
print('blocksize OK')
"
```

- [ ] **Step 2: Run it to verify it fails**

Run the command above.
Expected: `TypeError: _open_mic() got an unexpected keyword argument 'blocksize'` — `_open_mic` doesn't accept that parameter yet.

- [ ] **Step 3: Parameterize `_open_mic`**

In `backtalk/backtalk/ears.py`, change (current lines 140-149):

```python
def _open_mic():
    """Open the capture stream on the configured mic.

    Degrades to the system default if that device will not open --
    unplugged between the lookup and the open, busy, or refusing the
    sample rate. The mic gets worse; it never goes mute.
    """
    dev = _mic_index()
    opts = dict(samplerate=RATE, channels=1, dtype="int16",
                blocksize=FRAME_LEN)
```

to:

```python
def _open_mic(blocksize: int = FRAME_LEN):
    """Open the capture stream on the configured mic, at `blocksize`
    samples per read (default FRAME_LEN, the 30ms VAD frame size;
    wait_for_wake() below passes wakeword.FRAME_LEN instead).

    Degrades to the system default if that device will not open --
    unplugged between the lookup and the open, busy, or refusing the
    sample rate. The mic gets worse; it never goes mute.
    """
    dev = _mic_index()
    opts = dict(samplerate=RATE, channels=1, dtype="int16",
                blocksize=blocksize)
```

(The rest of `_open_mic` and `_reopen_after_device_change` already just pass `**opts` / `opts` through unchanged — no further edits needed there.)

- [ ] **Step 4: Run it to verify it passes**

Re-run the Step 1 command.
Expected: `blocksize OK` printed, no assertion error.

- [ ] **Step 5: Add `wait_for_wake()`**

In `backtalk/backtalk/ears.py`, insert after the `Ears` class's closing (current line 457, the `return transcribe(...)` line, before the blank lines leading into `def record_held`):

```python
def wait_for_wake(gate=None, abort=None, detector=None) -> bool:
    """Block until the wake word fires. Returns True on detection,
    False if `abort()` returns True first (a mode switch, matching
    listen_once()'s own abort contract).

    Opens its OWN stream at openWakeWord's native 80ms chunk size --
    a separate phase from listen_once()'s 30ms VAD frames, opened and
    closed independently, so the two frame sizes are never reconciled
    (see the design spec's rejected two-stream and buffering
    alternatives)."""
    from backtalk import wakeword
    from backtalk.config import CFG
    if detector is None:
        detector = wakeword.get_detector()
    threshold = float(CFG.get("wake_word", {}).get("threshold", 0.5))
    with _open_mic(blocksize=wakeword.FRAME_LEN) as stream:
        while True:
            if abort and abort():
                return False
            block, _ = stream.read(wakeword.FRAME_LEN)
            if gate and gate():
                continue
            mono = block[:, 0].copy()
            if detector.score(mono) >= threshold:
                return True
```

- [ ] **Step 6: Verify it live**

Run: `cd backtalk && .venv\Scripts\python.exe -c "
from backtalk.ears import wait_for_wake
print('say Hey Jarvis in the next 15 seconds...')
import threading, time
stop_at = time.time() + 15
result = wait_for_wake(abort=lambda: time.time() > stop_at)
print('detected!' if result else 'timed out, no detection')
"`

Expected: say "Hey Jarvis" during the 15-second window and see `detected!` printed. This is a real human-in-the-loop check — say it for real, don't assume.

- [ ] **Step 7: Commit**

```bash
cd backtalk
git add backtalk/ears.py
git commit -m "feat: add wait_for_wake() to ears.py

_open_mic() now takes a blocksize parameter (default unchanged).
wait_for_wake() opens its own stream at openWakeWord's native 80ms
chunk size and blocks until the hey_jarvis model fires, mirroring
listen_once()'s gate/abort contract. listen_once() itself is
untouched. Not yet wired into main.py's hands-free loop.

See docs/superpowers/specs/2026-09-10-wake-word-gating-design.md."
```

---

## Task 3: Wire the state machine into `main.py`

**Files:**
- Modify: `backtalk/backtalk/main.py:78-79` (import)
- Modify: `backtalk/backtalk/main.py:919` (startup warm-up)
- Modify: `backtalk/backtalk/main.py:1311-1379` (the hands-free loop)

**Interfaces:**
- Consumes: `wakeword.get_detector`, `wakeword.chime` (Task 1); `ears.wait_for_wake` (Task 2); `ears.listen_once` (existing, unchanged).

- [ ] **Step 1: Add the import and startup warm-up**

In `backtalk/backtalk/main.py`, change (current lines 78-79):

```python
from backtalk.ears import (Ears, explain_audio_failure, record_held,
                           warm as warm_ears)
```

to:

```python
from backtalk import wakeword
from backtalk.ears import (Ears, explain_audio_failure, record_held,
                           wait_for_wake, warm as warm_ears)
```

Near the top of the file, alongside the existing `_MIC = {...}` module-level state (current line 121), add:

```python
# Set False if the wake-word model fails to load at startup -- the
# hands-free loop then runs with plain VAD for this session instead
# of crashing or silently hanging on every wait_for_wake() call. See
# the "model load failure" case in the design spec's Error Handling.
_WAKE = {"ready": True}


def _load_wake_detector():
    try:
        wakeword.get_detector()
    except Exception as e:
        log(f"[wakeword] hey_jarvis model failed to load ({e!r}) -- "
            f"hands-free mode will use plain VAD this session, "
            f"same as wake_word.enabled being false.")
        _WAKE["ready"] = False
```

Then, at current line 919 (`loop.run_in_executor(None, warm_ears)`), add immediately after:

```python
    loop.run_in_executor(None, warm_ears)
    if CFG.get("wake_word", {}).get("enabled"):
        loop.run_in_executor(None, _load_wake_detector)
```

- [ ] **Step 2: Replace the hands-free loop**

This is a behavior change with no isolated unit to pytest against (it's the live asyncio loop itself) — Steps 3-4 below are the real verification, run against the actual voice line with `wake_word.enabled` still `false` (must reproduce today's behavior exactly) before Task 4 turns it on.

In `backtalk/backtalk/main.py`, replace the block currently reading (lines 1311-1379, from `try:` through the final `if text and not await handle(text): return`):

```python
    try:
        # ONE loop, two mic modes, switchable live (_MIC). The talk key
        # is constructed and honored in BOTH modes: in hands-free
        # listening it is the interrupt and the guaranteed way to be
        # heard over room noise. The open mic joins the wait-set only
        # in "open" mode; a mode switch bumps _MIC["gen"], the abort
        # callable closes the in-flight open mic promptly, and any
        # capture born under an old gen is discarded unprocessed.
        ptt = PTTListener(CFG["ptt_key"])
        press_fut: asyncio.Future | None = None
        mic_fut: asyncio.Future | None = None
        mic_gen_seen = _MIC["gen"]
        # The open mic yields while the BUTTON records (or the double
        # capture would turn one held utterance into two turns), and,
        # without barge-in, while the mouth speaks.
        mic_gate = (lambda: _MIC["btn"]
                    or (not barge_in and mouth.speaking))
        mic_fails = 0
        while True:
            if _MIC["gen"] != mic_gen_seen:
                mic_gen_seen = _MIC["gen"]
                # consume futures that completed under the old mode so
                # a stale press or capture can't fire after a switch
                if press_fut is not None and press_fut.done():
                    press_fut.result(); press_fut = None
                if mic_fut is not None and mic_fut.done():
                    mic_fut.result(); mic_fut = None
            if typed_fut is None:
                typed_fut = loop.run_in_executor(None, typed_q.get)
            if press_fut is None:
                press_fut = loop.run_in_executor(None, ptt.wait_press)
            waiters = {press_fut, typed_fut}
            if _MIC["mode"] == "open":
                if mic_fut is None:
                    g = _MIC["gen"]
                    mic_fut = loop.run_in_executor(
                        None, lambda g=g: (g, ears.listen_once(
                            gate=mic_gate,
                            abort=lambda: _MIC["gen"] != g)))
                waiters.add(mic_fut)
            done, _ = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED)
            if typed_fut in done:
                text = typed_fut.result(); typed_fut = None
                if text and not await handle(text):
                    return
                continue
            if mic_fut is not None and mic_fut in done:
                try:
                    g, text = mic_fut.result()
                except Exception as e:
                    mic_fut = None
                    mic_fails += 1
                    if not explain_audio_failure(e):
                        log(f"[ears] open mic failed ({mic_fails}): {e!r}")
                    if mic_fails >= 3:
                        _MIC["mode"] = "ptt"
                        _MIC["gen"] += 1
                        mic_fails = 0
                        mouth.say("The open microphone keeps failing, "
                                  "so I'm switching to push to talk. "
                                  "Hold the key to reach me, and "
                                  "check this window for the error.")
                    continue
                mic_fut = None
                if g != _MIC["gen"]:
                    continue             # captured before a switch
                if text and not await handle(text):
                    return
```

with:

```python
    # _WAKE["ready"] starts True and only ever flips False from
    # _load_wake_detector() above -- by the time this loop starts
    # (well after the brain-connect + warmup-ping awaits), the
    # detector's own small CPU model load has had ample time to
    # finish or fail, so this reads the real outcome, not a race.
    WAKE_ENABLED = (bool(CFG.get("wake_word", {}).get("enabled", False))
                     and _WAKE["ready"])
    GRACE_S = float(CFG.get("wake_word", {}).get("grace_window_s", 9))
    CHIME_ON = bool(CFG.get("wake_word", {}).get("chime", True))

    try:
        # ONE loop, two mic modes, switchable live (_MIC). The talk key
        # is constructed and honored in BOTH modes: in hands-free
        # listening it is the interrupt and the guaranteed way to be
        # heard over room noise. The open mic joins the wait-set only
        # in "open" mode; a mode switch bumps _MIC["gen"], the abort
        # callable closes the in-flight open mic promptly, and any
        # capture born under an old gen is discarded unprocessed.
        #
        # When WAKE_ENABLED, "open" mode cycles three states:
        # "wake" (must hear "Hey Jarvis" next) -> "capture" (wake just
        # fired, capturing that command, no timeout) -> "grace"
        # (a reply just finished; listening for an optional follow-up
        # for GRACE_S before re-arming "wake"). WAKE_ENABLED false
        # reproduces today's behavior exactly -- mic_state never
        # leaves "wake" and every call behaves like the old bare
        # listen_once(gate=mic_gate, abort=...).
        ptt = PTTListener(CFG["ptt_key"])
        press_fut: asyncio.Future | None = None
        mic_fut: asyncio.Future | None = None
        mic_gen_seen = _MIC["gen"]
        # The open mic yields while the BUTTON records (or the double
        # capture would turn one held utterance into two turns), and,
        # without barge-in, while the mouth speaks.
        mic_gate = (lambda: _MIC["btn"]
                    or (not barge_in and mouth.speaking))
        mic_fails = 0
        mic_state = "wake"
        while True:
            if _MIC["gen"] != mic_gen_seen:
                mic_gen_seen = _MIC["gen"]
                mic_state = "wake"     # a fresh mode entry always re-arms
                # consume futures that completed under the old mode so
                # a stale press or capture can't fire after a switch
                if press_fut is not None and press_fut.done():
                    press_fut.result(); press_fut = None
                if mic_fut is not None and mic_fut.done():
                    mic_fut.result(); mic_fut = None
            if typed_fut is None:
                typed_fut = loop.run_in_executor(None, typed_q.get)
            if press_fut is None:
                press_fut = loop.run_in_executor(None, ptt.wait_press)
            waiters = {press_fut, typed_fut}
            if _MIC["mode"] == "open":
                if mic_fut is None:
                    g = _MIC["gen"]
                    if WAKE_ENABLED and mic_state == "wake":
                        mic_fut = loop.run_in_executor(
                            None, lambda g=g: (g, "wake", wait_for_wake(
                                gate=mic_gate,
                                abort=lambda: _MIC["gen"] != g)))
                    else:
                        t = GRACE_S if (WAKE_ENABLED and
                                        mic_state == "grace") else None
                        mic_fut = loop.run_in_executor(
                            None, lambda g=g, t=t: (g, "listen",
                                ears.listen_once(
                                    gate=mic_gate, timeout_s=t,
                                    abort=lambda: _MIC["gen"] != g)))
                waiters.add(mic_fut)
            done, _ = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED)
            if typed_fut in done:
                text = typed_fut.result(); typed_fut = None
                if text and not await handle(text):
                    return
                continue
            if mic_fut is not None and mic_fut in done:
                try:
                    g, phase, result = mic_fut.result()
                except Exception as e:
                    mic_fut = None
                    mic_fails += 1
                    if not explain_audio_failure(e):
                        log(f"[ears] open mic failed ({mic_fails}): {e!r}")
                    if mic_fails >= 3:
                        _MIC["mode"] = "ptt"
                        _MIC["gen"] += 1
                        mic_fails = 0
                        mouth.say("The open microphone keeps failing, "
                                  "so I'm switching to push to talk. "
                                  "Hold the key to reach me, and "
                                  "check this window for the error.")
                    continue
                mic_fut = None
                if g != _MIC["gen"]:
                    continue             # captured before a switch
                if phase == "wake":
                    if result:           # wake word fired
                        if CHIME_ON:
                            await loop.run_in_executor(None, wakeword.chime)
                        mic_state = "capture"
                    continue             # either way, re-schedule next loop
                # phase == "listen"
                text = result
                if text:
                    if WAKE_ENABLED:
                        mic_state = "grace"   # a reply is about to play;
                                               # listen for a follow-up after
                    if not await handle(text):
                        return
                elif WAKE_ENABLED and mic_state == "grace":
                    mic_state = "wake"        # grace window timed out: re-arm
```

- [ ] **Step 3: Verify the disabled path is unchanged**

With `wake_word.enabled` still `false` in `backtalk.json` (the config default from Task 1 — leave it alone for this step), launch backtalk in hands-free mode:

Run: `cd backtalk && .venv\Scripts\python.exe -m backtalk.main --open-mic`

Speak a normal command with no wake word. Expected: identical behavior to before this task — it responds without requiring "Hey Jarvis", exactly as hands-free mode has always worked. This confirms the `WAKE_ENABLED=False` path truly reproduces the old code, not just in theory.

- [ ] **Step 4: Commit**

```bash
cd backtalk
git add backtalk/main.py
git commit -m "feat: wire wake-word state machine into hands-free loop

main.py's open-mic loop now cycles wake -> capture -> grace when
wake_word.enabled is true, calling wait_for_wake()/listen_once()
accordingly. With it false (the shipped default), behavior is
byte-for-byte the old always-listening VAD loop -- verified live.

See docs/superpowers/specs/2026-09-10-wake-word-gating-design.md."
```

---

## Task 4: Enable it, live acceptance test, final commit

**Files:**
- Modify: `backtalk/backtalk.json` (local, gitignored — not committed)

**Interfaces:** none — this task only exercises what Tasks 1-3 built.

- [ ] **Step 1: Enable it**

In the live `backtalk/backtalk.json` (not `config.py` — this is the per-machine override file, already gitignored per existing project convention), set:

```json
{
  "wake_word": { "enabled": true }
}
```

(Merge into the existing file rather than replacing it — `config.py`'s `load()` deep-merges dict-valued keys.)

- [ ] **Step 2: Restart and confirm the wake gate**

Restart the voice line in hands-free mode (`--open-mic` or however it's normally launched). Say something in the room *without* "Hey Jarvis" first.

Expected: silence — no transcription, no reply, nothing sent to Claude. This is the actual behavior change Sir asked for; confirm it for real, don't assume from the code review alone.

- [ ] **Step 3: Confirm the trigger + chime**

Say "Hey Jarvis," then a real command.

Expected: the chime plays immediately on "Hey Jarvis," then the command is captured and answered normally.

- [ ] **Step 4: Confirm the grace window**

Right after a reply finishes, speak again within ~9 seconds *without* saying "Hey Jarvis."

Expected: it responds (grace window caught it). Then wait past 9 seconds of silence and speak without the wake word.

Expected: silence again (grace window expired, back to requiring "Hey Jarvis").

- [ ] **Step 5: Confirm push-to-talk is untouched**

Switch to push-to-talk ("push to talk mode") and hold the key to speak.

Expected: works exactly as before, no wake word needed — push-to-talk was never touched by this change.

- [ ] **Step 6: Commit the spec and plan docs**

These were written earlier in the session but held per the double-confirm rule — commit them now alongside confirmation the feature works:

```bash
cd backtalk
git add docs/superpowers/specs/2026-09-10-wake-word-gating-design.md \
        docs/superpowers/plans/2026-09-10-wake-word-gating.md
git commit -m "docs: wake-word gating design spec and implementation plan

See docs/superpowers/specs/2026-09-10-wake-word-gating-design.md."
```

- [ ] **Step 7: Vault checkpoint**

Update `Active Priorities.md` and today's daily note: this feature is now live-confirmed, not just built — per the "document the moment it ships" rule, record the honest status (live and confirmed working, per Sir's own test) in both places.
