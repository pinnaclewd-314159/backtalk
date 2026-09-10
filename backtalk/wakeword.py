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
