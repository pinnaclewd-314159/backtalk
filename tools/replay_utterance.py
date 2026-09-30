"""Replay a saved utterance against both Whisper engines.

    python tools/replay_utterance.py            list the recent clips
    python tools/replay_utterance.py 1          replay the newest clip
    python tools/replay_utterance.py 3          ...the third newest
    python tools/replay_utterance.py NAME.wav   ...a clip by file name

Read-only. It sends the clip to the remote Whisper service (config
`stt_remote`) and to this machine's own Whisper (config `stt_model` /
`stt_device`), and prints both next to what backtalk heard at the time.
The local model is loaded only for the run (about 1.5 GB of GPU for
medium.en), then released. Clips come from the rolling buffer in
backtalk/utterances.py.
"""
import json
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtalk import utterances  # noqa: E402
from backtalk.config import CFG  # noqa: E402


def load(name: str) -> np.ndarray:
    with wave.open(str(utterances.buffer_dir() / name), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise SystemExit(f"{name}: expected 16 kHz mono, got {w.getframerate()} Hz x{w.getnchannels()}")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def remote(pcm: np.ndarray):
    c = CFG.get("stt_remote") or {}
    if not c.get("url"):
        return "(no stt_remote url configured)", 0.0
    req = urllib.request.Request(c["url"].rstrip("/") + "/transcribe?reject_hallucinations=0",
                                 data=pcm.astype("<i2").tobytes(), method="POST")
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["text"], time.time() - t
    except Exception as e:
        return f"(remote failed: {type(e).__name__}: {e})", time.time() - t


def local(pcm: np.ndarray):
    from backtalk import ears
    ears._add_cuda_dll_dirs()
    from faster_whisper import WhisperModel
    t = time.time()
    try:
        m = WhisperModel(CFG["stt_model"], device=CFG["stt_device"], compute_type=CFG["stt_compute"])
        lang = "en" if CFG["stt_model"].endswith(".en") else None
        segs = list(m.transcribe(pcm.astype(np.float32) / 32768.0, temperature=0.0, language=lang)[0])
        return "".join(s.text for s in segs).strip(), time.time() - t
    except Exception as e:
        return f"(local failed: {type(e).__name__}: {e})", time.time() - t


def main(argv):
    clips = utterances.recent(50)
    if not clips:
        print(f"no clips in {utterances.buffer_dir()} (buffer off, or nothing said yet)")
        return 1
    if not argv:
        for i, e in enumerate(clips, 1):
            print(f"{i:2d}  {e['ts']}  {e['secs']:5.1f}s  {e['backend']:6s}  {e.get('text')!r}")
        return 0
    arg = argv[0]
    pick = clips[int(arg) - 1] if arg.isdigit() and 1 <= int(arg) <= len(clips) else \
        next((e for e in clips if e["file"] == arg), None)
    if pick is None:
        print(f"no such clip: {arg}")
        return 1
    pcm = load(pick["file"])
    print(f"{pick['file']}  {pick['secs']}s  saved {pick['ts']}")
    print(f"  heard at the time ({pick['backend']}, filter {'on' if pick.get('hallucination_filter') else 'off'}): {pick.get('text')!r}")
    text, dt = remote(pcm)
    print(f"  remote Whisper  {dt:5.2f}s: {text!r}")
    text, dt = local(pcm)
    print(f"  local Whisper   {dt:5.2f}s (includes model load): {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
