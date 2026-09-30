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
"""A rolling buffer of the last few utterances, so a mishear can be replayed.

ears.transcribe() drops the exact 16 kHz mono audio it is about to hand to
Whisper here, then records what came back. Without the audio a wrong
transcript cannot be diagnosed afterwards (2026-09-29: a reply of "home
coordinates" arrived as "PUM coordinates" and there was nothing to replay
against the other Whisper). Replay with tools/replay_utterance.py.

Off by default: these are recordings of whoever is near the microphone.
`utterance_buffer` in the config turns it on and names the folder. Only the
newest `keep` clips are kept, and nothing older than `max_age_hours`.

NOTHING HERE MAY BREAK TRANSCRIPTION. record() and note() never raise: a
folder that is missing or unwritable (a pooled drive that has not come up,
a full disk) logs one line when the state changes and the utterance is
transcribed as usual.
"""
import json
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from backtalk.config import CFG
from backtalk.vlog import log

RATE = 16000
INDEX = "index.jsonl"

_lock = threading.Lock()
_saving = None      # last logged state: True / False / None (unknown)


def _cfg() -> dict:
    c = CFG.get("utterance_buffer") or {}
    return c if c.get("enabled") else {}


def buffer_dir() -> Path:
    """The configured folder, whether or not the buffer is switched on (the
    replay tool reads clips a previous run saved)."""
    d = str((CFG.get("utterance_buffer") or {}).get("dir") or "").strip()
    return Path(d) if d else Path(__file__).resolve().parent.parent / "logs" / "utterances"


def _note_state(ok: bool, where: Path, why: str = ""):
    """Log only when the state changes."""
    global _saving
    if ok != _saving:
        _saving = ok
        log(f"[utterances] saving clips to {where}" if ok else
            f"[utterances] clip buffer unavailable ({why}) - clips are not being saved")


def record(pcm: np.ndarray):
    """Save `pcm` (int16, mono, 16 kHz) as a WAV. Returns a ticket for note(),
    or None when the buffer is off or the save failed. Never raises."""
    c = _cfg()
    if not c:
        return None
    where = buffer_dir()
    try:
        with _lock:
            where.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            name = now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}.wav"
            data = np.asarray(pcm).astype("<i2", copy=False)
            with wave.open(str(where / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(data.tobytes())
        _note_state(True, where)
        return {"file": name, "ts": now.isoformat(timespec="seconds"),
                "secs": round(data.size / RATE, 2), "dir": where}
    except Exception as e:
        _note_state(False, where, f"{type(e).__name__}: {str(e)[:60]}")
        return None


def note(ticket, text, backend: str, ms: int, rejecting: bool, error: str = ""):
    """Append what Whisper made of the clip, then prune. Never raises."""
    if not ticket:
        return
    c = _cfg()
    if not c:
        return
    where = ticket["dir"]
    try:
        entry = {"file": ticket["file"], "ts": ticket["ts"], "secs": ticket["secs"],
                 "text": text, "backend": backend, "ms": ms,
                 "hallucination_filter": bool(rejecting)}
        if error:
            entry["error"] = error
        with _lock:
            with (where / INDEX).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _prune(where, int(c.get("keep", 20)), float(c.get("max_age_hours", 24)))
    except Exception as e:
        _note_state(False, where, f"{type(e).__name__}: {str(e)[:60]}")


def _prune(where: Path, keep: int, max_age_hours: float):
    """Keep the newest `keep` clips, none older than the limit, and trim the
    index to the clips that remain. Caller holds _lock."""
    clips = sorted(where.glob("*.wav"), key=lambda p: p.name, reverse=True)
    cutoff = time.time() - max_age_hours * 3600
    for i, p in enumerate(clips):
        if i >= keep or p.stat().st_mtime < cutoff:
            try:
                p.unlink()
            except OSError:
                pass
    left = {p.name for p in where.glob("*.wav")}
    idx = where / INDEX
    if not idx.exists():
        return
    lines = idx.read_text(encoding="utf-8").splitlines()
    kept = []
    for ln in lines:
        try:
            if json.loads(ln).get("file") in left:
                kept.append(ln)
        except ValueError:
            continue
    if len(kept) != len(lines):
        idx.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def recent(limit: int = 20):
    """Newest-first index entries that still have their clip (for the replay tool)."""
    where = buffer_dir()
    idx = where / INDEX
    if not idx.exists():
        return []
    out = []
    for ln in idx.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if (where / e.get("file", "")).exists():
            out.append(e)
    return list(reversed(out))[:limit]
