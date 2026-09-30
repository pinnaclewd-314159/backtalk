"""The rolling utterance buffer saves the exact audio Whisper hears plus the
result, keeps only the newest N clips inside an age limit, and can never
break transcription: a missing or unwritable folder logs on the state change
and the utterance is transcribed as usual. Off, it must be inert."""
import json
import os
import shutil
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from backtalk import ears, utterances

PCM = (np.arange(16000) % 300 - 150).astype(np.int16)        # 1 s
ON_REMOTE = {"enabled": True, "url": "http://mac:8720", "timeout_s": 5.0,
             "down_s": 30.0}


class _Resp:
    def __init__(self, j):
        self._j = j

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


class BufferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dir = self.tmp / "clips"
        utterances._saving = None
        self.cfg = {"enabled": True, "dir": str(self.dir), "keep": 20,
                    "max_age_hours": 24}
        self.log = mock.patch.object(utterances, "log")
        self.logm = self.log.start()
        self.addCleanup(self.log.stop)
        p = mock.patch.dict(utterances.CFG, {"utterance_buffer": self.cfg})
        p.start()
        self.addCleanup(p.stop)

    def _save(self, text="hello"):
        t = utterances.record(PCM)
        utterances.note(t, text, "remote", 100, True)
        return t

    def test_disabled_is_inert(self):
        self.cfg["enabled"] = False
        self.assertIsNone(utterances.record(PCM))
        utterances.note(None, "x", "remote", 1, False)
        self.assertFalse(self.dir.exists())
        self.logm.assert_not_called()

    def test_record_writes_a_valid_16k_mono_wav_with_the_same_samples(self):
        t = utterances.record(PCM)
        with wave.open(str(self.dir / t["file"]), "rb") as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()),
                             (1, 2, 16000))
            got = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        self.assertEqual(got.tolist(), PCM.tolist())
        self.assertEqual(t["secs"], 1.0)

    def test_note_appends_an_index_line_with_the_result(self):
        t = self._save("turn off the lights")
        lines = (self.dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        e = json.loads(lines[0])
        self.assertEqual((e["file"], e["text"], e["backend"], e["ms"],
                          e["hallucination_filter"]),
                         (t["file"], "turn off the lights", "remote", 100, True))

    def test_empty_and_failed_results_are_kept_too(self):
        t = utterances.record(PCM)
        utterances.note(t, "", "local", 5, True)
        t2 = utterances.record(PCM)
        utterances.note(t2, None, "local", 5, False, error="RuntimeError")
        rows = utterances.recent()
        self.assertEqual({r["text"] for r in rows}, {"", None})
        self.assertIn("RuntimeError", {r.get("error") for r in rows})

    def test_keeps_only_the_newest_n_and_trims_the_index(self):
        self.cfg["keep"] = 3
        names = []
        for i in range(6):
            names.append(self._save(f"t{i}")["file"])
            time.sleep(0.002)       # names carry milliseconds
        left = sorted(p.name for p in self.dir.glob("*.wav"))
        self.assertEqual(left, sorted(names)[-3:])
        idx = [json.loads(l)["text"] for l in
               (self.dir / "index.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(idx, ["t3", "t4", "t5"])
        self.assertEqual([r["text"] for r in utterances.recent()], ["t5", "t4", "t3"])

    def test_deletes_clips_older_than_the_age_limit(self):
        old = self._save("old")
        time.sleep(0.002)
        new = self._save("new")
        past = time.time() - 25 * 3600
        os.utime(self.dir / old["file"], (past, past))
        utterances.note(new, "new2", "remote", 1, False)      # triggers a prune
        self.assertFalse((self.dir / old["file"]).exists())
        self.assertTrue((self.dir / new["file"]).exists())

    def test_unwritable_folder_never_raises_and_logs_only_on_change(self):
        blocker = self.tmp / "afile"
        blocker.write_text("x")
        self.cfg["dir"] = str(blocker / "sub")               # cannot be created
        self.assertIsNone(utterances.record(PCM))
        self.assertIsNone(utterances.record(PCM))
        self.assertEqual(self.logm.call_count, 1)
        self.assertIn("unavailable", self.logm.call_args[0][0])
        self.cfg["dir"] = str(self.dir)                       # it comes back
        self.assertIsNotNone(utterances.record(PCM))
        self.assertEqual(self.logm.call_count, 2)
        self.assertIn("saving clips", self.logm.call_args[0][0])

    def test_recent_ignores_index_rows_whose_clip_is_gone(self):
        t = self._save("gone")
        (self.dir / t["file"]).unlink()
        self.assertEqual(utterances.recent(), [])


class TranscribeHookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        utterances._saving = None
        ears._remote_down_until = 0.0
        ears._remote_up = None
        cfg = {"enabled": True, "dir": str(self.tmp), "keep": 20, "max_age_hours": 24}
        for p in (mock.patch.object(utterances, "log"), mock.patch.object(ears, "log"),
                  mock.patch.dict(utterances.CFG, {"utterance_buffer": cfg}),
                  mock.patch.dict(ears.CFG, {"stt_remote": dict(ON_REMOTE)})):
            p.start()
            self.addCleanup(p.stop)

    def _rows(self):
        return utterances.recent()

    def test_a_remote_transcription_is_saved_as_remote(self):
        with mock.patch("httpx.post", return_value=_Resp({"text": "hi there"})):
            self.assertEqual(ears.transcribe(PCM, reject_hallucinations=True), "hi there")
        (row,) = self._rows()
        self.assertEqual((row["text"], row["backend"], row["hallucination_filter"]),
                         ("hi there", "remote", True))
        self.assertTrue((self.tmp / row["file"]).exists())

    def test_a_local_fallback_is_saved_as_local(self):
        fake = mock.Mock()
        fake.transcribe.return_value = (
            [mock.Mock(text=" local ", no_speech_prob=0.0, compression_ratio=1.0)], None)
        with mock.patch("httpx.post", side_effect=ConnectionError("down")), \
                mock.patch.object(ears, "warm", return_value=fake), \
                mock.patch.object(ears, "_backend", "faster-whisper"):
            self.assertEqual(ears.transcribe(PCM), "local")
        (row,) = self._rows()
        self.assertEqual((row["text"], row["backend"]), ("local", "local"))

    def test_the_backend_flag_does_not_leak_between_calls(self):
        with mock.patch("httpx.post", return_value=_Resp({"text": "one"})):
            ears.transcribe(PCM)
        fake = mock.Mock()
        fake.transcribe.return_value = (
            [mock.Mock(text=" two ", no_speech_prob=0.0, compression_ratio=1.0)], None)
        with mock.patch.object(ears, "_transcribe_remote", return_value=None), \
                mock.patch.object(ears, "warm", return_value=fake), \
                mock.patch.object(ears, "_backend", "faster-whisper"):
            ears.transcribe(PCM)
        self.assertEqual([r["backend"] for r in self._rows()], ["local", "remote"])

    def test_a_crash_is_recorded_and_still_raised(self):
        with mock.patch.object(ears, "_transcribe_inner", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                ears.transcribe(PCM)
        (row,) = self._rows()
        self.assertEqual((row["text"], row["error"]), (None, "RuntimeError"))

    def test_an_unwritable_buffer_does_not_break_transcription(self):
        blocker = self.tmp / "afile"
        blocker.write_text("x")
        with mock.patch.dict(utterances.CFG, {"utterance_buffer": {
                "enabled": True, "dir": str(blocker / "sub"), "keep": 20,
                "max_age_hours": 24}}), \
                mock.patch("httpx.post", return_value=_Resp({"text": "still works"})):
            self.assertEqual(ears.transcribe(PCM), "still works")


if __name__ == "__main__":
    unittest.main()
