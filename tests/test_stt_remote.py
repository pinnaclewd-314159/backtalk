"""Remote STT: ears.transcribe tries the remote Whisper service first, falls
back to the local model for that utterance on any failure, and skips the
remote for `down_s` afterwards. Disabled, it must be inert."""
import time
import unittest
from unittest import mock

import numpy as np

from backtalk import ears

PCM = (np.ones(1600) * 1000).astype(np.int16)
ON = {"enabled": True, "url": "http://mac:8720", "timeout_s": 5.0,
      "down_s": 30.0}


class _Resp:
    def __init__(self, j):
        self._j = j

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


class RemoteSttTests(unittest.TestCase):
    def setUp(self):
        ears._remote_down_until = 0.0
        ears._remote_up = None
        for p in (mock.patch.object(ears, "log"),
                  # ears.transcribe also feeds the rolling utterance buffer;
                  # keep these synthetic clips out of the real one.
                  mock.patch.dict(ears.CFG, {"stt_remote": dict(ON),
                                             "utterance_buffer": {"enabled": False}})):
            p.start()
            self.addCleanup(p.stop)

    def test_success_uses_remote_and_passes_the_flag(self):
        with mock.patch("httpx.post",
                        return_value=_Resp({"text": " Turn on the lights. "})) as post:
            out = ears._transcribe_remote(PCM, True)
        self.assertEqual(out, "Turn on the lights.")
        self.assertEqual(post.call_args.kwargs["params"],
                         {"reject_hallucinations": 1})
        self.assertEqual(len(post.call_args.kwargs["content"]), PCM.size * 2)

    def test_failure_returns_none_and_opens_the_down_window(self):
        with mock.patch("httpx.post", side_effect=ConnectionError("down")) as post:
            self.assertIsNone(ears._transcribe_remote(PCM, False))
            self.assertIsNone(ears._transcribe_remote(PCM, False))
        self.assertEqual(post.call_count, 1)        # second call skipped
        self.assertGreater(ears._remote_down_until, time.time() + 25)

    def test_retries_after_the_down_window(self):
        ears._remote_down_until = time.time() - 1
        with mock.patch("httpx.post",
                        return_value=_Resp({"text": "hello"})) as post:
            self.assertEqual(ears._transcribe_remote(PCM, False), "hello")
        post.assert_called_once()

    def test_disabled_is_inert(self):
        with mock.patch.dict(ears.CFG, {"stt_remote": {"enabled": False,
                                                       "url": ON["url"]}}), \
                mock.patch("httpx.post") as post, mock.patch("httpx.get") as get:
            self.assertIsNone(ears._transcribe_remote(PCM, False))
            self.assertFalse(ears._remote_healthy())
        post.assert_not_called()
        get.assert_not_called()

    def test_transcribe_returns_remote_text_without_touching_local(self):
        with mock.patch.object(ears, "_transcribe_remote", return_value="ok"), \
                mock.patch.object(ears, "warm") as warm:
            self.assertEqual(ears.transcribe(PCM), "ok")
        warm.assert_not_called()

    def test_transcribe_falls_back_to_local_when_remote_returns_none(self):
        fake = mock.Mock()
        fake.transcribe.return_value = (
            [mock.Mock(text=" local ", no_speech_prob=0.0,
                       compression_ratio=1.0)], None)
        with mock.patch.object(ears, "_transcribe_remote", return_value=None), \
                mock.patch.object(ears, "warm", return_value=fake) as warm, \
                mock.patch.object(ears, "_backend", "faster-whisper"):
            self.assertEqual(ears.transcribe(PCM), "local")
        warm.assert_called_once_with(force_local=True)

    def test_healthy_remote_skips_the_local_model_load(self):
        with mock.patch.object(ears, "check_microphone"), \
                mock.patch.object(ears, "_remote_healthy", return_value=True), \
                mock.patch.object(ears, "_model", None):
            self.assertIsNone(ears.warm())
            self.assertIsNone(ears._model)

    def test_health_requires_the_model_loaded(self):
        with mock.patch("httpx.get",
                        return_value=_Resp({"ok": True, "warm": False})):
            self.assertFalse(ears._remote_healthy())
        with mock.patch("httpx.get",
                        return_value=_Resp({"ok": True, "warm": True})):
            self.assertTrue(ears._remote_healthy())


if __name__ == "__main__":
    unittest.main()
