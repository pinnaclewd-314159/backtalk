"""Remote Kokoro: synth_stream tries Voicebox, then the remote Kokoro
service, then the in-process Kokoro. A remote failure falls through for
that sentence and opens a down window. Disabled, it must be inert."""
import time
import unittest
from unittest import mock

import numpy as np

from backtalk import mouth

ON = {"enabled": True, "url": "http://mac:8721", "timeout_s": 5.0,
      "down_s": 30.0}
AUDIO = (np.arange(2400) % 200).astype("<i2")


class _Resp:
    def __init__(self, content=b"", rate="24000"):
        self.content = content
        self.headers = {"X-Sample-Rate": rate}

    def raise_for_status(self):
        pass


class RemoteKokoroTests(unittest.TestCase):
    def setUp(self):
        mouth._kokoro_remote_down_until = 0.0
        mouth._kokoro_remote_up = None
        for p in (mock.patch.object(mouth, "log"),
                  mock.patch.dict(mouth.CFG, {"tts_remote": dict(ON)}),
                  mock.patch.object(mouth, "_voicebox_ready", return_value=False),
                  mock.patch.object(mouth, "_elevenlabs_ready", return_value=False)):
            p.start()
            self.addCleanup(p.stop)

    def test_success_returns_rate_and_audio_and_sends_voice(self):
        with mock.patch("httpx.post",
                        return_value=_Resp(AUDIO.tobytes())) as post:
            rate, pcm = mouth._stream_kokoro_remote("Hello there.")
        self.assertEqual(rate, 24000)
        self.assertEqual(pcm.tolist(), AUDIO.tolist())
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["text"], "Hello there.")
        self.assertEqual(body["voice"], mouth.CFG["voice"])

    def test_failure_returns_none_and_opens_the_down_window(self):
        with mock.patch("httpx.post", side_effect=ConnectionError("x")) as post:
            self.assertIsNone(mouth._stream_kokoro_remote("Hi."))
            self.assertIsNone(mouth._stream_kokoro_remote("Hi."))
        self.assertEqual(post.call_count, 1)
        self.assertGreater(mouth._kokoro_remote_down_until, time.time() + 25)

    def test_empty_audio_counts_as_failure(self):
        with mock.patch("httpx.post", return_value=_Resp(b"")):
            self.assertIsNone(mouth._stream_kokoro_remote("Hi."))

    def test_retries_after_the_down_window(self):
        mouth._kokoro_remote_down_until = time.time() - 1
        with mock.patch("httpx.post",
                        return_value=_Resp(AUDIO.tobytes())) as post:
            self.assertIsNotNone(mouth._stream_kokoro_remote("Hi."))
        post.assert_called_once()

    def test_disabled_is_inert(self):
        with mock.patch.dict(mouth.CFG, {"tts_remote": {"enabled": False,
                                                        "url": ON["url"]}}), \
                mock.patch("httpx.post") as post:
            self.assertIsNone(mouth._stream_kokoro_remote("Hi."))
        post.assert_not_called()

    def test_synth_stream_uses_remote_and_skips_local(self):
        with mock.patch.object(mouth, "_stream_kokoro_remote",
                               return_value=(24000, AUDIO)), \
                mock.patch.object(mouth, "_stream_kokoro") as local:
            out = list(mouth.synth_stream("Hello."))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], 24000)
        local.assert_not_called()

    def test_synth_stream_falls_back_to_local_kokoro(self):
        with mock.patch.object(mouth, "_stream_kokoro_remote",
                               return_value=None), \
                mock.patch.object(mouth, "_stream_kokoro",
                                  return_value=iter([AUDIO])) as local:
            out = list(mouth.synth_stream("Hello."))
        self.assertEqual(out, [(mouth.KOKORO_RATE, AUDIO)])
        local.assert_called_once()

    def test_voicebox_still_comes_first(self):
        with mock.patch.object(mouth, "_voicebox_ready", return_value=True), \
                mock.patch.object(mouth, "_stream_voicebox",
                                  return_value=iter([(24000, AUDIO)])), \
                mock.patch.object(mouth, "_stream_kokoro_remote") as remote:
            out = list(mouth.synth_stream("Hello."))
        self.assertEqual(len(out), 1)
        remote.assert_not_called()

    def test_voicebox_failure_falls_to_the_remote_not_the_local(self):
        with mock.patch.object(mouth, "_voicebox_ready", return_value=True), \
                mock.patch.object(mouth, "_stream_voicebox",
                                  side_effect=ConnectionError("down")), \
                mock.patch.object(mouth, "_stream_kokoro_remote",
                                  return_value=(24000, AUDIO)), \
                mock.patch.object(mouth, "_stream_kokoro") as local:
            out = list(mouth.synth_stream("Hello."))
        self.assertEqual(len(out), 1)
        local.assert_not_called()


if __name__ == "__main__":
    unittest.main()
