"""warm_voicebox must render one throwaway sentence, never raise, and do
nothing when Voicebox is not configured. The startup task must retry until
Voicebox answers and then stop."""
import asyncio
import unittest
from unittest import mock

from backtalk import main, mouth


class WarmVoiceboxTests(unittest.TestCase):
    def test_disabled_makes_no_request(self):
        with mock.patch.object(mouth, "_voicebox_ready", return_value=False), \
                mock.patch.object(mouth, "_stream_voicebox") as s:
            self.assertFalse(mouth.warm_voicebox())
            s.assert_not_called()

    def test_success_renders_and_discards(self):
        gen = iter([(24000, b"x")])
        with mock.patch.object(mouth, "_voicebox_ready", return_value=True), \
                mock.patch.object(mouth, "_stream_voicebox",
                                  return_value=gen) as s:
            self.assertTrue(mouth.warm_voicebox())
            s.assert_called_once()

    def test_failure_returns_false_never_raises(self):
        with mock.patch.object(mouth, "_voicebox_ready", return_value=True), \
                mock.patch.object(mouth, "_stream_voicebox",
                                  side_effect=ConnectionError("down")):
            self.assertFalse(mouth.warm_voicebox())


class StartupTaskTests(unittest.TestCase):
    def setUp(self):
        # Keep the test's "gave up" / "warm" lines out of the live log.
        p = mock.patch.object(main, "log")
        p.start()
        self.addCleanup(p.stop)

    def test_retries_until_success_then_stops(self):
        results = iter([False, False, True])
        calls = []

        def fake():
            calls.append(1)
            return next(results)

        with mock.patch.object(mouth, "warm_voicebox", side_effect=fake):
            asyncio.run(main._warm_voicebox_at_startup(attempts=5,
                                                       retry_s=0))
        self.assertEqual(len(calls), 3)

    def test_gives_up_after_attempts(self):
        with mock.patch.object(mouth, "warm_voicebox",
                               return_value=False) as w:
            asyncio.run(main._warm_voicebox_at_startup(attempts=3,
                                                       retry_s=0))
        self.assertEqual(w.call_count, 3)


if __name__ == "__main__":
    unittest.main()
