"""Idle tracking must start from construction (a freshly-launched voice
line isn't "idle" from epoch zero), and mark_activity must reset the
clock every time it's called, from any source."""
import unittest

from backtalk.hygiene import SessionHygiene


class HygieneActivityTest(unittest.TestCase):
    def make(self, now=None, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        cfg.update(overrides)
        return SessionHygiene(cfg, now=now)

    def test_idle_starts_at_zero_on_construction(self):
        h = self.make(now=100.0)
        self.assertAlmostEqual(h.seconds_idle(now=100.0), 0.0, places=3)

    def test_seconds_idle_grows_with_time(self):
        h = self.make()
        h.mark_activity(now=100.0)
        self.assertAlmostEqual(h.seconds_idle(now=130.0), 30.0, places=3)

    def test_mark_activity_resets_the_clock(self):
        h = self.make()
        h.mark_activity(now=100.0)
        h.mark_activity(now=200.0)
        self.assertAlmostEqual(h.seconds_idle(now=205.0), 5.0, places=3)


if __name__ == "__main__":
    unittest.main()
