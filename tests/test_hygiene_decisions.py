import unittest

from backtalk.hygiene import SessionHygiene


class HygieneDecisionsTest(unittest.TestCase):
    def make(self, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        cfg.update(overrides)
        return SessionHygiene(cfg)

    def test_should_clear_false_under_threshold(self):
        h = self.make(idle_clear_minutes=60)
        self.assertFalse(h.should_clear(59 * 60))

    def test_should_clear_true_at_threshold(self):
        h = self.make(idle_clear_minutes=60)
        self.assertTrue(h.should_clear(60 * 60))

    def test_should_compact_false_under_threshold(self):
        h = self.make(compact_context_threshold=0.6)
        self.assertFalse(h.should_compact(0.59))

    def test_should_compact_true_at_threshold(self):
        h = self.make(compact_context_threshold=0.6)
        self.assertTrue(h.should_compact(0.6))

    def test_should_compact_false_when_fraction_unknown(self):
        h = self.make()
        self.assertFalse(h.should_compact(None))

    def test_compaction_cap_not_reached_initially(self):
        h = self.make(max_compactions_per_session=3)
        self.assertFalse(h.compaction_cap_reached())

    def test_compaction_cap_reached_after_max(self):
        h = self.make(max_compactions_per_session=3)
        h._compactions_this_session = 3
        self.assertTrue(h.compaction_cap_reached())


if __name__ == "__main__":
    unittest.main()
