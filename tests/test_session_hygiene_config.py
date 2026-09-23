"""session_hygiene must default to off, with sane thresholds, so
turning it on is a deliberate config change rather than a silent
behavior flip on upgrade."""
import unittest

from backtalk.config import DEFAULTS


class SessionHygieneConfigTest(unittest.TestCase):
    def test_block_exists_with_expected_keys(self):
        block = DEFAULTS["session_hygiene"]
        self.assertEqual(
            set(block.keys()),
            {"enabled", "idle_clear_minutes", "compact_context_threshold",
             "max_compactions_per_session", "check_interval_s"})

    def test_off_by_default(self):
        self.assertFalse(DEFAULTS["session_hygiene"]["enabled"])

    def test_default_thresholds_match_spec(self):
        block = DEFAULTS["session_hygiene"]
        self.assertEqual(block["idle_clear_minutes"], 60)
        self.assertEqual(block["compact_context_threshold"], 0.60)
        self.assertEqual(block["max_compactions_per_session"], 3)
        self.assertEqual(block["check_interval_s"], 60)


if __name__ == "__main__":
    unittest.main()
