import unittest

from backtalk.config import CFG
from backtalk.main import console_match


class ConsoleModelVerbsTest(unittest.TestCase):
    def test_light_phrases_match(self):
        for phrase in ("switch to the light model", "use the light model",
                       "slash model light", "Switch to the light model."):
            self.assertEqual(console_match(phrase), "light", phrase)

    def test_standard_phrases_match(self):
        for phrase in ("switch to the standard model",
                       "use the standard model",
                       "back to the standard model",
                       "slash model standard"):
            self.assertEqual(console_match(phrase), "standard", phrase)

    def test_old_fast_phrases_still_alias_standard(self):
        for phrase in ("switch to the fast model", "use the fast model",
                       "back to the fast model", "slash model fast"):
            self.assertEqual(console_match(phrase), "standard", phrase)

    def test_deep_unchanged(self):
        self.assertEqual(console_match("switch to the deep model"), "deep")

    def test_needs_exact_phrase(self):
        self.assertIsNone(console_match("that model feels light today"))
        self.assertIsNone(console_match("the standard model is fine"))

    def test_three_distinct_model_ids(self):
        ids = {CFG["model"], CFG["deep_model"], CFG["light_model"]}
        self.assertEqual(len(ids), 3)
        self.assertIn("haiku", CFG["light_model"])


if __name__ == "__main__":
    unittest.main()
