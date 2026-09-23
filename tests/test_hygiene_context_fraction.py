"""Must match _spoken_usage's own category rules (main.py:426-439):
'free' and 'buffer' categories are excluded from the occupied total.
Total = occupied + the 'Free space' category's own tokens. Anything
malformed must return None, never raise -- a bad context-usage payload
must never crash the watcher loop."""
import unittest

from backtalk.hygiene import context_occupied_fraction


class ContextFractionTest(unittest.TestCase):
    def test_typical_breakdown(self):
        ctx = {"categories": [
            {"name": "System prompt", "tokens": 3000},
            {"name": "Messages", "tokens": 27000},
            {"name": "Free space", "tokens": 60000},
            {"name": "Autocompact buffer", "tokens": 10000},
        ]}
        # occupied = 3000 + 27000 = 30000; total = 30000 + 60000 = 90000
        frac = context_occupied_fraction(ctx)
        self.assertAlmostEqual(frac, 30000 / 90000, places=4)

    def test_object_with_categories_attribute(self):
        class Cat:
            def __init__(self, name, tokens):
                self.name, self.tokens = name, tokens

        class Ctx:
            categories = None

        # Object form uses dicts nested under an attribute, matching
        # the real SDK shape main.py already handles via getattr().
        ctx = Ctx()
        ctx.categories = [{"name": "Messages", "tokens": 40},
                          {"name": "Free space", "tokens": 60}]
        frac = context_occupied_fraction(ctx)
        self.assertAlmostEqual(frac, 40 / 100, places=4)

    def test_none_input_returns_none(self):
        self.assertIsNone(context_occupied_fraction(None))

    def test_missing_categories_returns_none(self):
        self.assertIsNone(context_occupied_fraction({}))

    def test_empty_categories_returns_none(self):
        self.assertIsNone(context_occupied_fraction({"categories": []}))

    def test_no_free_space_category_returns_none(self):
        # Can't compute a fraction without knowing the total.
        ctx = {"categories": [{"name": "Messages", "tokens": 100}]}
        self.assertIsNone(context_occupied_fraction(ctx))

    def test_non_dict_category_entries_are_skipped(self):
        ctx = {"categories": ["garbage", None,
                              {"name": "Messages", "tokens": 10},
                              {"name": "Free space", "tokens": 90}]}
        frac = context_occupied_fraction(ctx)
        self.assertAlmostEqual(frac, 10 / 100, places=4)


if __name__ == "__main__":
    unittest.main()
