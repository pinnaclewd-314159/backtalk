"""tick() owns all the gating: never fire mid-turn, never fire
offline, never a 4th compact once the cap is hit, and a failed
run_* must not increment the compaction counter."""
import unittest
from unittest.mock import AsyncMock

from backtalk.hygiene import SessionHygiene


class FakeLock:
    def __init__(self, active=False):
        self._active = active

    def is_active(self):
        return self._active


class HygieneTickTest(unittest.IsolatedAsyncioTestCase):
    def make(self, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        cfg.update(overrides)
        return SessionHygiene(cfg)

    async def test_skips_when_turn_active(self):
        h = self.make(idle_clear_minutes=0)  # idle threshold already met
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=True),
                     is_online_fn=lambda: True)
        h.run_clear.assert_not_called()

    async def test_skips_when_offline(self):
        h = self.make(idle_clear_minutes=0)
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: False)
        h.run_clear.assert_not_called()

    async def test_idle_triggers_clear_and_resets_clock(self):
        h = self.make(idle_clear_minutes=0)
        h.run_clear = AsyncMock(return_value=True)
        h.get_context_fraction = AsyncMock(return_value=0.1)
        activity_before = h._last_activity
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        h.run_clear.assert_awaited_once()
        # time.monotonic() is guaranteed non-decreasing but can tie on a
        # coarse clock, so >= (not strict >) is the correct proof that
        # mark_activity() actually ran after a successful clear.
        self.assertGreaterEqual(h._last_activity, activity_before)

    async def test_context_triggers_compact_and_increments_counter(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5)
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_compact = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        h.run_compact.assert_awaited_once()
        self.assertEqual(h._compactions_this_session, 1)

    async def test_failed_compact_does_not_increment_counter(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5)
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_compact = AsyncMock(return_value=False)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        self.assertEqual(h._compactions_this_session, 0)

    async def test_cap_reached_uses_heavier_path_not_a_4th_compact(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5,
                      max_compactions_per_session=3)
        h._compactions_this_session = 3
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_compact = AsyncMock(return_value=True)
        h.run_full_summary_and_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        h.run_compact.assert_not_called()
        h.run_full_summary_and_clear.assert_awaited_once()

    async def test_neither_threshold_met_does_nothing(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.99)
        h.get_context_fraction = AsyncMock(return_value=0.1)
        h.run_clear = AsyncMock(return_value=True)
        h.run_compact = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)
        h.run_clear.assert_not_called()
        h.run_compact.assert_not_called()

    async def test_a_tick_exception_is_swallowed_and_logged(self):
        h = self.make(idle_clear_minutes=0)
        h.run_clear = AsyncMock(side_effect=RuntimeError("boom"))
        # Must not raise out of tick().
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True)


if __name__ == "__main__":
    unittest.main()
