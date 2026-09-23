"""watch() is a thin loop: sleep check_interval_s, call tick(), repeat,
until cancelled. Uses a real short interval and real (tiny) sleeps --
matches this repo's existing testing style (see test_ptt_max_hold.py)
rather than a fake clock."""
import asyncio
import unittest
from unittest.mock import AsyncMock

from backtalk.hygiene import SessionHygiene


class HygieneWatchTest(unittest.IsolatedAsyncioTestCase):
    def make(self, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 0.05}
        cfg.update(overrides)
        return SessionHygiene(cfg)

    async def test_calls_tick_on_each_interval(self):
        h = self.make()
        h.tick = AsyncMock(return_value=None)
        task = asyncio.create_task(
            h.watch(brain=object(), turn_lock=object(),
                    is_online_fn=lambda: True,
                    is_autoapprove_fn=lambda: True))
        await asyncio.sleep(0.17)   # ~3 intervals
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self.assertGreaterEqual(h.tick.await_count, 2)

    async def test_cancellation_stops_the_loop_cleanly(self):
        h = self.make()
        h.tick = AsyncMock(return_value=None)
        task = asyncio.create_task(
            h.watch(brain=object(), turn_lock=object(),
                    is_online_fn=lambda: True,
                    is_autoapprove_fn=lambda: True))
        await asyncio.sleep(0.06)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    unittest.main()
