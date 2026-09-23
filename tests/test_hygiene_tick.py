"""tick() owns all the gating: never fire mid-turn, never fire
offline, never a 4th compact once the cap is hit, and a failed
run_* must not increment the compaction counter.

Also covers the final-review fix pass (2026-09-23):
- a real user turn must be able to preempt an in-flight cycle
  (Critical #2 -- the SDK has one shared message stream)
- never fire in "ask" permission mode (Important #5)
- never idle-clear an already-empty session (Important #3)
- back off after a failure instead of retrying every tick (Important #4)
- never fire while the plan's quota is exhausted (Important #4)
- reset the compaction counter after any successful clear (Important #6)
"""
import asyncio
import unittest
from unittest.mock import AsyncMock

from backtalk.hygiene import SessionHygiene


class FakeLock:
    def __init__(self, active=False):
        self._active = active

    def is_active(self):
        return self._active


class HygieneTickTest(unittest.IsolatedAsyncioTestCase):
    def make(self, now=None, **overrides):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        cfg.update(overrides)
        return SessionHygiene(cfg, now=now)

    async def test_skips_when_turn_active(self):
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=True),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_clear.assert_not_called()

    async def test_skips_when_offline(self):
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: False,
                     is_autoapprove_fn=lambda: True)
        h.run_clear.assert_not_called()

    async def test_skips_when_not_autoapprove(self):
        """Important #5: a checkpoint in 'ask' mode would speak a
        permission prompt into an empty room, time out, and still
        /clear with nothing saved. Must not run at all outside
        bypassPermissions."""
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: False)
        h.run_clear.assert_not_called()

    async def test_skips_idle_clear_with_no_activity_since_last_clear(self):
        """Important #3: a session that's had nothing happen since its
        last clear has nothing to clear -- must not run a checkpoint +
        /clear every idle_clear_minutes forever on an empty session."""
        h = self.make(idle_clear_minutes=0)
        # No mark_activity() call -- fresh instance, nothing has
        # happened since construction (which counts as "last clear").
        h.run_clear = AsyncMock(return_value=True)
        h.get_context_fraction = AsyncMock(return_value=0.1)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_clear.assert_not_called()

    async def test_idle_triggers_clear_when_activity_happened(self):
        # Construct at a known, arbitrarily-low baseline (0.0) rather
        # than comparing two real time.monotonic() reads -- those can
        # land on the same tick and falsely "pass" a >= comparison
        # whether or not _reset_after_clear() actually ran (this is
        # exactly the review's finding #7 against the old version of
        # this test). Real monotonic() values are always far above 0.
        h = self.make(now=0.0, idle_clear_minutes=0)
        h.mark_activity(now=0.0)
        h.run_clear = AsyncMock(return_value=True)
        h.get_context_fraction = AsyncMock(return_value=0.1)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_clear.assert_awaited_once()
        self.assertGreater(h._last_activity, 0.0)

    async def test_successful_idle_clear_resets_activity_flag(self):
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        self.assertFalse(h._has_activity_since_clear)

    async def test_context_triggers_compact_and_increments_counter(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5)
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_compact = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_compact.assert_awaited_once()
        self.assertEqual(h._compactions_this_session, 1)

    async def test_failed_compact_does_not_increment_counter(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5)
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_compact = AsyncMock(return_value=False)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
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
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_compact.assert_not_called()
        h.run_full_summary_and_clear.assert_awaited_once()

    async def test_cap_reset_after_successful_full_summary_clear(self):
        """Important #6: /clear starts a new session, so the cap must
        not stay latched forever -- the next 60% event after a full
        summary+clear should compact normally again, not immediately
        take the heavy path a second time."""
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.5,
                      max_compactions_per_session=3)
        h._compactions_this_session = 3
        h.get_context_fraction = AsyncMock(return_value=0.6)
        h.run_full_summary_and_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        self.assertEqual(h._compactions_this_session, 0)

    async def test_neither_threshold_met_does_nothing(self):
        h = self.make(idle_clear_minutes=999999,
                      compact_context_threshold=0.99)
        h.get_context_fraction = AsyncMock(return_value=0.1)
        h.run_clear = AsyncMock(return_value=True)
        h.run_compact = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_clear.assert_not_called()
        h.run_compact.assert_not_called()

    async def test_a_tick_exception_is_swallowed_and_logged(self):
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(side_effect=RuntimeError("boom"))
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)

    async def test_skips_when_quota_exhausted(self):
        """Important #4: don't keep trying (and burning 90s each time)
        against a cloud tier that's already spent for the window."""
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=True)

        class QuotaExhaustedBrain:
            def quota_exhausted(self):
                return True

        await h.tick(brain=QuotaExhaustedBrain(),
                     turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_clear.assert_not_called()

    async def test_bare_object_brain_without_quota_exhausted_still_runs(self):
        """Test doubles (and the gating tests above) pass brain=object()
        with no quota_exhausted method at all -- must not crash, must
        be treated as 'not exhausted'."""
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=True)
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_clear.assert_awaited_once()

    async def test_failure_sets_a_backoff(self):
        """Important #4: a failed cycle must not retry every single
        tick -- each retry can be a real ~90s billed model turn."""
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=False)
        import time
        before = time.monotonic()
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        self.assertGreater(h._backoff_until, before)

    async def test_backoff_skips_the_next_tick(self):
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=True)
        import time
        h._backoff_until = time.monotonic() + 999
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        h.run_clear.assert_not_called()

    async def test_success_clears_the_backoff(self):
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()
        h.run_clear = AsyncMock(return_value=True)
        h._backoff_until = 0.0
        h._consecutive_failures = 2
        await h.tick(brain=object(), turn_lock=FakeLock(active=False),
                     is_online_fn=lambda: True,
                     is_autoapprove_fn=lambda: True)
        self.assertEqual(h._backoff_until, 0.0)
        self.assertEqual(h._consecutive_failures, 0)

    async def test_preempt_cancels_an_in_flight_cycle(self):
        """Critical #2: a real user turn must be able to interrupt a
        checkpoint/command cycle before it shares the SDK's single
        message stream with that turn's own reset_turn()/query()."""
        h = self.make(idle_clear_minutes=0)
        h.mark_activity()

        started = asyncio.Event()
        finish_gate = asyncio.Event()

        async def slow_run_clear(brain):
            started.set()
            await finish_gate.wait()   # would hang forever if not cancelled
            return True

        h.run_clear = slow_run_clear
        tick_task = asyncio.create_task(
            h.tick(brain=object(), turn_lock=FakeLock(active=False),
                   is_online_fn=lambda: True,
                   is_autoapprove_fn=lambda: True))
        await started.wait()
        await h.preempt()
        # tick() must have unwound cleanly -- no hang, no exception
        # escaping to the caller.
        await asyncio.wait_for(tick_task, timeout=2)

    async def test_preempt_is_a_no_op_with_no_cycle_running(self):
        h = self.make()
        await h.preempt()   # must not raise


if __name__ == "__main__":
    unittest.main()
