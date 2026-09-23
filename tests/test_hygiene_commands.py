"""run_clear/run_compact/run_full_summary_and_clear must: call
reset_turn() first, send a checkpoint turn, then the actual slash
command, and report failure (without raising) if either call comes
back with brain.command()'s own "error:" sentinel (brain.py:307)."""
import unittest


class FakeBrain:
    def __init__(self, responses=None):
        self.reset_turn_calls = 0
        self.commands = []
        # responses: optional list of canned return values, one per
        # command() call, in order. Defaults to a generic ack for all.
        self._responses = list(responses or [])

    async def reset_turn(self):
        self.reset_turn_calls += 1

    async def command(self, cmd: str) -> str:
        self.commands.append(cmd)
        if self._responses:
            return self._responses.pop(0)
        return "ok"


from backtalk.hygiene import SessionHygiene


class HygieneCommandsTest(unittest.IsolatedAsyncioTestCase):
    def make(self):
        cfg = {"idle_clear_minutes": 60, "compact_context_threshold": 0.6,
               "max_compactions_per_session": 3, "check_interval_s": 60}
        return SessionHygiene(cfg)

    async def test_run_clear_success(self):
        h = self.make()
        brain = FakeBrain()
        ok = await h.run_clear(brain)
        self.assertTrue(ok)
        self.assertEqual(brain.reset_turn_calls, 1)
        self.assertEqual(len(brain.commands), 2)
        self.assertIn("checkpoint", brain.commands[0].lower())
        self.assertEqual(brain.commands[1], "/clear")

    async def test_run_compact_success(self):
        h = self.make()
        brain = FakeBrain()
        ok = await h.run_compact(brain)
        self.assertTrue(ok)
        self.assertEqual(brain.commands[1], "/compact")

    async def test_run_clear_fails_on_checkpoint_error(self):
        h = self.make()
        brain = FakeBrain(responses=["error: the command timed out"])
        ok = await h.run_clear(brain)
        self.assertFalse(ok)
        # Must not have gone on to send /clear after a failed checkpoint.
        self.assertEqual(len(brain.commands), 1)

    async def test_run_compact_fails_on_command_error(self):
        h = self.make()
        brain = FakeBrain(responses=["ok", "error: the command timed out"])
        ok = await h.run_compact(brain)
        self.assertFalse(ok)
        self.assertEqual(len(brain.commands), 2)

    async def test_run_full_summary_and_clear_success(self):
        h = self.make()
        brain = FakeBrain()
        ok = await h.run_full_summary_and_clear(brain)
        self.assertTrue(ok)
        self.assertEqual(brain.commands[1], "/clear")
        self.assertIn("summary", brain.commands[0].lower())


if __name__ == "__main__":
    unittest.main()
