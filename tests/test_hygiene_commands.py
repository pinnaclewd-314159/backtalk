"""run_clear/run_compact/run_full_summary_and_clear must: call
reset_turn() first, send a checkpoint turn, then the actual slash
command. The checkpoint half is judged by its text (empty or
brain.command()'s own "error:" sentinel, brain.py:307, both mean
nothing was actually written). The slash-command half is judged by
brain.last_command_is_error (the SDK ResultMessage's own is_error
flag) instead, because /clear and /compact routinely -- and
correctly -- return no text on success at all."""
import unittest
from unittest import mock


class FakeBrain:
    def __init__(self, responses=None, command_errors=None):
        self.reset_turn_calls = 0
        self.commands = []
        # responses: optional list of canned return values, one per
        # command() call, in order. Defaults to a generic ack for all.
        self._responses = list(responses or [])
        # command_errors: optional list of bools, one per command() call
        # in order, mirroring the real Brain's last_command_is_error.
        # When not given, inferred per-call from the response text
        # (starts with "error:") to match real Brain's timeout path
        # without every existing test needing to set it explicitly.
        self._command_errors = (list(command_errors)
                                 if command_errors is not None else None)
        self.last_command_is_error = None

    async def reset_turn(self):
        self.reset_turn_calls += 1

    async def command(self, cmd: str) -> str:
        self.commands.append(cmd)
        resp = self._responses.pop(0) if self._responses else "ok"
        if self._command_errors:
            self.last_command_is_error = self._command_errors.pop(0)
        else:
            self.last_command_is_error = resp.startswith("error:")
        return resp


from backtalk.hygiene import SessionHygiene


class HygieneCommandsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Keep the scenarios these tests provoke out of the live backtalk.log.
        p = mock.patch("backtalk.hygiene.log")
        p.start()
        self.addCleanup(p.stop)

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

    async def test_run_clear_fails_on_empty_checkpoint_reply(self):
        """Review finding #9: an empty/interrupted checkpoint reply
        must not be treated as success -- an empty string has no
        'error:' prefix but proves nothing was actually written."""
        h = self.make()
        brain = FakeBrain(responses=[""])
        ok = await h.run_clear(brain)
        self.assertFalse(ok)
        self.assertEqual(len(brain.commands), 1)

    async def test_run_compact_succeeds_on_empty_command_reply(self):
        """The bug this guards against: /clear and /compact routinely
        return no text at all on a genuinely successful run (confirmed
        against main.py's spoken "clear" verb, which never checks resp
        before saying "Cleared. Fresh slate."). An empty reply with no
        SDK-reported error must count as success, not failure -- treating
        it as failure meant _reset_after_clear() never ran, idle time
        never reset, and the retry backoff settled at its 30-minute cap
        forever (root cause of the 2026-09-24 to 2026-09-28 auto-/clear
        loop)."""
        h = self.make()
        brain = FakeBrain(responses=["ok", ""])
        ok = await h.run_compact(brain)
        self.assertTrue(ok)

    async def test_run_compact_fails_on_command_is_error(self):
        """A genuinely failed slash command (SDK ResultMessage.is_error
        True) must still be reported as a failure even with empty text
        -- this is what actually distinguishes a real failure from
        /clear's/compact's normal silent success."""
        h = self.make()
        brain = FakeBrain(responses=["ok", ""],
                           command_errors=[False, True])
        ok = await h.run_compact(brain)
        self.assertFalse(ok)

    async def test_run_clear_fails_on_command_is_error(self):
        h = self.make()
        brain = FakeBrain(responses=["ok", ""],
                           command_errors=[False, True])
        ok = await h.run_clear(brain)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
