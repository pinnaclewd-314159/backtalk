"""A hold must be bounded, because a release is not guaranteed to arrive.

Measured on hardware 2026-09-17 with the Bluetooth PTT button: turning the
PC's Bluetooth off mid-hold killed the link before the board could send its
key-up report, and Windows never synthesised one. The probe recorded
raw_up=0 for the remaining 90 seconds of the run, so the listener stayed
held indefinitely and would have kept the microphone open forever.

The board cannot fix this: it has no link left to send a release over. The
guard therefore lives here.
"""
import time
import unittest

from pynput import keyboard

from backtalk.ptt import PTTListener


class PTTMaxHoldTest(unittest.TestCase):
    def make(self, **kwargs):
        ptt = PTTListener("`", **kwargs)
        self.addCleanup(ptt._listener.stop)
        return ptt

    def test_hold_past_max_is_released(self):
        """The stuck-key case: a press with no release ever arriving."""
        ptt = self.make(max_hold=0.15)
        ptt._on_press(keyboard.KeyCode.from_char("`"))
        self.assertTrue(ptt.is_held())
        time.sleep(0.2)
        self.assertFalse(ptt.is_held(),
                         "a hold past max_hold must be treated as a stuck key")

    def test_hold_under_max_is_untouched(self):
        """A normal hold must not be cut short."""
        ptt = self.make(max_hold=5.0)
        ptt._on_press(keyboard.KeyCode.from_char("`"))
        time.sleep(0.2)
        self.assertTrue(ptt.is_held())

    def test_repeat_presses_do_not_extend_the_deadline(self):
        """Auto-repeat must not keep a stuck key alive forever.

        A stuck BLE key still produced 17 repeat events, so the deadline has
        to run from the ORIGINAL press, not from the most recent one.
        """
        ptt = self.make(max_hold=0.15)
        ptt._on_press(keyboard.KeyCode.from_char("`"))
        for _ in range(5):
            time.sleep(0.04)
            ptt._on_press(keyboard.KeyCode.from_char("`"))   # auto-repeat
        self.assertFalse(ptt.is_held(),
                         "repeat presses must not extend the stuck-key deadline")

    def test_new_press_after_timeout_holds_again(self):
        """Recovery: once released, a fresh press must work normally."""
        ptt = self.make(max_hold=0.15)
        ptt._on_press(keyboard.KeyCode.from_char("`"))
        time.sleep(0.2)
        self.assertFalse(ptt.is_held())
        ptt._on_press(keyboard.KeyCode.from_char("`"))
        self.assertTrue(ptt.is_held(),
                        "a press after a stuck-key release must hold again")

    def test_normal_release_still_settles(self):
        """The existing release path must be unaffected."""
        ptt = self.make(max_hold=5.0)
        ptt._on_press(keyboard.KeyCode.from_char("`"))
        ptt._on_release(keyboard.KeyCode.from_char("`"))
        time.sleep(PTTListener.RELEASE_GRACE + 0.05)
        self.assertFalse(ptt.is_held())

    def test_default_max_hold_is_generous(self):
        """The default must never cut off a real person mid-sentence."""
        self.assertGreaterEqual(PTTListener.MAX_HOLD, 60.0)

    def test_default_is_used_when_not_given(self):
        ptt = self.make()
        self.assertEqual(ptt._max_hold, PTTListener.MAX_HOLD)


if __name__ == "__main__":
    unittest.main()
