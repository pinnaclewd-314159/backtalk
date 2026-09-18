import time
import unittest

from pynput import keyboard

from backtalk.ptt import PTTListener


class PTTExtraKeysTest(unittest.TestCase):
    def make(self, **kwargs):
        ptt = PTTListener("`", **kwargs)
        self.addCleanup(ptt._listener.stop)
        return ptt

    def test_extra_key_press_holds(self):
        ptt = self.make(extra_keys=["f13"])
        ptt._on_press(keyboard.Key.f13)
        self.assertTrue(ptt.is_held())

    def test_extra_key_release_settles(self):
        ptt = self.make(extra_keys=["f13"])
        ptt._on_press(keyboard.Key.f13)
        ptt._on_release(keyboard.Key.f13)
        time.sleep(PTTListener.RELEASE_GRACE + 0.05)
        self.assertFalse(ptt.is_held())

    def test_extra_key_repeat_press_cancels_release(self):
        ptt = self.make(extra_keys=["f13"])
        ptt._on_press(keyboard.Key.f13)
        ptt._on_release(keyboard.Key.f13)
        ptt._on_press(keyboard.Key.f13)
        time.sleep(PTTListener.RELEASE_GRACE + 0.05)
        self.assertTrue(ptt.is_held())

    def test_primary_key_still_works(self):
        ptt = self.make(extra_keys=["f13"])
        ptt._on_press(keyboard.KeyCode.from_char("`"))
        self.assertTrue(ptt.is_held())

    def test_unlisted_key_ignored(self):
        ptt = self.make(extra_keys=["f13"])
        ptt._on_press(keyboard.Key.f14)
        self.assertFalse(ptt.is_held())

    def test_default_has_no_extra_keys(self):
        ptt = self.make()
        ptt._on_press(keyboard.Key.f13)
        self.assertFalse(ptt.is_held())


if __name__ == "__main__":
    unittest.main()
