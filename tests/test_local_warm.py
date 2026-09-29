"""LocalBrain.warm / keep_warm must prime llama-server with exactly the
prefix real turns send, never raise, and be switchable off by config."""
import asyncio
import unittest
from unittest import mock

import httpx

from backtalk import local_brain
from backtalk.config import DEFAULTS
from backtalk.local_brain import LocalBrain, _chat_body


class _FakeClient:
    """Stands in for httpx.AsyncClient; records posted JSON bodies."""
    posts: list = []
    fail = False

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        if _FakeClient.fail:
            raise httpx.ConnectError("down")
        _FakeClient.posts.append((url, json))
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))


class WarmTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _FakeClient.posts = []
        _FakeClient.fail = False
        p = mock.patch.object(local_brain.httpx, "AsyncClient", _FakeClient)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(local_brain, "log", lambda *_a, **_k: None)
        q.start()
        self.addCleanup(q.stop)
        self.brain = LocalBrain(can_use_tool=None)

    def test_chat_body_prefix_matches_real_turn(self):
        real = _chat_body("turn on the shop lights")
        warm = _chat_body("Hello.", max_tokens=1)
        self.assertEqual(real["messages"][0], warm["messages"][0])
        self.assertEqual(real["tools"], warm["tools"])
        self.assertEqual(real["tool_choice"], warm["tool_choice"])
        self.assertNotIn("max_tokens", real)
        self.assertEqual(warm["max_tokens"], 1)

    async def test_warm_posts_one_token_request(self):
        self.assertTrue(await self.brain.warm())
        (url, body), = _FakeClient.posts
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(body["max_tokens"], 1)

    async def test_warm_returns_false_and_does_not_raise_when_down(self):
        _FakeClient.fail = True
        self.assertFalse(await self.brain.warm())

    async def test_keep_warm_disabled_at_zero(self):
        await asyncio.wait_for(self.brain.keep_warm(0), timeout=1)
        self.assertEqual(_FakeClient.posts, [])

    async def test_keep_warm_repeats_and_survives_outage(self):
        task = asyncio.create_task(self.brain.keep_warm(0.01))
        await asyncio.sleep(0.1)
        _FakeClient.fail = True
        await asyncio.sleep(0.05)
        _FakeClient.fail = False
        n = len(_FakeClient.posts)
        await asyncio.sleep(0.1)
        task.cancel()
        self.assertGreater(n, 1)
        self.assertGreater(len(_FakeClient.posts), n)  # resumed after outage

    def test_config_default(self):
        self.assertEqual(DEFAULTS["local_fallback"]["warm_interval_s"], 300)


if __name__ == "__main__":
    unittest.main()
