# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The n9router brain — the middle fallback tier, for quota exhaustion
while the internet is still up.

Sits between WarmBrain (cloud Claude via the Agent SDK, tier 1) and
LocalBrain (genuine offline, tier 3): when the 5-hour plan window is
spent but connectivity is fine, turns route HERE instead of dropping
straight to the local model. n9router's own "jarvis-voice-fallback"
combo (dashboard: Combo & Vision Adapter) does the actual model
selection — Kimi K2.5 first, five more behind it, fallback-in-order
strategy — this class only knows n9router's OpenAI-compatible endpoint
and never picks a model itself.

Deliberately NOT scope-limited like LocalBrain: no Agent SDK here, so
still no tools, no skills, no vault access — n9router only proxies
chat completions — but no canned-reply/HA-only restriction either.
This is meant to read as a full stand-in conversation, not a utility
fallback: see backtalk/n9_fallback config and main.py's three-way
routing in handle().
"""
import json
import re
from typing import AsyncIterator, Optional

import httpx

from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

# Bounds the per-call payload so a long fallback stretch can't grow the
# prompt without limit — same reasoning as WarmBrain's own session, just
# capped by hand here since there's no SDK doing it for us. Counts
# messages, not tokens: cheap and good enough for a voice conversation's
# short turns.
_MAX_HISTORY_MESSAGES = 20

SYSTEM_PROMPT = (
    "You are Jarvis, a chief of staff and operating partner, speaking "
    "out loud over a voice line. Cloud Claude's 5-hour usage window is "
    "temporarily spent, so you're standing in through a fallback model "
    "until it resets -- keep the same persona and tone (refined, "
    "precise British delivery, dry wit, address the user as \"sir\"), "
    "but you have no tools, no file access, and no memory beyond this "
    "conversation. If asked to do something that needs hands (files, "
    "code, devices), say plainly that it's queued for when the primary "
    "brain is back -- never pretend to have done it."
)

UNREACHABLE_LINE = (
    "The fallback line's down as well, sir -- I'm cut off from "
    "everything for the moment."
)


class N9Brain:
    """Talks to n9router's `jarvis-voice-fallback` combo over its
    OpenAI-compatible endpoint. One request per turn, not a token
    stream — n9router's fallback-on-failure needs a complete response
    before it can retry the next model in the combo, so there's no
    partial stream to forward. The reply is still split into sentences
    at yield time so the existing sentence-at-a-time speak pipeline
    (built for WarmBrain's real stream) gets it the same shape either
    way."""

    def __init__(self, base_url: str = "http://127.0.0.1:20128",
                 model: str = "jarvis-voice-fallback",
                 timeout_s: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._history: list[dict] = []

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        self._history.append({"role": "user", "content": utterance})
        body = {
            "model": self._model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         *self._history],
            "temperature": 0.4,
            # Explicit, not just "omitted means false": at least one
            # combo member (Gemini) was observed streaming SSE delta
            # chunks back on a request with no "stream" key at all,
            # which the parsing below can't read (no choices[0].message,
            # only choices[0].delta). raw_decode below stays anyway as
            # a second line of defense, not a replacement for this.
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/chat/completions", json=body)
                resp.raise_for_status()
                # NOT resp.json(): when a combo falls through past its
                # first model, n9router has been observed appending a
                # stray SSE "data: [DONE]" terminator after the JSON
                # body of an otherwise ordinary (non-streaming) 200 --
                # trailing garbage a strict decode rejects outright.
                # raw_decode reads the one JSON value at the front and
                # ignores whatever's stuck on after it.
                data, _ = json.JSONDecoder().raw_decode(resp.text.lstrip())
            text = ((data.get("choices") or [{}])[0]
                     .get("message", {}).get("content") or "").strip()
        except Exception as e:
            log(f"[n9_brain] request failed: {e!r}")
            yield UNREACHABLE_LINE
            return
        if not text:
            log("[n9_brain] empty reply from combo")
            yield UNREACHABLE_LINE
            return
        self._history.append({"role": "assistant", "content": text})
        # Cap AFTER a successful exchange only — a failed call above
        # never appended an assistant turn, so trimming here can't
        # strand a dangling user message with no reply behind it.
        if len(self._history) > _MAX_HISTORY_MESSAGES:
            self._history = self._history[-_MAX_HISTORY_MESSAGES:]

        buf = text
        while True:
            m = _SENTENCE_END.search(buf)
            if not m:
                break
            sentence, buf = buf[:m.end()].strip(), buf[m.end():]
            if sentence:
                yield sentence
        tail = buf.strip()
        if tail:
            yield tail


if __name__ == "__main__":
    import asyncio

    async def _run():
        b = N9Brain()

        async def collect(utterance):
            return " ".join([s async for s in b.ask_stream(utterance)])

        out = await collect("Voice check: greet me in one sentence.")
        print(f"  {out}")
        assert out and out != UNREACHABLE_LINE, f"got {out!r}"
        out = await collect("What did I just ask you to do?")
        print(f"  {out}")
        assert out and out != UNREACHABLE_LINE, f"got {out!r}"
        print("n9_brain self-test: OK")

    asyncio.run(_run())
