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
"""The offline brain — Qwen3-4B-Instruct-2507 via a local llama-server,
scoped deliberately to Home Assistant control plus a short list of
canned utility replies. NOT a smaller version of WarmBrain: no vault,
no skills, no general tool ecosystem. Anything outside that scope gets
an honest "I can't do that right now", never a freelanced answer.

Model/benchmark/tool-call-reliability work: local-llm-benchmark/ and
[[Active Priorities]]. Design: backtalk/docs/superpowers/specs/
2026-09-09-offline-fallback-design.md.
"""
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))
import homeassistant  # noqa: E402

from backtalk.vlog import log

_TIME_RE = re.compile(
    r"what(?:'s| is) the (?:current )?time|what time is it", re.I)
_DATE_RE = re.compile(
    r"what(?:'s| is) (?:today's )?date|what day is it", re.I)
_MATH_RE = re.compile(
    r"^\s*(?:what(?:'s| is)\s+)?(-?\d+(?:\.\d+)?)\s*"
    r"(plus|minus|times|divided by|x|\+|-|\*|/)\s*"
    r"(-?\d+(?:\.\d+)?)\s*\??\s*$", re.I)
_MATH_OPS = {"plus": "+", "minus": "-", "times": "*", "x": "*",
             "divided by": "/", "+": "+", "-": "-", "*": "*", "/": "/"}
_TIMER_TRIGGER_RE = re.compile(r"\b(?:set|start)\b.*\btimer\b|\btimer\b", re.I)

# Whisper transcribes small spoken numbers as words ("two minutes"), not
# digits ("2 minutes") - so the duration parser has to understand both.
_ONES_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine", "ten", "eleven", "twelve", "thirteen",
               "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
               "nineteen"]
_TENS_WORDS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy",
               "eighty", "ninety"]
_ONES_MAP = {w: i for i, w in enumerate(_ONES_WORDS)}
_TENS_MAP = {w: (i + 2) * 10 for i, w in enumerate(_TENS_WORDS)}
_NUMBER_WORD_ALT = "|".join(["a", "an"] + _TENS_WORDS + _ONES_WORDS)
_DURATION_PART_RE = re.compile(
    r"(\d+(?:\.\d+)?|(?:" + _NUMBER_WORD_ALT + r")(?:[\s-]+(?:" +
    "|".join(_ONES_WORDS) + r"))?)\s*"
    r"(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b", re.I)
_UNIT_SECONDS = {
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
}


def _word_to_number(phrase: str) -> float:
    """Turns a spelled-out number like "twenty five" into 25.0. Assumes
    every token is already a known number word (only ever called on text
    that matched _NUMBER_WORD_ALT)."""
    total = 0
    for word in re.split(r"[\s-]+", phrase.strip()):
        total += _TENS_MAP.get(word, _ONES_MAP.get(word, 0))
    return float(total)

FALLBACK_LINE = ("I'm offline right now. I can only handle device "
                  "control and a few basics until the connection's back.")
UNREACHABLE_LINE = "I can't reach my local fallback either right now."

# Real entity_ids, not mocked names — same set validated in
# local-llm-benchmark/tool_call_test.py's 10/10 reliability run. Extend
# this list by hand as you add rooms/devices; it is deliberately static
# rather than dynamically built from the full ~980-entity HA instance,
# which would blow the context and slow the model down for no benefit
# given the actual (small) scope of offline control.
SYSTEM_PROMPT = (
    "You are Jarvis, a home automation assistant. You control real "
    "devices in Sir's house via Home Assistant. Known entities: "
    "light.shop_benches (Shop Benches), light.electronics_bench "
    "(Electronics Bench), light.computer_benches (Computer Benches), "
    "switch.ewelink_switch_zr03_1_switch_2 (window fan, also wrapped as "
    "fan.window_fan_switch). Use the call_ha_service tool for any "
    "action (turn on/off, set a value). Use get_entity_state only to "
    "read a status, never to take an action. If the request has "
    "nothing to do with home automation, do not call any tool - just "
    "say you can't help with that right now."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_ha_service",
            "description": "Call a Home Assistant service to control a real device (lights, switches, fans, climate, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "HA domain, e.g. light, switch, fan, climate"},
                    "service": {"type": "string", "description": "HA service, e.g. turn_on, turn_off, set_temperature"},
                    "entity_id": {"type": "string", "description": "the exact entity_id to target"},
                    "data": {"type": "object", "description": "optional extra service params, e.g. {\"brightness_pct\": 50}"},
                },
                "required": ["domain", "service", "entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_state",
            "description": "Read the current state of a Home Assistant entity, no side effects.",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        },
    },
]


def _parse_duration_seconds(text: str) -> float:
    """Sums every "<number> <unit>" span found, so "5 minutes 30 seconds"
    and "1 hour and 15 minutes" both work - as do spelled-out equivalents
    like "two minutes" or "a minute". Returns 0 if none found."""
    total = 0.0
    for amount, unit in _DURATION_PART_RE.findall(text):
        amount = amount.strip().lower()
        if amount in ("a", "an"):
            value = 1.0
        elif amount[0].isdigit():
            value = float(amount)
        else:
            value = _word_to_number(amount)
        total += value * _UNIT_SECONDS[unit.lower()]
    return total


def _humanize_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return " ".join(parts) if parts else "0 seconds"


def _canned_reply(utterance: str) -> str | None:
    """A short, deliberately non-growing list — see the module docstring.
    Returns None (fall through to the model) for anything not on it."""
    if _TIME_RE.search(utterance):
        return f"It's {datetime.now().strftime('%I:%M %p').lstrip('0')}."
    if _DATE_RE.search(utterance):
        return f"It's {datetime.now().strftime('%A, %B %d')}."
    m = _MATH_RE.match(utterance.strip())
    if m:
        a_s, op_word, b_s = m.groups()
        op = _MATH_OPS[op_word.lower()]
        a, b = float(a_s), float(b_s)
        if op == "/" and b == 0:
            return "I can't divide by zero."
        result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
        result = int(result) if result == int(result) else round(result, 2)
        return f"That's {result}."
    return None


class LocalBrain:
    def __init__(self, can_use_tool, base_url: str = "http://127.0.0.1:8712",
                 timeout_s: float = 60.0,
                 speak_fn: Optional[Callable[[str], None]] = None):
        self._can_use_tool = can_use_tool
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        # Lets a timer announce itself once it fires, unprompted, the
        # same way main.py's connectivity-change handler calls
        # mouth.say() directly outside of any turn. None in tests/CLI
        # use (see __main__ below) - a fired timer just logs instead.
        self._speak_fn = speak_fn
        self._timers: set[asyncio.Task] = set()
        # True right after we ask "how long should the timer be?" so the
        # very next utterance is read as the answer even without the
        # word "timer" in it — LocalBrain has no other turn memory.
        self._awaiting_timer_duration = False

    def _start_timer(self, seconds: float, label: str) -> None:
        task = asyncio.create_task(self._run_timer(seconds, label))
        self._timers.add(task)
        task.add_done_callback(self._timers.discard)

    async def _run_timer(self, seconds: float, label: str) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        log(f"[local_brain] timer done: {label}")
        if self._speak_fn is None:
            log("[local_brain] timer fired but no speak_fn wired - "
                "can't announce it")
            return
        try:
            self._speak_fn(f"Sir, your {label} timer is up.")
        except Exception as e:
            log(f"[local_brain] timer announce failed: {e!r}")

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        if self._awaiting_timer_duration:
            self._awaiting_timer_duration = False
            seconds = _parse_duration_seconds(utterance)
            if seconds > 0:
                label = _humanize_duration(seconds)
                self._start_timer(seconds, label)
                yield f"Timer set for {label}."
                return
            # Not a duration answer - treat this utterance normally
            # instead of swallowing it as a failed duration parse.

        canned = _canned_reply(utterance)
        if canned is not None:
            yield canned
            return

        if _TIMER_TRIGGER_RE.search(utterance):
            seconds = _parse_duration_seconds(utterance)
            if seconds <= 0:
                self._awaiting_timer_duration = True
                yield "How long should the timer be?"
                return
            label = _humanize_duration(seconds)
            self._start_timer(seconds, label)
            yield f"Timer set for {label}."
            return

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={
                        "model": "local",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": utterance},
                        ],
                        "tools": TOOLS,
                        "tool_choice": "auto",
                        "temperature": 0,
                    })
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            log(f"[local_brain] llama-server unreachable: {e!r}")
            yield UNREACHABLE_LINE
            return

        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            yield FALLBACK_LINE
            return

        loop = asyncio.get_running_loop()
        for tc in calls:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"])
            except Exception:
                continue
            if fn["name"] == "get_entity_state":
                entity_id = args.get("entity_id", "")
                try:
                    state = await loop.run_in_executor(
                        None, homeassistant.get_state, entity_id)
                    yield f"{entity_id} is {state.get('state', 'unknown')}."
                except Exception as e:
                    log(f"[local_brain] get_state failed: {e!r}")
                    yield "I couldn't check that."
            elif fn["name"] == "call_ha_service":
                domain = args.get("domain", "")
                service = args.get("service", "")
                entity_id = args.get("entity_id", "")
                data_arg = args.get("data") or {}
                result = await self._can_use_tool(
                    "HAServiceCall",
                    {"domain": domain, "service": service,
                     "entity_id": entity_id, "data": data_arg},
                    None)
                if getattr(result, "behavior", "") != "allow":
                    yield "Okay, I won't do that."
                    continue
                try:
                    await loop.run_in_executor(
                        None, homeassistant.call_service, domain, service,
                        entity_id, data_arg)
                    yield "Done."
                except Exception as e:
                    log(f"[local_brain] call_service failed: {e!r}")
                    yield "That didn't work — check the log."


if __name__ == "__main__":
    async def _allow_all(tool, tool_input, ctx):
        class _Allow:
            behavior = "allow"
        return _Allow()

    async def _run():
        brain = LocalBrain(can_use_tool=_allow_all)

        async def collect(utterance):
            return " ".join([s async for s in brain.ask_stream(utterance)])

        out = await collect("what's 12 plus 7")
        assert "19" in out, f"expected 19 in {out!r}"
        out = await collect("what time is it")
        assert ":" in out, f"expected a time in {out!r}"
        out = await collect("turn on the shop lights")
        assert out.strip() == "Done.", f"expected Done., got {out!r}"
        out = await collect("what's the weather like today")
        assert out.strip() == FALLBACK_LINE.strip(), f"expected fallback line, got {out!r}"

        announced = []
        timer_brain = LocalBrain(can_use_tool=_allow_all,
                                  speak_fn=announced.append)
        out = " ".join([s async for s in
                         timer_brain.ask_stream("set a timer for 2 seconds")])
        assert "2 seconds" in out, f"expected '2 seconds' in {out!r}"
        assert not announced, "timer fired before its duration elapsed"
        await asyncio.sleep(2.3)
        assert len(announced) == 1, f"expected one announcement, got {announced!r}"
        assert "2 seconds" in announced[0], f"expected duration in {announced[0]!r}"
        out = " ".join([s async for s in
                         timer_brain.ask_stream("set a timer")])
        assert out.strip() == "How long should the timer be?", f"got {out!r}"

        # Follow-up answer to "how long" shouldn't need the word "timer".
        out = " ".join([s async for s in
                         timer_brain.ask_stream("set a timer")])
        assert out.strip() == "How long should the timer be?", f"got {out!r}"
        out = " ".join([s async for s in
                         timer_brain.ask_stream("5 seconds")])
        assert "5 seconds" in out, f"expected '5 seconds' in {out!r}"

        # A non-duration reply after the question should fall through
        # normally instead of being swallowed as a failed duration parse.
        out = " ".join([s async for s in
                         timer_brain.ask_stream("set a timer")])
        assert out.strip() == "How long should the timer be?", f"got {out!r}"
        out = " ".join([s async for s in
                         timer_brain.ask_stream("what time is it")])
        assert ":" in out, f"expected a time in {out!r}"

        # Spelled-out numbers, straight from tonight's live failure:
        # Whisper transcribes "two minutes" as words, not digits, and
        # the parser used to only understand digits.
        out = " ".join([s async for s in
                         timer_brain.ask_stream("start a timer for two minutes")])
        assert "2 minutes" in out, f"expected '2 minutes' in {out!r}"
        out = " ".join([s async for s in
                         timer_brain.ask_stream("set a timer for twenty five seconds")])
        assert "25 seconds" in out, f"expected '25 seconds' in {out!r}"
        out = " ".join([s async for s in
                         timer_brain.ask_stream("set a timer for a minute")])
        assert "1 minute" in out, f"expected '1 minute' in {out!r}"

        # Same spelled-out numbers on the two-step follow-up path.
        out = " ".join([s async for s in
                         timer_brain.ask_stream("start a timer")])
        assert out.strip() == "How long should the timer be?", f"got {out!r}"
        out = " ".join([s async for s in
                         timer_brain.ask_stream("two minutes")])
        assert "2 minutes" in out, f"expected '2 minutes' in {out!r}"
        print("local_brain self-test: OK")

    asyncio.run(_run())
