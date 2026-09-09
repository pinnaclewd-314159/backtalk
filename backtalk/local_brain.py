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
from typing import AsyncIterator

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
                 timeout_s: float = 60.0):
        self._can_use_tool = can_use_tool
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        canned = _canned_reply(utterance)
        if canned is not None:
            yield canned
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
        print("local_brain self-test: OK")

    asyncio.run(_run())
