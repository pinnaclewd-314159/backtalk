# Backtalk session hygiene: automatic idle-clear and context-compact

**Status:** drafted 2026-09-23, design only. Not yet implemented — next step is the writing-plans skill to turn this into an implementation plan.

## Problem

Backtalk's voice line (`WarmBrain` in `brain.py`) is a persistent Claude Agent SDK session that can sit connected for days. CLAUDE.md's Session hygiene rule — checkpoint before compacting, compact around 60% context capped at three times per session, clear after an hour idle — was written as something Jarvis "does," but it can't: `/clear` and `/compact` have no self-trigger mechanism from inside a Claude Code session (confirmed against official docs, no tool/hook/SDK path exists). For a typed chat session that's fine — Sir is present and I can just flag it. Voice is the opposite: it sits idle for long stretches with nobody there to read a flag, so unmanaged context growth silently costs tokens and (eventually) latency. This spec covers letting backtalk itself watch the clock and the context window and drive the already-existing `brain.command()` channel — no keystrokes, no window automation, no new SDK capability needed.

## Scope

- **In scope:** a background watcher inside backtalk that tracks idle time and context usage, checkpoints the vault via a normal (silent) turn, then calls `brain.command("/clear")` or `brain.command("/compact")` exactly the way the existing spoken "clear"/"compact" verbs already do; a per-session compaction cap; safe interaction with an in-progress turn, offline mode, and the existing turn lock.
- **Out of scope:** the chat-session side (already solved — CLAUDE.md now has me flag it, Sir runs it manually); any change to the manual spoken "clear"/"compact"/"deep"/"fast" verbs, which stay exactly as they are; changing what a checkpoint actually contains (that's ordinary vault-discipline behavior already defined in CLAUDE.md, not new logic here).

## Design

### Component 1 — `session_hygiene` config block (`config.py` DEFAULTS + `backtalk.json`)

Follows the existing `local_fallback`/`n9_fallback` block pattern — off by default, since this changes always-on production behavior and turning it on is Sir's call:

```python
"session_hygiene": {
    "enabled": False,
    "idle_clear_minutes": 60,
    "compact_context_threshold": 0.60,   # fraction of context occupied
    "max_compactions_per_session": 3,
    "check_interval_s": 60,
},
```

### Component 2 — `backtalk/hygiene.py` (new): `SessionHygiene`

A small class, one background asyncio loop, mirroring `ConnectivityMonitor`'s shape (`connectivity.py`):

- `mark_activity()` — called from a single shared point (see Component 3) to reset the idle clock. Just `self._last_activity = time.monotonic()`.
- `async def watch(self, brain, turn_lock, is_online_fn)` — the loop: sleep `check_interval_s`, then each tick:
  1. Skip entirely if `not is_online_fn()` (offline mode can't checkpoint or compact — no cloud call available) or `turn_lock.current_owner() is not None` (never interrupt a live turn — re-checked fresh each tick, not cached).
  2. **Idle check:** if `time.monotonic() - self._last_activity >= idle_clear_minutes * 60`, run the clear sequence (Component 4), then reset the idle clock so it doesn't fire again next tick.
  3. **Context check:** otherwise, call `await brain.context_usage()`, compute occupied fraction from `ctx_usage.categories` the same way `_spoken_usage` already does (`main.py:426-439` — sum non-"free"/non-"buffer" category tokens as occupied, add the "Free space" category's tokens for total, divide). If occupied/total ≥ `compact_context_threshold`:
     - If `self._compactions_this_session < max_compactions_per_session`: run the compact sequence (Component 4), increment the counter.
     - Else: run the **heavier** path — full session summary (not just a checkpoint) to today's daily note and the relevant vault notes, then clear instead of a fourth compact, matching CLAUDE.md's existing "after the third, write a full summary, then clear" rule.
- All exceptions inside a tick are caught and logged (`log(f"[hygiene] tick failed: {e!r}")`) — a bad tick must never kill the watcher loop or take the voice line down with it.

### Component 3 — activity tracking hook

One line added at the top of `handle()` in `main.py`: `hygiene.mark_activity()`. `handle()` is the single shared entrypoint for every utterance regardless of source (local PTT, satellite, web PTT — confirmed by reading it in full), so this correctly resets the idle clock no matter which room triggered it. No per-source wiring needed.

### Component 4 — checkpoint-then-command sequence

Both the clear and compact paths do the same two-step thing, reusing existing primitives with zero new methods on `WarmBrain`:

1. **Checkpoint turn** — `await brain.command("Per your memory discipline in CLAUDE.md, checkpoint current session state to the vault now — today's daily note and any note whose contextual home this session touched. This is an automatic session-hygiene checkpoint, not a request from Sir.")`. `brain.command()` already sends arbitrary text through `self._client.query()` and drains the reply silently (`brain.py:277-309`) — it isn't slash-command-specific, so no change to `brain.py` is needed at all.
2. **The actual command** — `await brain.command("/clear")` or `await brain.command("/compact")`, exactly what the spoken verbs already call (`main.py:1183`, `1187`).

Both steps run through `brain.reset_turn()` first (same as `_run_console_inner` does), for the same reason: keep the shared message pipe aligned.

### Component 5 — silent by default

The watcher never calls `mouth.say()`. Sir is very likely not in the room when this fires — that's the whole point. Everything goes to the log only: `[hygiene] idle 62min, no activity — checkpointing and clearing`, `[hygiene] context at 61% (2/3 compactions this session) — checkpointing and compacting`, `[hygiene] compaction cap hit — full summary and clear instead`. This is a deliberate difference from the manual verbs (which do speak, because a human just asked out loud).

### Component 6 — wiring into `amain()`

```python
hygiene = SessionHygiene(CFG.get("session_hygiene", {}))
if CFG.get("session_hygiene", {}).get("enabled"):
    hygiene_task = asyncio.create_task(
        hygiene.watch(brain, turn_lock, lambda: connectivity.is_online))
```
Cancelled in the same `finally:` block that already stops `brain`/`connectivity` (`main.py:1644-1654`).

## Safety / edge cases

- **Never mid-turn.** `turn_lock.current_owner() is not None` is checked fresh at the top of every tick, not cached — a turn that starts between ticks is caught on the next check before anything fires.
- **Never offline.** Both the checkpoint turn and the command need a live cloud connection; the watcher just skips its tick while `connectivity.is_online` is false and resumes checking once it flips back.
- **Failed checkpoint or command.** `brain.command()` already returns `"error: the command timed out"` on failure (`brain.py:307`) rather than raising. On that sentinel, log it, skip incrementing the compaction counter (a failed compact shouldn't count against the cap), and try again next tick — never crash the loop.
- **Off by default.** `session_hygiene.enabled` defaults to `False`. Turning it on in `backtalk.json` is a live-config change and needs Sir's confirmation, same as any other config change affecting a running system.
- **Compaction counter is in-memory, per-process.** It resets naturally on every backtalk restart, which is the correct behavior — "per session" here means "per running voice-line process," matching how the manual verbs and `brain.session` stats already work.

## Testing

- Set `idle_clear_minutes: 1` and `check_interval_s: 15` locally; confirm an idle voice line auto-clears within ~75s, logs the checkpoint and clear, and does **not** fire while a conversation is actively going (start a turn right as the timer would trip, confirm it's skipped and catches on the next tick).
- Set `compact_context_threshold: 0.05` locally to force early compaction; confirm it checkpoints, compacts, increments the counter, and that a fourth trigger takes the heavier summary+clear path instead of a fourth compact.
- Kill connectivity mid-idle-window; confirm the watcher skips cleanly and picks back up once `connectivity.is_online` flips true.
- Force `brain.command()` to time out (e.g. block the SDK client artificially) and confirm the watcher logs the failure, doesn't increment the counter, and survives to the next tick.
- Confirm a manually-spoken "clear"/"compact" verb still works unchanged — this spec adds a second caller of the same `brain.command()` path, not a replacement for the existing one.
