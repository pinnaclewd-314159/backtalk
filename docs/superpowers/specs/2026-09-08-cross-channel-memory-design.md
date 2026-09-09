# Cross-channel text memory (voice ↔ Telegram)

**Status:** approved by Sir 2026-09-08, implemented same night. Telegram side live and verified (bridge restarted, confirmed polling). Voice side (`backtalk/main.py`) written and compiles clean, but NOT yet live — needs a restart of the voice line to load, which ends whatever conversation is running at the time. Restart pending Sir's explicit go-ahead.

## Problem

Backtalk's voice line (this house, live conversational loop, `backtalk/main.py`) and the Telegram bridge (`tools/telegram_bridge.py`) are two entirely separate processes with no shared context. The bridge spins a fresh, memoryless Claude Agent SDK session per incoming message (deliberate 2026-09-01 design — "fresh and simple over a continuous thread"). Backtalk's voice session has full memory of its own turns for the life of that one session, but no visibility into what happened on Telegram, and Telegram has no visibility into voice.

Concrete trigger: Sir sent a photo on Telegram mid-voice-conversation with a caption ("This is the computer you are installed on"). The bridge's fresh session replied on Telegram with no awareness of the live voice conversation; the voice session had no way to see the photo, the caption, or the bridge's reply, and could only reconstruct what happened by grepping the bridge's log file after the fact.

Sir's ask: "I need you to have a memory between when I talk to you in this house and when I talk to you via Telegram and have a full contextual idea of what I've been talking to you about." Text only — images are not required to be shared or persisted across channels, only text (captions, questions, replies).

## Scope

- **In scope:** a shared, append-only, text-only transcript of every turn (user message + Jarvis reply) from both channels; both channels read a recent window of it before responding and write their own turns into it.
- **Retention surfaced automatically:** last 24–48 hours (Sir's call, "last day or two"). Nothing is ever deleted from the log itself — older history stays searchable (grep) even though it isn't auto-injected.
- **Awareness model:** check-per-turn, not live/real-time push. Confirmed with Sir: it's enough that each new turn (a new voice exchange, or a new incoming Telegram message) checks the shared log for anything new from the other channel since it last looked. No requirement to interrupt an in-progress turn if something arrives on the other channel mid-turn.
- **Out of scope:** images/photos (explicitly excluded by Sir), true real-time cross-channel push, a shared/unified single agent session (each channel keeps its own session mechanics — Telegram's fresh-per-message model is unchanged, voice's long-lived-per-session model is unchanged; only the context each starts with changes).

## Design

### Shared log

New module: `tools/cross_channel_log.py` — lives in `tools/`, not inside either backtalk's or any other package, since both `tools/telegram_bridge.py` and `backtalk/main.py` need to reach it and `tools/` is already the neutral integration-scripts location both draw on (matches `tools/homeassistant.py`'s existing role). Not under any git repo, consistent with the rest of `tools/`.

Storage: JSONL, `tools/cross_channel_log.jsonl`. One line per turn:
```json
{"ts": "2026-09-08T19:23:00-06:00", "channel": "voice"|"telegram", "role": "user"|"assistant", "text": "..."}
```
JSONL chosen over the `.remember` plugin's existing summarized-memory format on purpose — `.remember`'s `now.md`/`recent.md` files hold compressed, periodically-written summaries (a different cadence and purpose, longer-arc memory). This log needs low-latency, turn-by-turn raw text so a photo caption sent 30 seconds ago is visible immediately, not after whatever cadence produces a `.remember` summary. The two systems serve different jobs and shouldn't be conflated (no bloat: this is a new, narrowly-scoped file, not a rename or repurposing of `.remember`).

Module API:
- `append_turn(channel: str, role: str, text: str) -> None` — appends one JSON line. Uses a simple file lock (Windows: `msvcrt.locking` on a lock byte, held only for the duration of the append) since two independent processes write concurrently.
- `recent_transcript(hours: int = 48) -> str` — reads the file, filters to entries within the window, formats as a plain-text block (`[19:23 voice] Sir: ...` / `[19:24 voice] Jarvis: ...`) ready to prepend into a system/context message. Returns `""` if nothing in window or file doesn't exist yet.
- Both functions fail soft: any exception (missing file, bad line, lock timeout) is caught and logged to stderr, never raised into the caller — a broken shared log must not take down either channel's ability to respond.

### Telegram bridge integration (`tools/telegram_bridge.py`)

- Before building the `content` for `_ask_jarvis()`, call `recent_transcript(hours=48)` and prepend it as a leading text block (clearly delimited, e.g. `"Recent cross-channel history (voice + Telegram), last ~2 days:\n<transcript>\n---\n"`) ahead of the actual incoming message content.
- After a reply is generated (success path only — not on timeout/error, since nothing meaningful was actually said), call `append_turn("telegram", "user", <incoming text or caption>)` and `append_turn("telegram", "assistant", <full reply text>)`. For photos, log the caption only (or `"[photo, no caption]"` if none) — never the image bytes, per Sir's explicit text-only scope.

### Backtalk voice integration (`backtalk/main.py`)

- At the point where each new voice turn is about to be handled (same place `source` already distinguishes PTT/typed/satellite turns), call `recent_transcript(hours=48)` filtered to entries where `channel == "telegram"` newer than the last one this session already surfaced (track a simple `_last_seen_telegram_ts` in memory for the life of the process) — avoids re-injecting the same Telegram history into every single turn once it's already been seen this session. If there's anything new, fold it into context ahead of handling Sir's turn (as a brief system-style note, not spoken aloud unless relevant).
- After each voice turn completes, call `append_turn("voice", "user", <what Sir said>)` and `append_turn("voice", "assistant", <what Jarvis replied>)` so Telegram's next fresh session sees it.
- **This is a real edit to `backtalk/main.py`, the file running this very conversation.** Building and testing it doesn't affect the live process; loading it requires a restart of the voice line, which ends the current conversation the same way the Voicebox wiring did earlier tonight. Sir gets an explicit heads-up immediately before that restart is triggered, not folded silently into a broader checkpoint.

### Error handling

- Missing/corrupt log file: treated as empty history, not an error surfaced to Sir.
- Lock contention (near-simultaneous writes from both channels): short retry loop (a few ms, a handful of attempts) before giving up and skipping that particular append with a stderr log line — losing one turn from cross-channel history is acceptable; blocking either channel's actual reply on a lock is not.
- Unbounded growth: JSONL of short text lines is small (a very active day is a few hundred KB at most) — no rotation/pruning needed for now. Revisit only if it actually becomes a problem.

### Testing

- Unit: `append_turn`/`recent_transcript` round-trip (write N entries across both channels, confirm the window filter and formatting are correct); missing-file and corrupt-line behavior (skip bad lines, don't crash).
- Integration, manual: send a Telegram text message, confirm it lands in the JSONL; ask about it in the next voice turn (after the restart) and confirm it's surfaced; reverse direction, mention something in voice, confirm the next Telegram message's fresh session references it correctly.
