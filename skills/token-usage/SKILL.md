---
name: token-usage
description: Aggregates Claude Code token usage (input, output, cache write, cache read) from
  local session transcripts under ~/.claude/projects/, one row per conversation, and renders an
  interactive HTML report — stat tiles, a daily stacked bar chart by model, and a sortable full
  table — published as an Artifact. Handles subagent/Task-tool transcripts (nested under
  <sessionId>/subagents/) by attributing their tokens back to the parent conversation, and skips
  unreadable files (e.g. other users' sessions on a shared machine) without failing. Use whenever
  the user explicitly runs /token-usage, or asks things like "how many tokens have I used",
  "token usage history", "show my Claude Code usage", or "how much have I spent on tokens" for
  local/session-transcript data (not Console billing, which this cannot see).
disable-model-invocation: false
user-invocable: true
---

# token-usage

## Purpose

Turn `~/.claude/projects/**/*.jsonl` session transcripts into a per-conversation token-usage
report. Every assistant message in a transcript carries a `usage` block
(`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`); this
skill sums those per session, including subagent transcripts nested under
`<sessionId>/subagents/*.jsonl`, which get attributed back to their parent conversation since they
represent real token spend incurred by that conversation.

## Key concept: "billable" vs "total"

Cache reads dominate the raw sum once a conversation runs long (each turn re-reads the entire
prior context from cache). That number is real but misleading as a "how much did I use" headline
— cache reads are priced far below fresh input. Always report both:

- **billable** = `input + output + cache_creation` — tracks actual new work/spend
- **total** = `billable + cache_read` — the all-in figure, dominated by cache mechanics in long
  sessions

Lead with billable when asked "how many tokens have I used"; mention total as context, not as the
headline, and explain the gap if it's large (it usually is, often 10-30x).

## Steps

1. Run the collector to produce a rows JSON (one row per conversation):
   ```
   python3 ~/.claude/skills/token-usage/assets/collect_usage.py --json /tmp/token_usage_rows.json
   ```
   Optional filters: `--since YYYY-MM-DD`, `--project SUBSTRING`. Capture stderr — the last line
   is `SKIPPED_COUNT=N` (unreadable files, e.g. another user's sessions on a shared box).

2. Render the HTML report:
   ```
   python3 ~/.claude/skills/token-usage/assets/render_report.py /tmp/token_usage_rows.json /tmp/token_usage_report.html --skipped N
   ```
   (`N` = the `SKIPPED_COUNT` captured above.)

3. Publish it with the Artifact tool (`file_path` = the rendered HTML, pick a `📊` favicon). If
   updating a report published earlier in this conversation, republish the same file path to reuse
   the URL; if the user references a report from an earlier conversation, use `action: "list"` to
   find its URL and pass it as `url`.

4. In your reply, don't just link the artifact — pull 2-4 concrete numbers into the summary
   (billable total, all-in total, busiest day or project, model mix) so the answer stands on its
   own even before the user opens the link.

## Notes

- The report template (`assets/report_template.html`) assigns chart/legend colors dynamically by
  rank (busiest model gets slot 1), using the 8-slot validated categorical palette from the
  `dataviz` skill — it isn't hardcoded to today's model lineup, so new model names show up
  correctly without editing the template. Past 8 distinct models in one report, the rest fold into
  a neutral "Other" color rather than cycling hues.
- Model display names are derived generically (`prettyModel()` in the template's script strips the
  `claude-` prefix and any trailing date stamp, e.g. `claude-haiku-4-5-20251001` → `Haiku 4.5`) —
  no per-model lookup table to maintain.
- `collect_usage.py` also works as a standalone CLI for a quick terminal summary (no report): run
  it with no `--json`/`--csv` flags.
- This only sees local transcript data on this machine — it has no visibility into Console
  billing, other machines, or usage outside `~/.claude/projects/`. Say so if the user seems to
  want billing-accurate numbers.
