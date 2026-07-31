---
name: training-queue
description: Schedules one or more long-running commands (typically GPU training runs inside a
  Docker container) to execute unattended via cron -- sequentially, in a single self-cleaning
  script, with per-stage logs and a summary log. Also handles appending more jobs to a queue
  that's already scheduled or already running. Use when the user wants to "run this overnight",
  "schedule a training run", "queue up experiments while I'm away/idle", "run these one after
  another", "add another training to the queue", or explicitly invokes /training-queue. Verifies
  each job's command actually works (config dry-run + a fast smoke pass) before committing to the
  unattended run -- never queues a job untested.
disable-model-invocation: false
user-invocable: true
---

# training-queue

## Purpose

Let long GPU jobs run unattended while the user is away, by queuing them as one cron-triggered
shell script instead of babysitting each one interactively or guessing at durations to chain them
by time.

## Core pattern

1. **One self-cleaning cron entry**, not `at` (often not installed) — a dated line
   (`min hour day month *`) fires once on today's date; the script removes its own crontab line
   as its last step, so it can't fire again next year.
2. **One shell script runs every queued job sequentially** — not `&&`-chained (one crash would
   silently kill everything after it) and not scheduled as separate cron times (durations are
   unknown up front, so time-based chaining risks two GPU jobs overlapping).
3. **Each job logs to its own file**; a shared summary log records a start/end timestamp + exit
   code per stage, so progress is checkable at a glance without grepping full training logs.
4. **Docker jobs**: `docker exec -w <workdir> <container> /bin/bash -c "<command>"` per stage.
   Confirm the container is already running (`docker ps`) — don't `docker run` a fresh one that
   could conflict with it.

## Steps

1. **Gather the job list** from the user (or a doc's "still to run" table) — exact command +
   overrides per job, in run order.
2. **Check idle state first**: `nvidia-smi` (GPU actually free?), `docker ps` (target container
   up?), `crontab -l` (nothing already scheduled that would collide with this window).
3. **Verify every job's command before scheduling it — this is a hard gate, not an optional
   nicety.** A job that crashes at step 1 of an unattended overnight queue wastes every GPU-hour
   budgeted for the jobs after it, discovered only the next morning. Do not add a job to the queue
   script until it clears both checks:
   - *Config-only check*: run the entrypoint with its dry-run flag (e.g. Hydra's `--cfg job`) —
     catches a typo'd override key without launching anything.
   - *Smoke check*: if the project has a lightweight forward-pass/benchmark tool, run it with the
     **same overrides** at minimal batch size — catches instantiation errors a config dry-run
     can't (bad `_target_`, shape mismatch, missing class). Confirm exit code 0 and no exception
     in the tail of output, for every job in the queue, not just the first.
4. **Write the queue script** from `assets/queue_template.sh` — one `run_stage <name> <logfile>
   <command>` call per job. Un-chained, sequential, each stage's exit code logged regardless of
   success.
5. **Schedule it** by reading, then appending to, the crontab — never blind-overwrite:
   `( crontab -l 2>/dev/null; echo "<min> <hour> <day> <month> * <script path>" ) | crontab -`.
6. **Report back plainly**: what's queued and in what order, a total time estimate (sum of
   expected per-job durations, or "unknown" if a config hasn't run before), where the summary log
   lives, and how to cancel (`crontab -e` to edit, `crontab -r` to clear everything).

## Adding jobs to an existing queue

Two different cases — check which one applies before touching anything:

- **Queue hasn't fired yet** (its cron entry is still in `crontab -l`, summary log doesn't exist
  or has no `QUEUE COMPLETE` line): safe to edit the pending script directly. Run the new job
  through the same hard-gate verification as step 3 above, then insert its `run_stage` call
  *before* the trailing summary/self-remove lines at the end of the file. Nothing else changes —
  same cron entry picks it up.
- **Queue is already running**: do **not** edit the live script file — a currently-executing bash
  script reading from disk can behave unpredictably if the file changes underneath it. Instead,
  write a second, small "continuation" script that (1) polls the shared summary log until it
  contains `QUEUE COMPLETE`, then (2) runs the new job(s) the same way, appending to that same
  summary log. Schedule the continuation with its own near-immediate, self-cleaning cron entry
  (fires in a couple of minutes, then sits in a polling loop) rather than guessing when the
  running queue will finish — an ETA-based schedule risks a second GPU job starting while the
  first is still on the card.

## Dataloader workers on a consumer-grade GPU

If the target is a single consumer desktop GPU (not a shared/datacenter multi-GPU node), keep
`num_workers` modest relative to `batch_size` — an oversized worker pool competes with the training
process itself for host CPU/RAM on a machine that doesn't have much headroom to spare:

| `batch_size` | `num_workers` |
|---|---|
| 1 | 2 |
| 2 | 4 (max) |

For batch sizes outside this table, don't keep extrapolating linearly — check host core count
(`nproc`) and any project-specific guidance already written down (e.g. a CLAUDE.md note recording
a validated batch/worker combo) before picking a value. This table is a default for *new* configs
being queued, not a reason to override a combo the project has already confirmed works.

## Verifying the schedule actually fires (cheap, worth doing once per environment)

Before trusting a multi-hour unattended queue on a new host/container, schedule a 2-minute dummy
job first: write a timestamp + run a trivial command inside the target container, then
self-remove. Confirms the cron → container path end-to-end for near-zero cost. Skip this if the
exact same cron/container path was already verified earlier in the same environment.

## Common pitfalls

- **No `at` daemon on minimal hosts** — check `which at` before reaching for it; a dated one-shot
  cron line works everywhere cron does and self-removes just as cleanly.
- **Comma-separated list overrides are ambiguous to Hydra** (`key=a,b,c` looks like a list). Quote
  it (`key='a,b,c'`), escaped as `key=\'a,b,c\'` when the whole command is itself wrapped in
  `docker exec ... bash -c "..."`.
- **Read-modify-write the crontab, never blind-overwrite** — always `crontab -l` first and
  append/filter, so unrelated jobs the user already had scheduled survive.
- **Don't `&&`-chain jobs on one line** — a single failed stage would silently cancel every job
  queued after it. Run each stage independently, capture its exit code, move on regardless.

## Self-Improve

**Check `references/use-cases.md` first**, before building a new queue — most environment quirks
(no `at` daemon, a project's own dry-run flag, a non-Docker target, a different scheduler
entirely) have likely already been hit once and catalogued there; don't rediscover them from
scratch.

After using this skill, if the invocation surfaced anything not already in that file — a new kind
of pre-flight check, a different container/scheduler setup, a GPU-tier worker-count rule for a
different class of hardware, a new failure mode a job hit mid-queue — add a dated entry to
`references/use-cases.md`. If an existing entry just got reconfirmed rather than extended, add the
project's name to its "confirmed in" list instead of duplicating it. This is what keeps the next
invocation from re-deriving the same thing.

## Reference

- `assets/queue_template.sh` — copy-paste skeleton: `run_stage` helper, per-stage
  `docker exec` + logfile, shared summary log, self-removing crontab line at the end. Fill in the
  container name, repo path, and one `run_stage` call per job.
- `references/use-cases.md` — indexed catalog of environment-specific quirks and edge cases seen
  across projects; check before starting, extend after finishing.
