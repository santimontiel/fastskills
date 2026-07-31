# training-queue: use-case catalog

Indexed by environment/quirk, not chronologically — check here before building a new queue.
Each entry: the situation, what to do about it, and which project(s) it's been confirmed on.

---

## No `at` daemon installed

**Situation**: `which at` finds nothing, or `atd` isn't running. Common on minimal Docker hosts
and bare cloud VMs — `at` usually isn't installed by default even when `cron` is.

**What to do**: don't install `at` just for this (an unrequested package change). Use a dated
one-shot cron line instead — `<min> <hour> <day> <month> *` fires exactly once on that date, and
the queue script removing its own crontab line makes it behave like a true one-shot. Confirm
`cron`/`crond` itself is active first (`systemctl is-active cron`) — it almost always is, since
it's a base-image default far more often than `at`.

**Confirmed in**: tinycar-dev (2026-07-31).

---

## Hydra comma-separated list overrides need quoting

**Situation**: an override like `compile_stage=image_encoder,camera_unprojection,decoder` fails
with "Ambiguous value for argument" — Hydra reads the bare comma as list syntax.

**What to do**: quote the value (`compile_stage='a,b,c'`). When the whole command is itself
wrapped in `docker exec ... /bin/bash -c "..."`, the outer double quotes consume a bare `'`, so
escape it: `compile_stage=\'a,b,c\'`.

**Confirmed in**: tinycar-dev (2026-07-31) — hit on `compile_stage`, a project-specific Hydra key
(not `compile_stages`, a different key that exists in the same repo's `train.py` config but not
`benchmark.py`'s — worth double-checking the exact key name per entrypoint, they aren't always
shared between a project's own scripts).

---

## Consumer single-GPU desktop: dataloader worker count

**Situation**: scheduling training on a single consumer-grade GPU workstation (not a
shared/datacenter multi-GPU node) — an oversized `num_workers` pool competes with the training
process for host CPU/RAM that a desktop doesn't have much slack in.

**What to do**: default to `batch_size=1 → num_workers=2`, `batch_size=2 → num_workers=4` (cap).
For other batch sizes, check `nproc` and any project-specific note (e.g. a CLAUDE.md line
recording a validated combo) rather than extrapolating linearly — don't override a combo the
project has already confirmed works.

**Confirmed in**: tinycar-dev (2026-07-31), RTX 5090 desktop.

---

## Verifying a job before queueing it: config dry-run alone isn't enough

**Situation**: a Hydra (or similar) config dry-run flag (`--cfg job`) confirms the override keys
resolve, but doesn't call `hydra.utils.instantiate` (or equivalent) — so a bad `_target_` path, a
shape mismatch, or a missing class only surfaces once the real job is already running unattended.

**What to do**: pair the config-only check with a fast smoke pass through the project's own
lightweight forward-pass/benchmark tool if one exists (e.g. `tools/benchmark.py`), using the exact
same overrides at minimal batch size. Confirm exit code 0 and no exception in the tail of output
for *every* queued job, not just the first — each job's overrides differ, so a clean smoke pass on
job 1 says nothing about job 3's `_target_` typo.

**Confirmed in**: tinycar-dev (2026-07-31) — a 4-job overnight queue, each config smoke-tested via
`tools/benchmark.py` before being added to the script.

---

## Adding a job to a queue already mid-run

**Situation**: the queue script is currently executing (summary log has a `START` but no
`QUEUE COMPLETE`), and a new job needs to be appended.

**What to do**: don't edit the live script file in place. Write a small continuation script that
polls the shared summary log for `QUEUE COMPLETE`, then runs the new job(s) using the same
`run_stage` pattern, appending to the same summary log. Schedule it with its own near-immediate
cron entry rather than an ETA-based one — a guessed completion time risks two GPU jobs overlapping
on one card.

**Confirmed in**: not yet exercised end-to-end on any project — written from the same
already-running-script-editing risk noted in step 3 of `SKILL.md`. Update this entry with a real
project name once actually used.
