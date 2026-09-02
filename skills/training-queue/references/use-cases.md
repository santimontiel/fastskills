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

**Confirmed in**: tinycar-dev (2026-07-31); gaussiancarpred-dev (2026-08-10).

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

**Confirmed in**: tinycar-dev (2026-07-31), gaussiancarpred-dev (2026-08-10), RTX 5090 desktop.

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

**Extension — when the change under test is in the *data* layer, benchmark.py is not a valid
smoke** (tinycar-dev, 2026-08-08): `tools/benchmark.py` synthesizes its own random batch and never
constructs the DataModule, so it exercises none of the store, dataloader, worker processes,
collate, or Lightning's `on_after_batch_transfer`. Smoke with a real short run instead —
`tools/train.py ... trainer.max_epochs=1 trainer.limit_train_batches=6 trainer.limit_val_batches=3`
with the *exact* production overrides — and leave `compile_stages` at its default so
`torch.compile` is exercised too. Pick the smoke tool by what the change touches, not by what is
cheapest.

---

## Estimate the queue's duration from a warm-cache rate probe, not from the smoke

**Situation**: smoke-run wall times are dominated by `torch.compile` warmup (104 s for 6 batches
cold vs 39 s warm, same project), so extrapolating an overnight ETA from them is wildly wrong.

**What to do**: after the smokes (which leave the compile cache warm), run one short probe with
enough steps to reach steady state and read the rate off the progress bar:
`... limit_train_batches=60 2>&1 | tr '\r' '\n' | grep -oE '[0-9.]+it/s' | tail -3`. Combine with
the split sizes (`wc -l < <store>/splits/train.txt`) for a real per-epoch figure. Report the total
*before* the queue fires — a request phrased as an innocuous "10 epochs each" was ~31 h here, which
the user should get to veto while the cron entry is still 5 minutes out.

**Confirmed in**: tinycar-dev (2026-07-31 / 2026-08-08), RTX 5090, 1.31 it/s at batch_size=4.

**Extension — tqdm's `it/s` is a running average over the whole epoch, not a steady-state rate**
(tinycar-dev, 2026-08-08): a probe of `limit_train_batches=120` reported 1.30 it/s while the true
warm rate was 7.0 it/s, because the first several batches carry cudnn/compile warmup and tqdm
averages them in for the entire epoch. Reading the *last* value off a single-epoch probe therefore
still under-reports by ~5x, which turns a 5 h queue into a quoted 34 h. Run `max_epochs=3` and read
the **final epoch's** line (`grep -E "^Epoch [0-9]+: 100%"`) instead — epoch 2 starts warm. Then
sanity-check the quote against the real run's own progress bar a few minutes after it fires.

---

## Secrets must be forwarded at `docker run` time, not per `docker exec`

**Situation**: an agent-created container (started ad-hoc for interactive work, rather than via the
project's `make run`) is missing `HF_TOKEN`/`WANDB_API_KEY`, because those were passed per-command
as `docker exec -e HF_TOKEN=... ` during the session. Cron's `docker exec` in the queue script has
no such flag, so the unattended run loses both — and a **gated** model checkpoint (e.g. DINOv3 on
HF) makes that fatal minutes after everyone has gone to bed, not a degraded-logging annoyance.

**What to do**: check *inside* the container (`docker exec <c> bash -lc 'echo ${HF_TOKEN:+present}'`)
and, if missing, recreate the container with `-e HF_TOKEN -e WANDB_API_KEY` (bare, no `=`, so the
values come from the host env and never land in a script or a log). Recreating is cheap when
nothing is running in it. Do this *before* the smoke run, or the smoke passes for reasons the
queued job won't have.

**Careful with the check itself**: `echo "${VAR:+SET}${VAR:-UNSET}"` prints the *value* when the
variable is set, because `${VAR:-UNSET}` expands to it. Use `${VAR:+present}` alone.

**Confirmed in**: gaussiancarpred-dev (2026-08-10).

---

## Multi-frame configs move the batch-size ceiling — probe it, don't inherit it

**Situation**: a temporal/multi-frame task reuses a single-frame project's batch size. Activation
memory scales with the frame count, so a batch size that has always worked now OOMs.

**What to do**: probe the intended batch size before queueing, and treat the result as a ceiling
rather than a preference. On gaussiancarpred-dev's `task=tempseg` (3 camera frames), `batch_size=4`
OOMed a 32 GB RTX 5090 outright while `batch_size=2` ran at 2.26 it/s. Gradient accumulation
(`effective_batch_size`) absorbs the change, so dropping the batch size costs throughput but not
the optimisation trajectory — say so when reporting the ETA, since a smaller batch is the reason
the night got longer.

**Confirmed in**: gaussiancarpred-dev (2026-08-10), RTX 5090 32 GB.

---

## Unattended runs need their logger's auth checked as a pre-flight

**Situation**: the project's default logger is W&B; an unset `WANDB_API_KEY` makes an interactive
run prompt and an unattended one fail minutes after the user has walked away.

**What to do**: check the variable *inside the target container*, not just on the host — a
container started without `-e WANDB_API_KEY` will not have it even when the host does. If it is
missing, either export it into the container or queue with `logger=csv` rather than discovering it
overnight.

**Confirmed in**: tinycar-dev (2026-08-08); gaussiancarpred-dev (2026-08-10, 2026-08-12 -- WANDB_API_KEY absent inside an agent-started container, queued with logger=csv).

---

## "Has my queue fired yet?" — a shared summary log makes this ambiguous

**Situation**: the skill's default `SUMMARY="$REPO/outputs/queue_summary.log"` is reused by every
queue the project has ever scheduled. Checking "has this queue started?" by testing whether that
file exists returns a **false positive** whenever an earlier queue completed — the file is there,
full of a previous run's stages, and it is easy to conclude the new queue is already running and
therefore unsafe to edit.

**What to do**: give each queue its own summary file
(`outputs/queue_<purpose>.log`). Then existence *is* the answer, and the mid-run test from the
section below (`START` present, `QUEUE COMPLETE` absent) is unambiguous. If a shared log must be
used, test for this queue's own stage names and timestamps, never for the file.

**Confirmed in**: tinycar-dev (2026-08-08) — a 2026-07-31 queue's completed log was still present
when a new queue was scheduled 8 days later.

---

## Editing a pending queue after the user changes their mind

**Situation**: user revises a parameter (e.g. "10 epochs -> 6") after the cron entry is already in
place but before it fires.

**What to do**: this is the "hasn't fired yet" case — edit the script in place, no crontab change
needed, since the entry points at a path rather than at content. Update *everything* keyed to the
old value in the same pass: `max_epochs`, each `run_id`, per-stage log filenames, and the ETA in
the `QUEUE START` line, or the logs and W&B runs will be named after a configuration that never
ran. Re-run `bash -n` afterwards and re-check `date` against the cron minute — the edit has to land
before it fires, and there is no second chance.

**Confirmed in**: tinycar-dev (2026-08-08), edited with ~2.5 min of margin.

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

---

## Queueing behind an unrelated project's queue on a shared single GPU

**Situation**: the target host has only one GPU, and a *different* project already has its own
cron-scheduled queue occupying the time slot the user wants (e.g. `crontab -l` shows another
project's `experiment_queue_2pm.sh` already firing at 14:00). This isn't the "add a job to my own
queue" case above — it's two unrelated projects, each with their own repo/container/summary log,
that happen to need the same GPU.

**What to do**: don't append to the other project's queue script or summary log — they're
unrelated and mixing them makes both harder to audit. Instead write a *separate* continuation
script for the new project, with its own `SUMMARY` under its own repo, that polls the *other*
project's summary log for `QUEUE COMPLETE` before running anything. Schedule it with its own
near-immediate cron entry (fires in a couple of minutes, then polls in a `sleep`-loop), same
rationale as the same-project case: an ETA guess risks two jobs landing on the GPU at once. Always
surface the conflict to the user before picking a resolution — running at the requested time
anyway, waiting, or picking a different slot are all legitimate depending on how tight host RAM/GPU
is; don't silently decide.

**Confirmed in**: gcarpred-dev (2026-07-31) — tinycar-dev's `experiment_queue_2pm.sh` already
occupied 14:00 on the same host; wrote `gcarpred_sanity_after_tinycar.sh` polling tinycar-dev's
`outputs/queue_summary.log` every 5 min, scheduled at a near-immediate cron time, confirmed via a
live wait that it fires and enters the polling loop correctly.

---

## Smoke-testing when there's no separate benchmark tool matching the train config

**Situation**: the project has a lightweight benchmark/forward-pass script (e.g. `tools/benchmark.py`),
but it's wired to its own separate Hydra config (`benchmark.yaml`) rather than `train.yaml` — so
smoke-testing through it doesn't actually exercise the same config tree, overrides, callbacks, and
logger the queued job uses.

**What to do**: if the training entrypoint's own config already exposes batch-count limiters (e.g.
Lightning's `trainer.limit_train_batches` / `trainer.limit_val_batches`), smoke-test by calling the
*actual* queued command with those added (`limit_train_batches=3 limit_val_batches=3`) instead of
reaching for the separate benchmark tool. This exercises the literal command that will run
unattended — including checkpoint loading, the real loss/metric wiring, and callbacks — at the cost
of a `max_epochs=1` override. It also catches callback/logger interactions the benchmark tool
wouldn't (e.g. `LearningRateMonitor` raising `MisconfigurationException` when the smoke test
overrides `loggers=[]` to reduce noise — don't strip the logger, the callback depends on it).

**Confirmed in**: gcarpred-dev (2026-07-31) — `tools/benchmark.py` uses `configs/benchmark.yaml`,
unrelated to `train.yaml`; smoke-tested `tools/train.py` directly with
`limit_train_batches=3 limit_val_batches=3`, which caught a `loggers=[]`-induced
`MisconfigurationException` on the first attempt (test-harness mistake, not a real bug) before a
clean exit-0 pass on the second.

---

## Long-lived container predates a mount the queued job needs

**Situation**: the project keeps one long-running container (`make run`) that the skill would
normally `docker exec` into, but the job needs a dataset mounted only after that container was
started — e.g. a third dataset added to the Makefile the same day. `docker exec` then runs against
a container that cannot see the data, and fails at DataModule setup minutes into an unattended run.

**What to do**: check the running container's actual mounts before assuming `docker exec` works:
`docker inspect <name> --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'`.
If the needed mount is absent, have `run_stage` use `docker run --rm` with a per-stage `--name`
instead. This contradicts SKILL.md's default advice for a good reason, so say why in a comment at
the top of the queue script. Confirm first that the long-lived container is genuinely idle (a
shell, not a job) and the GPU is free, since the two containers will share the card. Asking the
user to restart their container is the other valid resolution — it just costs them their session.

**Confirmed in**: tinycar-dev (2026-08-08) — `tinycar_container` was started before View of Delft
was added to the Makefile's mount list, so the VoD queue used `docker run` per stage.

---

## Secrets a cron job needs, without putting them in the repo

**Situation**: cron runs with a minimal environment, so `HF_TOKEN` / `WANDB_API_KEY` / dataset-path
exports present in the interactive shell are all absent. Sourcing `~/.bashrc` does not fix it —
most distro `.bashrc` files return early on a non-interactive shell. Hardcoding the values into the
queue script puts live credentials in a file inside the repo.

**What to do**: write the exports once to a private file outside the repo
(`~/.config/<project>-queue.env`, `chmod 600`, created under `umask 077`) and have the queue script
`set -a; . "$HOME/.config/<project>-queue.env"; set +a`. The script stays committable and
credential-free, and the env survives cron's stripped environment. Verify the gated dependency
actually needs it before assuming — here a gated HuggingFace backbone made model construction fail
with a 401 long before any data was touched.

**Confirmed in**: tinycar-dev (2026-08-08).

---

## `torch.utils.checkpoint` and `torch.compile` on the same submodule are incompatible

**Situation**: a memory-saving gradient-checkpoint flag wraps a block that also appears in
`compile_stages`. Training crashes several minutes in — *after* the compile warmup, i.e. long
after the smoke would have looked fine if the smoke had skipped compile — with:

```
torch.utils.checkpoint.CheckpointError: A different number of tensors was saved during the
original forward and recomputation.
```

The compiled module saves a different number of tensors on the recompute than on the original
forward, so the checkpoint's bookkeeping fails. It is not a tolerance or a flakiness issue and it
does not depend on the model.

**What to do**: drop just the checkpointed block from `compile_stages` (here
`compile_stages='image_encoder,camera_unprojection'`, i.e. everything except `decoder`) —
confirmed to run clean. Better, add an explicit pre-flight in the entrypoint that raises with the
fix in the message when the two are combined, so it fails in the first second rather than after
warmup. Note the ordering trap this creates for the *skill*: a smoke run with
`compile_stages=none` passes and tells you nothing, so smoke every queued job at the **production
compile setting**, not a cheaper one.

**Confirmed in**: gaussiancarpred-dev (2026-08-12), `task=prediction`,
`module.model.checkpoint_decode=True` + compiled `decoder`, PyTorch 2.8.0.

---

## Hydra comma-quoting, reconfirmed on a second project

The "Hydra comma-separated list overrides need quoting" entry above also applies to
`compile_stages` (plural) in gaussiancarpred-dev, 2026-08-12: `compile_stages=a,b` raises
"Ambiguous value for argument", and inside `docker exec ... bash -c "..."` it needs
`compile_stages=\'a,b\'`. Same quirk, different key name from tinycar-dev's `compile_stage`
(singular) — check the exact key per entrypoint, as that entry already warns.

---

## A horizon/multi-output task moves the memory ceiling again, and the obvious knob is the wrong one

**Situation**: extending a multi-frame *input* task to multiple *output* frames (BEV instance
prediction: 3 input frames in, 5 predicted frames out). The instinct is to shrink the
transformer's query budget, since the per-query mask tensor `(B, Q, T_out, H, W)` looks like the
dominant term.

**What to do**: measure the decomposition before choosing a knob. On gaussiancarpred-dev
(RTX 5090, 32 GB, bf16, one forward+backward at `batch_size=1`): `T_out=5` peaked at 23.91 GiB,
`T_out=1` at 15.37 GiB, so the horizon costs ~2.1 GiB per output frame over a ~15.4 GiB floor that
is the image encoder at 18 views. Halving `num_queries` (100 → 50) saved **2%**; gradient
checkpointing the fuse+decode block saved 7%; disabling the whole motion predictor saved 0.3%.
`batch_size=2` OOMed regardless. So the only real levers are the horizon length itself and the
input frame count — report the measured table rather than a plausible-sounding knob order.

**Confirmed in**: gaussiancarpred-dev (2026-08-12), RTX 5090 32 GB.

---

## "N epochs per run" on a multi-config sweep: quote the total before writing the script

**Situation**: the user asks for a hyperparameter search at a fixed epoch count per run ("10
epochs per run"), phrased as a property of one run rather than of the queue. The number of
configs multiplies it, and on a single consumer GPU forced to `batch_size=1` there is no
parallelism to absorb it.

**What to do**: measure both rates (train *and* val separately — val can be 2x faster per clip
than train, so folding them into one number misestimates by hours over a long queue), multiply
out, and put the total in front of the user *as a decision* before writing any script. On
gaussiancarpred-dev: 2.28 it/s train over 23930 clips = 2.92 h/epoch, 5.43 it/s val over 5119 =
0.26 h/epoch, so 8 configs x 10 epochs = **254 h / 10.6 days**. Offering the arithmetic alongside
three or four costed alternatives (staged screen-then-finalists; fewer configs; fewer epochs) got
a decision in one exchange — the user picked 8 x 3 epochs / 76 h. Silently shrinking the request
would have been worse than asking, and so would have been queueing 10.6 days of GPU.

**Confirmed in**: gaussiancarpred-dev (2026-08-12), RTX 5090 32 GB, `task=prediction`.

---

## Editing a source file the *running* queue imports

**Situation**: the queue's stages are `docker exec ... uv run tools/train.py` against a
bind-mounted repo. Editing any module those runs import — not the queue script, the *source* —
means later stages execute different code than earlier ones, silently.

**What to do**: treat the mounted source tree as frozen for the queue's duration, the same way
the skill already treats the running queue script. This is stricter than it sounds for a smoke
queue in particular, whose whole value is that all N stages tested one revision. If an edit
cannot wait, kill the queue rather than let it straddle two versions. On gaussiancarpred-dev an
import line was added ~40 s after a 7-stage smoke queue started and was reverted before stage 2
began; the tell was that stage 1 had already logged `exit 0` against the old file.

**Confirmed in**: gaussiancarpred-dev (2026-08-12).

---

## A project's `--cfg job --resolve` dry-run can fail on its own custom resolvers

**Situation**: the config-only pre-flight (`tools/train.py <overrides> --cfg job --resolve`) dies with
`UnsupportedInterpolationType: Unsupported interpolation type mult` — nothing to do with the overrides
being tested. The project registers custom OmegaConf resolvers (`mult`, `add`, `last_token`) from
inside its `@hydra.main` function body, and `--cfg` prints the config *without* ever entering that
function, so the resolvers are never registered.

**What to do**: drop `--resolve` and run plain `--cfg job`. It still parses and applies the overrides,
so it catches the thing this gate is for — a typo'd or non-existent override key — and simply prints
interpolations unexpanded. If genuinely resolved values are needed (checking that a task's
`camera_frames` actually reaches `module.model`), write a throwaway script that calls the project's
`register_new_resolvers()` and then `hydra.compose(...)`; also call
`HydraConfig.instance().set_config(cfg)` or any `${hydra:runtime.choices.*}` interpolation (common in
logger configs) raises `HydraConfig was not set`.

**Confirmed in**: gaussiancarpred-dev (2026-08-19).

---

## Smoke the eval/resume stage too — a `torch.compile` checkpoint won't reload into an eager net

**Situation**: the train stage sets `compile_stages`, so Lightning saves the wrapped submodules'
keys with a `_orig_mod.` infix (`feature_extractor.bev_projections.0._orig_mod.reduce.0.weight`).
The queue's *eval* stage (and any resume) loads into an un-compiled network and dies with
`checkpoint did not match the network: N missing, N unexpected` — discovered only after training
already spent the night, because the smoke tested train but not eval-of-the-trained-ckpt.

**What to do**: smoke the eval/resume stage against a *real* mid-training checkpoint, not just the
published one. Fix belongs in the project's checkpoint loader: strip `_orig_mod.` from every key
before `load_state_dict` (a compile wrapper artifact, not a real parameter name). Re-check the
project's forward/numerics anchor after touching that loader.

**Confirmed in**: rapidradar-dev (2026-08-31) — `compile_stages="voxel_encoder,bev_projections"`
in `configs/train.yaml`; added `strip_compile_prefix` to `rapidradar/utils/checkpoint.py`.

---

## A prerequisite data artifact with no in-repo builder — port it from the named upstream repo

**Situation**: the data-prep entrypoint (`tools/create_data.py`) needs a per-sequence artifact
(`map_clean.npy`, an aggregated static map) that nothing in the repo produces — the README just
says "follow the instructions in the [upstream] repository". Raw SemanticKITTI was downloaded but
is not training-ready, so no training can run until the artifact exists.

**What to do**: find the upstream script (here LiDiff's `lidiff/map_from_scans.py`), port it into
`tools/` dropping deps the repo deliberately removed (MinkowskiEngine → the repo's own
`torch.unique`-based `efficient_voxel_downsample`), and smoke it on the *smallest* sequence
(seq 04, 271 frames) before the full 11-sequence run. Write a `dev/verify_dataprep.py` that checks
the obvious failure modes (empty/NaN, wrong shape, map bbox not covering the trajectory) since
there is no upstream reference to diff against. Also: `configs/rapidradar.yaml` (the flat config
`create_data.py` and `model.py` default to) was missing entirely — recreate it from the upstream
`configs/rapidlidar.yaml` in the init commit.

**Confirmed in**: rapidradar-dev (2026-08-31) — added `tools/build_maps.py`,
`dev/verify_dataprep.py`, `configs/rapidradar.yaml`; ~20 min maps + ~77 min gt/input for 23k
frames, ~60 GB. Container started detached (`sleep infinity`, `--restart unless-stopped`) because
`make run` is interactive-only. Dated one-shot cron line fired cleanly on this host (Ubuntu,
`cron` active, no `at`).

---

## An unrelated container can be holding most of the VRAM — check processes, not just free memory

**Situation**: pre-flight `nvidia-smi` shows plenty of total VRAM, but gates and smoke runs OOM at
implausibly small allocations ("tried to allocate 782 MiB ... this process has 5.95 GiB in use"). A
long-lived unrelated container — here an LLM inference server, `llama-cpp-server`, up 45 hours and
holding 25.7 GiB of a 32 GB card — was the real ceiling.

**What to do**: in the idle-state check, run `nvidia-smi --query-compute-apps=pid,process_name,used_memory
--format=csv` alongside the memory query. A free-memory number alone reads as "6 GiB free, fine" when the
real story is "one process owns 79% of the card". The tell is an OOM whose *own* process usage is small
relative to the card. Do not stop another container without asking — it is someone's running service —
but say plainly that the two workloads cannot coexist when their peaks sum past the card, and let the
user choose. Restart it with `docker start <name>` afterwards; a stopped (not removed) container keeps
its config.

**Confirmed in**: gaussiancarpred-dev (2026-08-19), RTX 5090 32 GB — 25.7 GiB llama-cpp-server vs a
19.0 GiB training peak, so the pair never fit and the training queue required the server down.
