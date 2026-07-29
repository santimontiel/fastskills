---
name: repo-adapter
description: Adapts an externally-authored research or baseline repo (a paper's code release, a cloned
  baseline) onto this team's internal compute infrastructure — uv-managed packaging, Docker + Slurm
  cluster deployment, a tools/ entrypoint layout, an installable src package, and optionally a Hydra
  config migration or a PyTorch Lightning training-loop port. Use this whenever the user explicitly runs
  /repo-adapter, or asks to "adapt", "port", "onboard", "dockerize", "productionize", or "modernize" an
  external/baseline/paper repo onto internal or cluster infra, to "match our infra conventions", or to
  make a cloned repo "buildable on the cluster" / "deployable via Slurm". Do not trigger automatically
  just because a repo looks old or has outdated dependencies — this is a heavy, multi-file-write, highly
  interactive workflow that must be explicitly requested, never inferred.
disable-model-invocation: true
user-invocable: true
---

# repo-adapter

## Purpose

Codify the repeatable process of taking an externally-authored research repo and adapting it onto this
team's internal compute infrastructure — `uv` packaging, Docker + Slurm deployment, a `tools/` entrypoint
layout, an installable src package, and (conditionally) a Hydra config migration or a PyTorch Lightning
training-loop port. This process has been done several times already (`fiery-radar`, `powerbev-radar`,
and most recently a repo with a hand-rolled `torch.distributed` loop rather than Lightning) — this skill
generalizes across all of it.

**Treat whichever reference repo is chosen for a given invocation as the live source of truth.** Read its
actual current files every time; never assume "the infra pattern" is a fixed, memorized template. Known
sibling repos that follow this infra (`tinycar-dev`, `gaussiancar`, `gcarpred-dev`, `fiery-radar`,
`powerbev-radar`) are **not** byte-identical — dataset mount counts, a `jupyter` Makefile target,
`deploy/slurm/debug_terminal.sh`, and vendored-CUDA-extension `pyproject.toml` machinery all vary
per-repo for justified reasons. Copying one repo's files wholesale into a target that doesn't need them
(e.g. CUDA-extension build machinery for a repo with no vendored extension) is a mistake, not diligence.

## Invocation

`/repo-adapter [target-repo-path] [reference-repo-path-or-name]` — both arguments optional.

- `target-repo-path` defaults to the current working directory.
- `reference-repo-path-or-name`: if given, resolve and read it directly. If omitted, **do not guess**.
  Glob the target repo's parent directory (and any other workspace roots already known from context) for
  sibling repos with infra fingerprints (`deploy/` dir, a `pytorch-cuXXX` uv index in `pyproject.toml`,
  a `tools/` dir). Surface candidates via `AskUserQuestion` with a free-text "other path" option. If
  nothing is found, ask directly for a path or name.
- Never hardcode a specific repo name, DockerHub account, or cluster path anywhere in this skill's own
  files — those are always confirmed per-invocation (see Before You Start), not skill defaults. This is
  what keeps the skill portable across users/teams, not just across target repos for one team.

## Before You Start

Run this checklist via `AskUserQuestion` before touching any file in the target repo. Getting these wrong
costs real rework given how much of the workflow depends on them.

- **Target repo + reference repo** — resolved per the invocation rule above.
- **Python/torch version**: match the reference repo exactly, or independent? Affects `pyproject.toml`
  `requires-python`, `torch==` pin, and which `pytorch-cuXXX` uv index to use.
- **Fix framework/precision bugs now, or defer to a follow-up pass?**
- **Deploy naming**: image name, DockerHub account, Slurm job names, cluster paths, GPU count — confirm
  real values, never guess.
- **Code layout**: `tools/`-wrapper-only around existing scripts, or full installable-package
  restructuring (`<pkg>/{data,modeling,engine,utils}/`)? Only do the latter if justified/requested — flat
  scripts with a thin `tools/` wrapper are often sufficient.
- **Config system**: preserve as-is (default), or migrate to Hydra? Only propose migration if the source
  system is genuinely ad hoc/fragile (e.g. a hand-rolled INI parser with a manual CLI-override precedence
  helper) **and** the user explicitly agrees. Never migrate a working yacs/OmegaConf/argparse system just
  because Hydra exists. See `references/config-systems.md`.
- **Training loop: port the custom/hand-rolled loop to PyTorch Lightning, or keep it as-is?** Every known
  reference repo (`tinycar-dev`, `gaussiancar`, `gcarpred-dev`, `fiery-radar`, `powerbev-radar`) is
  actually Lightning-based. Porting is what gets full parity — real multi-logger fan-out
  (`WandbLogger`+`CSVLogger`), automatic rank-zero gating, `ModelCheckpoint`, DDP setup — for free, at
  real migration cost/risk. Keeping the hand-rolled loop is lighter-weight but permanently diverges from
  every sibling repo and needs a compensating logging facade built by hand. **This is the single biggest
  scope/risk decision in the whole workflow — don't default either way, ask explicitly and state the
  tradeoff.** See `references/lightning-porting.md` vs `references/logging-facade.md`.
- **Dataset env-var convention**: which `*_DATA_ROOT`/`PATH_TO_*` name(s) does this target need? Reuse the
  reference repo's existing names where the dataset matches; mint a new one otherwise.
- **Logging**: framework-provided already (or after a Lightning port), or needs the hand-rolled facade?
  Is W&B wanted, and is `$WANDB_API_KEY` available?
- **Commit at the end?** Default: **no**. Only commit if explicitly asked.
- **Git safety**: confirm the working tree is clean (or get explicit acknowledgment that it isn't);
  suggest a dedicated branch, since edits here are extensive and there's no test suite to catch mistakes.
- **Baseline metrics to verify against**: does a documented reference number already exist (README, paper,
  a prior run's logs)? If not, this must be obtained — ask the user, or ask the original repo's author if
  reachable — rather than eyeballing whether post-adaptation numbers "look reasonable." Line this up now;
  it's the actual acceptance bar for the Verify phase, not an afterthought.

## Workflow

1. **Scope & Confirm** — the Before You Start checklist above, via `AskUserQuestion`. Do not proceed until
   answered.

2. **Explore** — read the reference repo's actual current `pyproject.toml`, `deploy/`, `Makefile`,
   `tools/`, and config system. Read the target repo's actual current dependency file(s), entrypoints, and
   config system. Check any other sibling repos for how they diverge from the reference repo and why,
   rather than assuming one canonical pattern applies unmodified. Detect whether the target repo is
   *already partially adapted* (an existing `pyproject.toml`/`deploy/`/`tools/`) and treat that as a
   resume, not a redo.

3. **Plan** — synthesize findings into a concrete, numbered adaptation plan (what infra gets
   copied/renamed, whether code restructuring happens and in what dependency-ordered sequence, whether
   config/training-loop migrate, what env vars are used, what logging approach). Present it for sign-off
   before writing any files. Track it with `TodoWrite` — this workflow spans a long session and must be
   resumable.

4. **Execute — Infra** → `references/infra-checklist.md`, plus `references/package-restructuring.md` if
   code layout was scoped in, plus `references/dependency-resolution.md`.

5. **Execute — Code** → `references/framework-migration.md` for framework/API-bump fixes,
   `references/config-systems.md` if migrating config, then either `references/lightning-porting.md` (if
   porting the training loop — the reference-repo-parity path) **or** `references/logging-facade.md` (if
   keeping the hand-rolled loop; start from `assets/run_logger_template.py`). These last two are
   mutually exclusive alternatives for the same problem — don't apply both.

6. **Verify & Wrap-up** → `references/verification.md`. **The bar is "the adaptation is correct," not
   just "the code runs":** a full one epoch on a real dataset (`num_workers` capped at 2 to avoid OOM —
   see below), with the resulting metrics compared against the known baseline lined up in step 1. A clean
   one-epoch run with no baseline comparison is necessary but not sufficient. Then update
   README/CLAUDE.md/`.gitignore`, delete superseded dependency files, and commit only if explicitly
   requested.

## Non-negotiables

- Never guess deploy naming, cluster paths, GPU counts, or CUDA/driver versions — confirm with the user,
  or verify directly (`nvidia-smi`).
- Commit `uv.lock` deliberately for any repo meant to be retrained, even if the reference repo gitignores
  it — state this as an intentional, explained deviation (e.g. in the README), not a silent inconsistency.
- Preserve load-bearing logic byte-for-byte when moving code, and re-verify it behaviorally (a real
  forward pass asserting the contract holds), not just by visual inspection.
- Only commit if explicitly asked.
- Before running any repo-scaffolding CLI tool with untested flags, verify its actual argument-parsing
  behavior first (read its source, or test in a scratch directory). Always run `git status` after any
  exploratory tool invocation to catch surprise untracked output.
- Cap `num_workers` at 2 by default during verification runs regardless of what the source repo's config
  specifies — an in-memory dataset index duplicated across forked dataloader workers via copy-on-write can
  exhaust memory fast on a modest-RAM host; check `free -h` before raising it.

## Reference index

- `references/infra-checklist.md` — pyproject/uv, Docker, Slurm, Makefile, `tools/`, dataset env-var
  convention, curated-checkpoint folder, sibling-repo divergence.
- `references/package-restructuring.md` — flat scripts → installable package, dependency-ordered moves,
  byte-for-byte preservation, comment translation.
- `references/dependency-resolution.md` — uv resolver conflicts, Python-version wheel gaps, CUDA
  index/driver matching.
- `references/framework-migration.md` — Lightning-style API-bump checklist, precision fixes,
  dependency-drift code gotchas.
- `references/config-systems.md` — preserve-vs-migrate decision, Hydra migration specifics, the
  DDP+Hydra chdir bug.
- `references/lightning-porting.md` — porting a hand-rolled `torch.distributed` loop to a
  `LightningModule`/`Trainer` (the reference-repo-parity path).
- `references/logging-facade.md` — hand-rolled metrics/W&B logging facade (the fallback path when not
  porting to Lightning); starting point: `assets/run_logger_template.py`.
- `references/verification.md` — staged, no-test-suite verification discipline and the one-epoch +
  baseline acceptance bar.
