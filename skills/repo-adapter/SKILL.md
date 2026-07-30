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
`JustDepth`) — this skill generalizes across all of it.

**Treat whichever reference repo is chosen for a given invocation as the live source of truth.** Read its
actual current files every time; never assume "the infra pattern" is a fixed, memorized template.

**Repo lineage** (so this skill's own history is legible, not a flat list of names to memorize):
- `tinycar-dev` — the current flagship/canonical implementation of this philosophy. It's the reference
  repo to read first when in doubt about what "the shape" actually looks like right now, and the
  `references/*.md` files in this skill are grounded in it (see "Reference architecture" below).
- `gaussiancar`, `gcarpred-dev` — tinycar-dev's heavier research *precursors*, not adaptation-target
  examples themselves. `gaussiancar` is the one **public** repo in this lineage
  (github.com/santimontiel/gaussiancar); its infra intentionally lags the current template because it's
  a stable, already-published repo that predates later infra fixes. Useful for understanding *why* a
  pattern exists, not for what to copy today.
- `fiery-radar`, `powerbev-radar`, `JustDepth` — repos already migrated to this philosophy. Each has real
  particularities (dataset mount counts, an extra Makefile target, presence/absence of a vendored CUDA
  extension) — these are justified divergence, not drift to correct.

Sibling repos following this infra are **not** byte-identical to tinycar-dev — dataset mount counts, a
`jupyter` Makefile target, `deploy/slurm/debug_terminal.sh`, and vendored-CUDA-extension `pyproject.toml`
machinery all vary per-repo for justified reasons. Copying tinycar-dev's files wholesale into a target
that doesn't need them (e.g. CUDA-extension build machinery for a repo with no vendored extension) is a
mistake, not diligence.

## Reference architecture: the tinycar-dev shape

This is the concrete shape to check a target repo against — not a template to copy wholesale (see the
divergence caveat above). Read `references/module-checklist.md` for the systematic per-module walkthrough;
this section is the narrative version.

- **Package layout**: `<pkg>/{data/, modeling/{model.py, module.py, components/}, ops/, render.py,
  losses.py, metrics.py, utils/}`. `model.py` is pure `nn.Module` forward wiring; `module.py` is the
  `LightningModule` (train/val loop, loss computation, metric logging, optimizer/scheduler config).
  `components/` holds interchangeable named blocks, often as `*_encoders/`-style subpackages selected via
  Hydra, each with an `__init__.py` re-exporting its public classes.
- **`ops/`** is explicitly for vendored/adapted third-party code (e.g. a CUDA rasterizer, a point-transformer
  implementation) — kept isolated in its own subtree, changes kept minimal, attribution preserved. It may
  have its own nested `pyproject.toml`/build config when it's a native extension installed as a local-path
  dependency.
- **`tools/`** holds thin `@hydra.main` CLI entrypoints (one script per task: create-data, train, eval,
  inference, benchmark) plus ad-hoc `debug_*.py` sanity scripts. `dev/` (distinct from `tools/`) is for
  dependency-light, framework-free analysis utilities — see Verification below.
- **Config**: Hydra composing `data`/`module`/`task` groups; every leaf uses `_target_` +
  `hydra.utils.instantiate` (grep class names to find implementations — there's no central registry); heavy
  OmegaConf interpolation for derived values (e.g. a wandb project name interpolated from dataset + task).
- **Dependencies**: `pyproject.toml` + `uv.lock`. `[tool.uv.sources]` for pinned custom wheel indices (a
  CUDA-specific torch index) and local-path deps for vendored native extensions;
  `[tool.uv.extra-build-dependencies]` for build-time-only deps (e.g. `flash-attn` needing `torch` present
  at its own build time); `[dependency-groups] dev` for dev-only extras. No lint/type-check config.
- **Deploy**: Docker-mandatory dev workflow — a `Makefile` wraps `build`/`run`/`attach`/`clear`; native CUDA
  ops need the container's toolchain, so `uv run`/`python` is never invoked on the host. `deploy/slurm/`
  holds cluster job scripts assuming a pushed image and cluster-specific paths. **Naming convention**: a
  repo's on-disk/checkout name may carry a `-dev` suffix (e.g. `tinycar-dev`), but the Docker image name,
  container name, Python package name, and Slurm job names all have that suffix trimmed — don't propagate
  `-dev` into deploy artifact names.
- **Scaffolding shortcut**: the user has a `faststart` CLI (`faststart <ProjectName> [dest]`) that
  generates the Docker/Slurm/Makefile/`pyproject.toml` skeleton above via placeholder substitution, kept in
  sync with tinycar-dev's own infra fixes. Prefer running it (or copying its template, normally installed at
  `~/.local/share/faststart/template`) over hand-authoring this boilerplate from scratch — but always verify
  it's current first (`faststart --update`, or diff the installed template against the source repo) since
  the locally installed copy can silently drift out of sync. It only covers the infra layer — it does not
  ship `configs/`, a package skeleton, or Lightning module code, so the config/package/Lightning layer below
  is still grounded directly in the actual reference repo's code, not in faststart's template.
- **Logging**: framework-provided (Lightning `WandbLogger`/`CSVLogger`), writing into a unified per-run
  output folder — config snapshot, train log, checkpoints, and csv/wandb subdirs all under one
  run-id-scoped directory. An older convention (checkpoints-only, split by project/run-id, no config
  snapshot) may still exist as legacy/orphaned output in a given repo — the unified-folder convention is the
  one to replicate going forward, not the legacy one.
- **Verification**: **no formal test suite by design.** Correctness comes from running real
  train/eval/inference scripts on real or debug data, plus standalone `debug_*.py` sanity scripts and a
  dependency-light, framework-free post-hoc run analyzer (reads a run's config snapshot + logs +
  checkpoints + metrics, no `torch` import required) — replicate this pattern in every ported repo rather
  than trying to retrofit a pytest suite that doesn't match how this ecosystem actually verifies itself.
- **Documentation**: a living `CLAUDE.md` architecture doc — a code-architecture map, a forward-pass
  walkthrough, and recorded historical gotchas (e.g. a precision bug that silently zeroed out a branch for
  a full training run — see `references/case-studies.md`). Porting a repo should include writing or porting
  an equivalent doc, not just updating command syntax.
- **Commits**: gitmoji-prefixed conventional messages.

## Principles

- **`uv` runs everything.** Every command in a ported repo — training, eval, debug scripts, the analyzer —
  goes through `uv run`, never a bare `python` invoked against a manually-activated venv.
- **Pythonic and KISS by default — but numerical fidelity to the prior version wins when the two conflict.**
  Don't refactor load-bearing logic into something more idiomatic if doing so risks changing its numeric
  output. This is not just a style preference: it's tied directly to the "preserve load-bearing logic
  byte-for-byte" non-negotiable below and to `references/verification.md`'s baseline-comparison acceptance
  gate — a port that reads cleaner but produces different numbers than the original has failed, regardless
  of how the code looks.
- **Every repo has real particularities.** The reference architecture above is the shape to check against,
  not a template to stamp out unmodified — see the Workflow's Nuance Pass below.

## Invocation

`/repo-adapter [target-repo-path] [reference-repo-path-or-name]` — both arguments optional.

- `target-repo-path` defaults to the current working directory.
- `reference-repo-path-or-name`: if given, resolve and read it directly. If omitted, **do not guess**.
  Glob the target repo's parent directory (and any other workspace roots already known from context) for
  sibling repos with infra fingerprints (`deploy/` dir, a `pytorch-cuXXX` uv index in `pyproject.toml`,
  a `tools/` dir). Surface candidates via `AskUserQuestion` with a free-text "other path" option. If
  nothing is found, ask directly for a path or name. Default to `tinycar-dev` as the reference repo when
  the user hasn't named one and it's available, since it's the current canonical shape.
- Never hardcode a specific repo name, DockerHub account, or cluster path anywhere in this skill's own
  files beyond the lineage context above — deploy-specific values are always confirmed per-invocation (see
  Before You Start), not skill defaults. This is what keeps the skill portable across users/teams, not
  just across target repos for one team.

## Before You Start

Run this checklist via `AskUserQuestion` before touching any file in the target repo. Getting these wrong
costs real rework given how much of the workflow depends on them.

- **Target repo + reference repo** — resolved per the invocation rule above.
- **Python/torch version**: match the reference repo exactly, or independent? Affects `pyproject.toml`
  `requires-python`, `torch==` pin, and which `pytorch-cuXXX` uv index to use.
- **Fix framework/precision bugs now, or defer to a follow-up pass?**
- **Deploy naming**: image name, DockerHub account, Slurm job names, cluster paths, GPU count — confirm
  real values, never guess. If the target repo's checkout name carries a `-dev` suffix, confirm whether
  deploy artifact names should trim it per the naming convention above.
- **Code layout**: `tools/`-wrapper-only around existing scripts, or full installable-package
  restructuring (`<pkg>/{data,modeling,engine,utils}/`)? Only do the latter if justified/requested — flat
  scripts with a thin `tools/` wrapper are often sufficient.
- **Config system**: preserve as-is (default), or migrate to Hydra? Only propose migration if the source
  system is genuinely ad hoc/fragile (e.g. a hand-rolled INI parser with a manual CLI-override precedence
  helper) **and** the user explicitly agrees. Never migrate a working yacs/OmegaConf/argparse system just
  because Hydra exists — this follows directly from the fidelity-over-idiom principle above. See
  `references/config-systems.md`.
- **Training loop: port the custom/hand-rolled loop to PyTorch Lightning, or keep it as-is?** tinycar-dev
  and every migrated sibling repo (`fiery-radar`, `powerbev-radar`, `JustDepth`) are Lightning-based. Porting
  is what gets full parity — real multi-logger fan-out (`WandbLogger`+`CSVLogger`), automatic rank-zero
  gating, `ModelCheckpoint`, DDP setup — for free, at real migration cost/risk. Keeping the hand-rolled loop
  is lighter-weight but permanently diverges from the reference shape and needs a compensating logging
  facade built by hand. **This is the single biggest scope/risk decision in the whole workflow — don't
  default either way, ask explicitly and state the tradeoff.** See `references/lightning-porting.md` vs
  `references/logging-facade.md`.
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

2. **Systematic Pass** — walk `references/module-checklist.md` module-by-module against the target repo:
   package layout, config system, dependency management, deploy, logging, verification tooling,
   documentation, commit convention — present / absent / needs-adapting for each. Also skim
   `references/case-studies.md` for nuances a previous invocation already ran into on a similar repo; don't
   rediscover the same gotcha twice.

3. **Nuance Pass** — explore what's specific to *this* target repo and doesn't fit the checklist: unusual
   data formats, custom losses, an already-partially-adapted state, oddities in its dependency tree,
   anything the reference repo doesn't have an equivalent of. Detect whether the target is *already
   partially adapted* (an existing `pyproject.toml`/`deploy/`/`tools/`) and treat that as a resume, not a
   redo. This is where "every repo has real particularities" gets captured explicitly, feeding directly
   into the Plan step and into the Self-Improve entry at the end.

4. **Plan** — synthesize the Systematic and Nuance passes into a concrete, numbered adaptation plan (what
   infra gets copied/renamed, whether code restructuring happens and in what dependency-ordered sequence,
   whether config/training-loop migrate, what env vars are used, what logging approach). Present it for
   sign-off before writing any files. Track it with `TodoWrite` — this workflow spans a long session and
   must be resumable.

5. **Execute — Infra** → `references/infra-checklist.md`, plus `references/package-restructuring.md` if
   code layout was scoped in, plus `references/dependency-resolution.md`.

6. **Execute — Code** → `references/framework-migration.md` for framework/API-bump fixes,
   `references/config-systems.md` if migrating config, then either `references/lightning-porting.md` (if
   porting the training loop — the reference-repo-parity path) **or** `references/logging-facade.md` (if
   keeping the hand-rolled loop; start from `assets/run_logger_template.py`). These last two are
   mutually exclusive alternatives for the same problem — don't apply both.

7. **Verify & Wrap-up** → `references/verification.md`. **The bar is "the adaptation is correct," not
   just "the code runs":** a full one epoch on a real dataset (`num_workers` capped at 2 to avoid OOM —
   see below), with the resulting metrics compared against the known baseline lined up in step 1. A clean
   one-epoch run with no baseline comparison is necessary but not sufficient. Then update
   README/CLAUDE.md/`.gitignore`, delete superseded dependency files, and commit only if explicitly
   requested.

8. **Self-Improve** — append a dated entry to `references/case-studies.md`: target repo, what was ported,
   what deviated from the tinycar-dev shape and why, any fidelity-vs-Pythonic tradeoffs made and why, new
   gotchas/nuances discovered, follow-ups worth checking on the next invocation. This is what makes the
   Systematic/Nuance passes on the *next* invocation start smarter instead of rediscovering the same
   surprises — don't skip it just because the port itself is done.

## Non-negotiables

- Never guess deploy naming, cluster paths, GPU counts, or CUDA/driver versions — confirm with the user,
  or verify directly (`nvidia-smi`).
- Commit `uv.lock` deliberately for any repo meant to be retrained, even if the reference repo gitignores
  it — state this as an intentional, explained deviation (e.g. in the README), not a silent inconsistency.
- **Preserve load-bearing logic byte-for-byte** when moving code, and re-verify it behaviorally (a real
  forward pass asserting the contract holds), not just by visual inspection. This is the fidelity-over-idiom
  principle in its most concrete, checkable form — see Principles above.
- Only commit if explicitly asked.
- Before running any repo-scaffolding CLI tool with untested flags (including `faststart`), verify its
  actual argument-parsing behavior first (read its source, or test in a scratch directory). Always run
  `git status` after any exploratory tool invocation to catch surprise untracked output.
- Cap `num_workers` at 2 by default during verification runs regardless of what the source repo's config
  specifies — an in-memory dataset index duplicated across forked dataloader workers via copy-on-write can
  exhaust memory fast on a modest-RAM host; check `free -h` before raising it.

## Reference index

- `references/module-checklist.md` — the systematic, module-by-module checklist the Systematic Pass walks
  through: package layout, config, dependencies, deploy, logging, verification, docs, commit convention.
- `references/case-studies.md` — the growing, append-only log of nuances and gotchas discovered on real
  invocations; read at the start of the Systematic Pass, appended to at Self-Improve.
- `references/infra-checklist.md` — pyproject/uv, Docker, Slurm, Makefile, `tools/`, dataset env-var
  convention, curated-checkpoint folder, sibling-repo divergence, the `faststart` scaffolding shortcut.
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
