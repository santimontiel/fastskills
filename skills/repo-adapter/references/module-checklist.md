# Module checklist (Systematic Pass)

Walk this module-by-module against the target repo before doing any nuance-hunting. For each item, decide:
**present** (matches the reference shape, nothing to do), **absent** (doesn't exist yet — plan to add it
if in scope), or **needs-adapting** (exists but in a divergent form — plan the change). Don't skip an item
just because the repo "looks modern" — check it explicitly. This is the systematic half of adaptation;
`references/case-studies.md` is where the nuance half (things that don't fit this checklist) gets logged.

## 1. Package layout & entrypoints

- [ ] Is there an installable package (`<pkg>/` with a matching `pyproject.toml` `packages.find`), or flat
  scripts at repo root? Determines whether `references/package-restructuring.md` applies.
- [ ] `tools/` — thin CLI entrypoints (one per task), or is CLI logic tangled into the scripts themselves?
- [ ] `dev/` (or equivalent) — any dependency-light, framework-free analysis tooling, separate from the
  main package's heavier dependencies?

## 2. Model / module split (only if Lightning-porting is in scope)

- [ ] Is model forward-wiring (`model.py`-equivalent) separated from the training loop
  (`module.py`-equivalent / `LightningModule`), or are they the same file/class?
- [ ] Are interchangeable architectural blocks (encoders, decoders, heads) grouped into a `components/`-style
  subpackage with clean `__init__.py` re-exports, or scattered across the codebase?

## 3. Config system

- [ ] What's the current config mechanism — Hydra, yacs, OmegaConf-raw, argparse, hand-rolled INI? See
  `references/config-systems.md`'s preserve-vs-migrate decision before touching this.
- [ ] If Hydra: are `data`/`module`/`task` composed as separate groups with `_target_`+`instantiate`, or is
  it a flatter/monolithic config file?
- [ ] Any derived config values that should be OmegaConf-interpolated (e.g. a run/project name built from
  dataset+task) rather than duplicated by hand?

## 4. Dependency management & vendored-extension isolation

- [ ] `uv`-based (`pyproject.toml`+`uv.lock`), or an older mechanism (`conda`/`environment.yml`,
  `requirements.txt`, bare `setup.py`)? See `references/dependency-resolution.md`.
- [ ] Any vendored/native CUDA extensions? If so, are they isolated in their own subtree with their own
  nested build config and preserved attribution, or mixed into the main package?
- [ ] Any build-time-only dependencies (e.g. `flash-attn` needing `torch` present at its own build time)
  that need `[tool.uv.extra-build-dependencies]`?

## 5. Docker / Slurm deploy

- [ ] `Makefile` with `build`/`run`/`attach`/`clear` targets, or no containerized dev workflow at all?
- [ ] `deploy/docker/{Dockerfile,entrypoint.sh}` and `deploy/slurm/*.sh` present, and do they reflect this
  target's actual dataset mounts / GPU count / cluster paths (not copy-pasted from the reference repo
  unmodified)?
- [ ] Does the checkout name carry a `-dev` suffix? If so, is it correctly trimmed from Docker image name,
  container name, package name, and Slurm job names?
- [ ] Is the `faststart` CLI available and current enough to generate this layer directly, rather than
  hand-authoring it? (See `references/infra-checklist.md`.)

## 6. Dataset env-var convention

- [ ] Does the target already use an external `*_DATA_ROOT`/`PATH_TO_*` env var, or point at a repo-local
  `data/` directory that needs migrating?
- [ ] Does an existing convention name fit this target's dataset, or does a new one need minting?

## 7. Logging & output-folder convention

- [ ] Framework-provided (Lightning `WandbLogger`/`CSVLogger`) or hand-rolled? See
  `references/logging-facade.md` vs `references/lightning-porting.md`.
- [ ] Does every run write into one unified, run-id-scoped output folder (config snapshot + logs +
  checkpoints + csv/wandb subdirs together), or is output scattered/split across multiple legacy locations?
- [ ] Is there a separate, human-curated `checkpoints/` folder distinct from the automatic per-run output
  directory?

## 8. Verification tooling

- [ ] Any existing test suite? (Expected answer in this ecosystem: no.) If one exists, understand why
  before assuming it should be removed or ignored.
- [ ] Are there `debug_*.py`-style sanity scripts, or is there no equivalent verification tooling at all?
- [ ] Is there a dependency-light, framework-free post-hoc run analyzer, or does inspecting a past run
  require re-importing the whole training stack?

## 9. Documentation

- [ ] Does `CLAUDE.md` exist and actually describe the architecture + known gotchas, or is it missing/stale?
- [ ] Does `README.md` document the current command surface (`uv run tools/...`, `make build`/`make run`),
  or does it still reference an old invocation style (conda activate, bare `python`)?

## 10. Commit convention

- [ ] Gitmoji-prefixed conventional commits, or a different/no convention? (Lower priority — note it, don't
  force a rewrite of existing history.)
