# Infra checklist

The pieces that make a repo buildable/deployable the same way as the reference repo. All of this assumes
Before You Start has already confirmed: which reference repo, deploy naming, Python/torch version, and
whether code restructuring is in scope.

## Recognizing an unadapted baseline

Typical starting state: pinned to an old Python/torch/framework version via `environment.yml`,
`requirements.txt`, or `setup.py`; no `Dockerfile`; no Slurm scripts; no `tools/` entrypoint layout; no
CI. Not every one of these needs to be true to proceed — adapt what's actually missing.

## `pyproject.toml` + `uv.lock`

- Match the reference repo's `requires-python` and `torch==` pin, and its custom `pytorch-cuXXX` uv index,
  **only if** the target will share the same cluster/GPU driver. Verify the actual driver via `nvidia-smi`
  — don't assume compatibility. If independent, pick current-stable versions deliberately (see
  `dependency-resolution.md`).
- **Commit `uv.lock` deliberately** for any repo meant to be retrained/rerun later, even if the reference
  repo itself gitignores its lockfile — this is an intentional, explained deviation for reproducibility of
  a baseline you intend to reuse, not an oversight. State the reason in the README.
- Prune vendored-CUDA-extension `pyproject.toml` machinery from copied config unless the target genuinely
  has one: `[tool.uv.sources]` local-path sources, `[tool.uv.extra-build-dependencies]` (used for things
  like `flash-attn` needing `torch` present at its own build time), `[[tool.uv.index]]` blocks beyond the
  main `pytorch-cuXXX` one, `[tool.uv] find-links` pointing at a wheel index (e.g. PyG). These appear in
  some reference repos (`gaussiancar`, `gcarpred-dev`) and not others (`tinycar-dev`) for a concrete reason
  — a vendored CUDA op — not because "that's the pattern."
- Delete the old dependency file (`environment.yml`/`requirements.txt`/`setup.py`) only once `uv sync`
  fully replaces it and has been verified to work.

## `deploy/docker/{Dockerfile,entrypoint.sh}` and `deploy/slurm/*.sh`

Copy near-verbatim from the reference repo's **actual current files** (read them fresh, don't rely on
memory), then rename per confirmed values from Before You Start:
- `REPO_NAME`/`IMAGE_NAME` throughout `Makefile`, `Dockerfile` build args, and every `deploy/slurm/*.sh`.
- DockerHub account in `build_and_push.sh`.
- Slurm job names, partition names, GPU counts, cluster paths (`/raid/${USER}/...`-style) in every
  `deploy/slurm/*.sh`.
- `figlet` banner text and any other cosmetic repo-name strings in `entrypoint.sh`.

Reference repos have differing script sets — some have `deploy/slurm/debug_terminal.sh`, some don't; some
Makefiles have a `jupyter` target, some don't. Only include what's actually useful for the target, but
default to keeping whatever the chosen reference repo has unless there's a reason to drop it.

## `Makefile`

Same `build`/`run`/`attach`/`clear` target pattern, same `check-env`/`run_docker` structure as the
reference repo. Two things that are easy to get wrong when adapting:
- **Dataset mount count is repo-specific.** `tinycar-dev` mounts one dataset (`NUSCENES_DATA_ROOT`),
  `gaussiancar` mounts two, `gcarpred-dev` mounts three. Base the mount list on what the target repo's
  code actually needs, not on how many the reference repo happens to mount.
- **The Makefile mounts the host env var's *value* to a fixed container path, but does not forward the
  env var *itself* into the container** (i.e. `-v $(NUSCENES_DATA_ROOT):/data/nuscenes` with no matching
  `-e NUSCENES_DATA_ROOT=...`). This is load-bearing for the Hydra `${oc.env:...}` fallback pattern
  described in `config-systems.md` — inside the container, code must resolve the fixed mount path
  (`/data/nuscenes`), not expect the env var to still be set.

## `tools/` entrypoints

Move (or create thin wrappers around) entry-point scripts into `tools/`. Each one starts with:

```python
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
```

before any other project imports, so the script works regardless of the cwd it's launched from. **Then
re-anchor every cwd-relative path in the moved code** — checkpoint paths, output directories, example data
paths, anything written as `./foo` or `foo/bar` relative to the *old* script location. These break
silently (or write to the wrong place) once the script moves a directory deeper into `tools/`, and won't
necessarily throw an error — verify explicitly, don't just move and assume.

## Dataset env-var convention

Point dataset root paths at external env vars, not a repo-local `data/` directory — matches every
reference repo's convention. Common existing names to check for and reuse where applicable:
`NUSCENES_DATA_ROOT`, `ZJU_DATA_ROOT`, `PATH_TO_TRUCKSCENES`, `PATH_TO_VOD`. Mint a new
`<DATASET>_DATA_ROOT`-style name only if none of the existing ones fit — check sibling repos first so
naming stays consistent across the team's repos, not just within this one.

## Curated checkpoint folder

Create/preserve a top-level `checkpoints/` directory, **separate from** the training loop's own automatic
per-run output directory (e.g. `train_log/models/` or `outputs/<run>/checkpoints/`, whichever the
Verify-phase training run writes to). This is a real, confirmed convention (`gcarpred-dev/checkpoints/`):
a human manually copies checkpoints worth keeping there after reviewing results. It is never
auto-populated by the training loop itself — don't wire any code to write into it automatically.

## Config system

Default: leave the existing config system exactly as it is, only bumping its own dependency version if
needed for the Python/framework bump. See `config-systems.md` for the migration decision criteria — this
is opt-in, not part of the default infra checklist.

## Docs

Update `CLAUDE.md`/`README.md` to reflect the new command surface (`uv run tools/...` instead of the old
invocation style, `make build`/`make run` instead of conda/pip setup instructions, new env var names).
Update `.gitignore` for `.venv/`, `*.egg-info/`, and whatever the training loop's new output directory is
— but don't gitignore `uv.lock` (see above).
