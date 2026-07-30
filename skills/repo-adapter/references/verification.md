# Verification

No test suite exists anywhere in this ecosystem, by design — not an oversight to fix. tinycar-dev itself
has no `tests/` directory and no lint/CI config; correctness comes entirely from running real
train/eval/inference scripts on real or debug data, plus two lighter-weight tools worth replicating in
every ported repo rather than retrofitting a pytest suite that doesn't match how this ecosystem actually
verifies itself:
- **`debug_*.py` sanity scripts** — small, standalone scripts that exercise one specific piece in isolation
  (a backbone's output shape, a normalization-layer mismatch, an encoder swap) without running a full
  training job.
- **A dependency-light, framework-free post-hoc run analyzer** — a tool that inspects a completed run's
  config snapshot, logs, checkpoints, and metrics history without importing `torch` or the training stack
  at all, so it stays fast and has no risk of dependency conflicts with the environment being analyzed.

Every adaptation in this family (`fiery-radar`, `powerbev-radar`, `JustDepth`, and the hand-rolled-loop case
this skill originally generalized from) has been verified this way — staged from cheap/fast checks up to
the real thing. "Ran once, no crash" is not the bar — see the acceptance test below.

## Staged checklist

1. **Import smoke test after every single file move** during restructuring (see
   `package-restructuring.md`), not just once at the end — this is what makes a broken import
   attributable to exactly the one change just made.
2. **`uv sync` + import smoke test** once infra setup (pyproject.toml, package structure) is in place.
3. **Hydra dry-run composition checks** for every meaningful override axis (every dataset variant, every
   module/task combination), if config was migrated:
   ```bash
   uv run python tools/train.py --cfg job --resolve data=<variant>
   ```
4. **Synthetic-tensor forward pass** — instantiate the model with dummy-shaped tensors, assert output
   shape *and* any load-bearing behavioral contract identified during restructuring (e.g. a training-only
   submodule actually gets swapped out in eval mode — see `package-restructuring.md`'s worked example).
5. **The real acceptance test: a full one epoch on a real dataset, not just a few iterations.** A handful
   of steps confirms the code path executes; it does not confirm the adaptation preserved correctness.
   Run a complete epoch.
   - **Cap `num_workers` at 2 by default**, regardless of what the source repo's own config specifies.
     Concrete lesson: an in-memory dataset index alone consumed ~8.8GB in the main process before any
     dataloader worker even spawned; forked workers duplicating chunks of that via copy-on-write scales
     memory use disastrously with worker count on a modest-RAM host. Check `free -h` — both before
     starting and while the run is going — before considering a higher worker count, and don't just reuse
     whatever number the original repo's config happened to use.
   - Background the run (`nohup ... & disown`, or the environment's background-process tooling) and poll
     the log periodically — don't block the session on a run that can take tens of minutes.
   - If polling via a log-tailing mechanism that fires a notification per matched line, use a **tight**
     filter (checkpoint-saved / run-completed / error/OOM/traceback signatures only) once the run is
     confirmed healthy from the first few log lines — a filter that fires on every logged training step
     over a long run will flood with low-value notifications.
6. **Check log scalars for NaN/Inf explicitly across the full epoch** — don't trust "the progress bar kept
   moving" as a proxy for correctness. NaN doesn't necessarily crash training; it can silently corrupt the
   checkpoint while the loop keeps running and printing plausible-looking (but garbage) numbers.
7. **Checkpoint save → reload → real eval run against that checkpoint** — closes the full
   train→checkpoint→eval loop with real numbers, not just "the training script ran."
8. **Compare the resulting metrics against a known baseline — this is the fidelity gate, not an optional
   nice-to-have.** This is the actual acceptance bar, not step 5 or 6 or 7 in isolation — a clean one-epoch
   run with sane-looking, non-NaN numbers is *necessary but not sufficient*. The bar is matching order of
   magnitude (or the expected directional trend after one epoch, for metrics that need full training to
   converge) against a real reference number. If no baseline was already lined up during Before You Start,
   get one now from the user or the original repo's author before declaring the adaptation verified — don't
   substitute "the numbers look plausible" for an actual comparison point. This is where `SKILL.md`'s
   fidelity-over-idiom principle actually gets enforced: a port that reads cleaner than the original but
   fails this comparison has not succeeded, no matter how the code looks.

## Tooling caution

Before running any repo-scaffolding CLI tool with untested flags against a real target repo (including
`faststart` — see `infra-checklist.md`), verify its actual argument-parsing behavior first — read its
source, or test it in a scratch directory. Concrete cited failure mode from a real session: a scaffolding
tool didn't recognize `--help`/`-h` as real flags at all, and instead silently treated them as a literal
project-name argument, scaffolding two full template directories directly into the target repo's root.
Always run `git status` after any exploratory tool invocation to catch this kind of surprise untracked
output before it's mistaken for something intentional.

## Log what you find

Any gotcha discovered during verification that isn't already covered by these reference docs — a new
numeric-instability pattern, a dataset-specific memory issue, a scaffolding-tool surprise — belongs in
`references/case-studies.md` as part of `SKILL.md`'s Self-Improve workflow step, not just in the
conversation it was found in. That's the difference between this skill staying static and it actually
getting better with each real adaptation.
