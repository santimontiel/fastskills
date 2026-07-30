# Package restructuring

Only do this when it was explicitly scoped in during Before You Start — a `tools/` wrapper around flat
scripts is often sufficient and lower-risk. Full restructuring is justified when the target repo's flat
layout is itself an obstacle (e.g. scripts import each other via relative paths that break once anything
moves, or the repo needs to become pip-installable as a dependency of something else).

## Target shape

Flat scripts at repo root (`train.py`, `model.py`, `dataset.py`, `utils.py`, ...) become an installable
package with a conventional split. This mirrors tinycar-dev's actual layout — treat it as the shape to
check against, adapting names/subpackages to what the target repo actually needs rather than stamping it
out unmodified:

```
<pkg>/
  data/               dataset classes, augmentation, shared constants, DataModule
  modeling/
    model.py           pure nn.Module forward wiring — no training-loop concerns
    module.py           LightningModule: train/val step, loss computation, metric logging,
                        optimizer/scheduler config (only if Lightning-porting is in scope —
                        see lightning-porting.md)
    components/
      *_encoders/        interchangeable backbone implementations, one subpackage per swappable
                        role (e.g. image_encoders/, radar_encoders/), each with an __init__.py
                        re-exporting its public classes so Hydra's _target_ paths stay short
      <block>.py          other named architectural blocks (fuser, decoder, heads, ...) — one
                        file per block, named after the role it plays in the forward pass
  ops/                  vendored/adapted third-party code (a CUDA kernel, a point-transformer
                        implementation) — see "Vendored code" below; NOT for original code
  render.py / losses.py / metrics.py   top-level concerns that don't belong inside modeling/
  utils/                geometry/math/config/misc helpers with no internal dependencies
tools/
  train.py              thin @hydra.main wrapper: compose config, instantiate, call trainer.fit
  eval.py               thin wrapper: same pattern for evaluation
  debug_<x>.py           ad-hoc sanity-check scripts (no test framework — see verification.md)
dev/
  analyze_run.py         dependency-light, framework-free post-hoc run analysis (no torch import)
```

`tools/*.py` should be thin — Hydra config composition plus one call into `<pkg>`. All real logic lives in
the package so it's importable and (informally, since there's no test suite) independently exercisable
from the CLI entrypoint. Note the `tools/` vs `dev/` distinction: `tools/` scripts assume the full
training stack is importable (they run inside the Docker container); `dev/` scripts are deliberately kept
dependency-light so they can inspect a run's config/logs/checkpoints without importing `torch` at all.

## Vendored code

Anything adapted or copied from a third party (a CUDA rasterizer, an external point-transformer
implementation) belongs in its own isolated subtree (`<pkg>/ops/<name>/`), not mixed into the main
package's original code. Keep changes to it minimal, preserve the original attribution/license headers,
and give it its own nested build config (`pyproject.toml`/`setup.py`) if it's a native extension installed
as a local-path `uv` dependency (see `dependency-resolution.md`). Treat "this is vendored, minimize
changes" as a stronger version of the byte-for-byte preservation rule below — it's not just about a single
function, but about keeping the entire subtree diffable against its upstream source.

## Move order: dependency-ordered, leaves first

Before moving anything, sketch the internal import graph of the flat repo (`A imports B` for every
project-internal import). Move files in leaf-first order: files with zero internal dependencies first
(usually `utils.py`-type modules), then files that depend only on those, and so on up to the entrypoint
scripts last.

This matters because there is no test suite anywhere in this ecosystem — the only signal that a move
broke something is an import error. Moving leaf-first means that if an import breaks after any given step,
it is attributable to *exactly* the one file just touched, not tangled up with several simultaneous moves.
Run an import smoke test after every single file move, not just at the end:

```bash
uv run python -c "from <pkg>.<subpackage> import <ThingJustMoved>"
```

## Preserve load-bearing logic byte-for-byte

This is the concrete, checkable form of `SKILL.md`'s fidelity-over-idiom principle: Pythonic/KISS is the
default, but never at the cost of changing what a load-bearing piece of code actually computes. Some code
in a research repo is not incidental — it encodes a real invariant the model depends on.
Concrete worked example from a real adaptation: a model class overrode `.eval()` to swap a training-only
submodule for `nn.Identity()`, making the training-only branch structurally absent (and therefore free) at
inference time:

```python
def eval(self):
    super().eval()
    self.confidence_decoder = torch.nn.Identity()
    return self
```

When moving a file containing something like this, copy it unchanged — don't "clean it up," don't
reformat it, don't refactor the surrounding class while you're in there. Then **re-verify the contract
holds after the move**, with a real check, not a visual diff:

```python
model.eval()
assert isinstance(model.confidence_decoder, torch.nn.Identity)
depth = model(images, radar, get_confidence=False)
```

Look for this pattern before starting a restructuring pass: any `__init__`/`eval`/`train`-mode-dependent
branching, any monkey-patched or dynamically-assigned submodule, any comment warning "must be preserved" —
treat all of these as signals to preserve exactly and re-verify explicitly.

## Translate non-English comments

As part of the restructuring pass, translate any non-English source comments/docstrings encountered into
English, so the codebase is consistently maintainable by the whole team. Do this file-by-file alongside
the move (not as a separate blanket pass) — you're already reading every line closely at that point.
