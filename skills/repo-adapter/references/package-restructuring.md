# Package restructuring

Only do this when it was explicitly scoped in during Before You Start — a `tools/` wrapper around flat
scripts is often sufficient and lower-risk. Full restructuring is justified when the target repo's flat
layout is itself an obstacle (e.g. scripts import each other via relative paths that break once anything
moves, or the repo needs to become pip-installable as a dependency of something else).

## Target shape

Flat scripts at repo root (`train.py`, `model.py`, `dataset.py`, `utils.py`, ...) become an installable
package with a conventional split, e.g.:

```
<pkg>/
  data/       dataset classes, augmentation, shared constants
  modeling/   model architecture
  engine/     run(cfg) implementations — the actual train/eval/etc. loop bodies
  utils/      geometry/math/misc helpers with no internal dependencies
tools/
  train.py    thin wrapper: parse config, call <pkg>.engine.train.run(cfg)
  eval.py     thin wrapper: parse config, call <pkg>.engine.eval.run(cfg)
```

`tools/*.py` should be thin — argument/config parsing plus one call into `<pkg>.engine.*`. All real logic
lives in the package so it's importable and testable independent of the CLI entrypoint.

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

Some code in a research repo is not incidental — it encodes a real invariant the model depends on.
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
