# Use cases / edge cases catalog

Where `case-studies.md` is a chronological log (what happened, what was measured, what was decided),
this file is an **indexed catalog of structural edge cases** — patterns in how a target repo's model
is shaped that change how stage discovery, the bf16 probe, or the sweep itself needs to be handled.
Check this file **before** stage discovery on a new repo (workflow step 2) — most edge cases a new
repo hits have already been seen once. When a genuinely new edge case turns up that isn't already
catalogued here, add it (Self-Improve, workflow step 11) — don't only log it narratively in
`case-studies.md`; a future invocation needs to find it by *pattern*, not by reading through history.

## Entry format

```
### Edge case: <short name>

**What it looks like**: the structural symptom, in code terms.
**Handling**: how assets/compile_bf16_sweep.py or the workflow already accounts for it (or doesn't
yet, if this is a newly-catalogued gap).
**Confirmed present in**: repo names, updated every time it's seen again.
```

## Catalog

### Edge case: a stage's `forward()` returns a dict, not a bare tensor

**What it looks like**: `feats = model.image_encoder(images)` where `feats` is a dict of named
tensors (`{"depth": ..., "offsets": ..., "features": ..., "opacity": ...}`), consumed piecemeal by the
rest of `forward()`.
**Handling**: `probe_bf16_compatibility`'s `_iter_tensors()` walks nested dict/list/tuple output
structures recursively rather than assuming a bare tensor or plain tuple — this was a real bug in the
tool's first draft (see `case-studies.md`'s 2026-07-30 self-review entry), now fixed.
**Confirmed present in**: `tinycar-dev`, `gcarpred-dev` (both `image_encoder` stages).

### Edge case: a stage is invoked more than once per forward pass

**What it looks like**: a per-timestep loop (e.g. `for dt in DT_STEPS: ... model.gs_render_image(...)`)
calling the same submodule several times in one `forward()`.
**Handling**: both `StageLatencyProbe` (accumulates every call's latency as a separate sample — correct
by construction) and the bf16 probe (accumulates "any call, any NaN" rather than "last call only" —
fixed after being a real bug, see `case-studies.md`) already handle this correctly.
**Confirmed present in**: `gcarpred-dev` (`gs_render_image`/`gs_render_radar`/`gaussian_evolution`,
each called once per `DT_STEPS` entry).

### Edge case: a stage is a bound method, not an `nn.Module` child

**What it looks like**: `model._pred_depth(...)` called directly inside `forward()`, where
`_pred_depth` is a plain method, not a registered submodule — `named_children()` can't find it, and
no forward hook fires on it independently.
**Handling**: must be added by hand to `EXTRA_METHOD_STAGES` (see `references/stage-discovery.md`).
The bf16 probe and `StageLatencyProbe` do not cover these automatically — inspect them manually.
**Confirmed present in**: `tinycar-dev`/`gcarpred-dev` (`camera_unprojection` → `_pred_depth`;
`gaussian_evolution` → `_evolve_covariances`/`_evolve_opacities`/`_evolve_features`).

### Edge case: which attribute to compile is resolved from a config value at runtime

**What it looks like**: the actual bound method to compile for a logical stage depends on another
config field (`model.query_init_mode` selecting one of three possible method names).
**Handling**: `EXTRA_METHOD_STAGES` accepts a `callable(model) -> attr_name` in addition to a plain
string, specifically for this case (see `references/stage-discovery.md`).
**Confirmed present in**: `gcarpred-dev` (`query_init` → `_grid_sample_query_init` /
`_knn_gaussian_query_init` / `_hybrid_query_init`, keyed by `model.query_init_mode`).

### Edge case: the target repo's config system is yacs, not Hydra

**What it looks like**: per-module compile is a single boolean per module
(`cfg.MODEL.COMPILE_ENCODER`), not a comma-separated string field — the `compile_stages="a,b,c"`
convention in `references/hydra-wiring.md` doesn't map onto the config directly (only onto a
benchmark tool's own `argparse` flag, which is independent of production config).
**Handling**: not yet automated — wire one boolean config field per stage instead of the
comma-separated convention when the target repo is yacs-based; the *sweep tool itself* is unaffected
(it doesn't depend on the target's production config system at all), only the final config-wiring
step (workflow step 8) changes shape.
**Confirmed present in**: `fiery-radar`, `powerbev-radar`.

### Edge case: mixed precision is architecturally blocked for one submodule, with no available fix

**What it looks like**: the bf16 probe (or a real training NaN) traces back to a submodule whose
*backward pass*, not just forward, lacks bf16/fp16 support at the framework/kernel level — no amount
of `autocast(enabled=False)` scoping fixes this, since the limitation is in the op's CUDA kernel
itself, not in how it's being called.
**Handling**: the actual fix in this lineage was swapping the submodule for a different, bf16-native
implementation entirely (see `references/case-studies.md`'s PTv3 entry) — not an escape hatch. If the
probe flags a stage and no escape-hatch pattern from `references/precision-escape-hatches.md` clears
it, check whether an alternative implementation already exists in the repo/lineage before concluding
the repo simply can't use bf16.
**Confirmed present in**: `gaussiancar` (PTv3 radar encoder's backward pass; repo stays FP32-only
today as a result).

### Edge case: a precision bug that isn't a bf16-autocast issue at all

**What it looks like**: a NaN appears under a specific matmul-precision setting (TF32,
`torch.set_float32_matmul_precision("high")`) that has nothing to do with bf16 autocast — the bf16
probe in this tool would report "ok" for a stage with this exact bug, since it only tests bf16
autocast, not TF32.
**Handling**: not automatable by this tool as designed — documented as a known blind spot in
`probe_bf16_compatibility`'s own docstring. If the target repo uses `torch.cdist` or any other
pairwise-distance-style op, explicitly check its behavior under TF32 separately, per
`references/case-studies.md`.
**Confirmed present in**: `tinycar-dev`, independently also `gcarpred-dev`.

### Edge case: the target repo has no compile/bf16 prior art at all

**What it looks like**: no `tools/benchmark.py`-equivalent, no `torch.compile`/`autocast` call sites
anywhere, precision still `"32-true"`.
**Handling**: this is the common case this skill is built for, not an edge case requiring special
handling — start fresh from `assets/compile_bf16_sweep.py`, no existing tool to extend.
**Confirmed present in**: `JustDepth` (mid-migration to the family's Hydra/`tools/` shape at time of
writing, no benchmark tooling yet).
