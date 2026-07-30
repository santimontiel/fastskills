# Precision escape hatches

When the bf16-compatibility probe (see `SKILL.md` workflow step 4) flags a stage, there are **two
distinct fixes** used across the lineage — picking the right one depends on *why* it failed, not on
a fixed rule. Read the actual exception or NaN pattern before choosing.

## Pattern A: `torch.autocast(enabled=False)` around a whole submodule

Use when the submodule has **no bf16 kernel at all** for its core operation — the failure mode is
usually a hard `RuntimeError` naming an unsupported dtype/backend, not a silent NaN. Confirmed real
case: PTv3's sparse convolutions (`spconv`), used by the `PointsToGaussians` radar encoder in both
`tinycar-dev` and `gcarpred-dev`:

```python
# spconv's sparse convolutions don't support bf16 kernels, so this submodule always
# runs in fp32 regardless of an enclosing autocast context.
with torch.autocast(device_type="cuda", enabled=False):
    radar_point_features = self.encoder(radar_dict)
```

This disables the *ambient* autocast for everything inside the `with` block — the submodule computes
entirely in whatever dtype its inputs already are (cast them to fp32 first if they arrived as bf16
from an upstream autocast region). Scope it as tightly as possible: `gcarpred-dev`'s equivalent
module only wraps the PTv3 encoder call itself, not the surrounding mean/offset/covariance/opacity
MLPs, which run fine under the outer bf16 autocast and get explicit `.float()` casts afterward instead
(a case of both patterns used side-by-side, at different granularity, in the same module — narrow the
scope to exactly what needs it).

**A submodule needing this pattern is often (not always) also a `KNOWN_COMPILE_HOSTILE_STAGES`
member** — the same vendored/sparse backend that lacks a bf16 kernel frequently also lacks a
fake-tensor kernel for Dynamo. Confirm both independently; they're correlated, not identical (a
submodule can need one without the other).

## Pattern B: explicit `.float()` casts around a specific op

Use when the ambient bf16 autocast dtype is fine for the surrounding network, but **one specific
operation** needs fp32 regardless of it — the failure mode is usually a silent numerical problem
(NaN, large error) rather than a hard crash, because the op still "runs," just wrong. Two confirmed
real cases:

- **The Gaussian rasterizer** (`tinycar-dev`'s `render.py`, `gcarpred-dev`'s equivalent): the vendored
  CUDA rasterizer only accepts float32 tensors. `GaussianRenderer.forward` explicitly casts
  `means3D`/`colors_precomp`/`opacities`/`cov3D_precomp` to `.float().contiguous()` before calling it,
  regardless of the network's ambient precision, then casts the BEV output back to the ambient dtype
  afterward.
- **`torch.cdist` under TF32** (not bf16-autocast at all, but the same "narrow fp32 escape hatch"
  shape, and the single most consequential precision bug found in this lineage so far — see
  `references/case-studies.md` for the full story): a `_full_fp32_matmul()`-style context manager
  wrapping just the `cdist` call, forcing full fp32 matmul precision for that one operation while
  leaving TF32 enabled everywhere else.

The tell that distinguishes this from Pattern A: the op **doesn't crash and doesn't need a backend
that lacks bf16 support at all** — it produces a *wrong* answer under reduced precision (rounding
error large enough to flip a downstream mask, corrupt a matrix inverse, etc.). If the probe shows a
NaN/Inf appearing several ops downstream of where you'd expect, suspect this pattern and trace
backward to the actual op producing bad values, rather than wrapping the first "suspicious-looking"
submodule in `autocast(enabled=False)`.

## Also watch for: an implicit-cast assumption that silently breaks

Not every bf16 bug is "this op needs a fp32 escape hatch" — sometimes it's dead/incorrect code that
assumed a specific precision string. Confirmed real case (`gaussiancar-dev`, fixed in `gcarpred-dev`'s
commit `30102b6`): a blanket batch→bf16 cast gated on `if self.cfg.trainer.precision == "bf16":` that
never fired once the config actually used Lightning's real string, `"bf16-mixed"` — inert code that
would have blindly cast label/integer tensors too if it had ever matched. When adapting a precision
code path, grep for every literal `"bf16"`/`"fp16"`/`"32-true"`-style string comparison and confirm it
actually matches what the config produces, not what the author assumed the config would say.

## Verify the fix, don't just apply it

After applying either pattern, **re-run the bf16 probe** on that exact stage before moving on. A
fix that "looks right" but doesn't actually clear the NaN (wrong scope, wrong op targeted) is worse
than not fixing it at all, since it looks resolved in the report but isn't.
