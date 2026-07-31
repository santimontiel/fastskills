# Precision escape hatches

When the bf16-compatibility probe (see `SKILL.md` workflow step 4) flags a stage, there are **three
distinct fixes** used across the lineage — picking the right one depends on *why* it failed, not on
a fixed rule. Read the actual exception or NaN pattern before choosing. Pattern C is a different
*class* of bug from A/B — read its own section before assuming the probe already covers it, because
it doesn't.

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

## Pattern C: `.float()` before a bf16 tensor crosses into non-PyTorch code

Use when the crash isn't in the model's math at all, but in what happens to the model's OUTPUT
afterward — logging, preview images, metrics computed with NumPy/OpenCV/Matplotlib, anything outside
PyTorch's own tensor ops. The tell: `TypeError: Got unsupported ScalarType BFloat16` (or an equivalent
from whatever library), not a `RuntimeError` from a CUDA kernel and not a silent NaN — NumPy has no
native bfloat16 dtype at all, so `.cpu().numpy()` on a bf16 tensor hard-crashes unconditionally,
regardless of GPU/backend.

**Confirmed recurring across repos in this lineage** (per direct user report — this is not a one-off):
a training loop's preview/visualization logging (`_log_previews`-style helper, called periodically to
save/log sample images) calls `.detach().cpu().numpy()` on the model's raw output tensors. Those
tensors are computed inside the bf16-mixed autocast region, so they arrive as `bfloat16` — the FIRST
time bf16-mixed is actually exercised end-to-end in a real training run, not before. Confirmed real on
JustDepth (2026-07-30): `JustDepthModule._log_previews` crashed on `logits[0].detach().cpu().numpy()`
~150 steps into the first bf16-mixed training run ever attempted on this repo. Fixed the same way each
time — insert `.float()` right before the `.cpu()`/`.numpy()` boundary, on exactly the tensors that
are themselves model outputs (raw dataloader inputs like images/lidar/radar are untouched by autocast
and don't need this):

```python
logit = logits[0].float().detach().cpu().numpy()  # was: logits[0].detach().cpu().numpy()
confidence_vis = confidence_vis[0].float().detach().cpu().numpy()
```

**Why the bf16-compatibility probe (SKILL.md step 4) does NOT catch this**: the probe only runs the
model's `forward()` under bf16 autocast and checks the returned tensors for NaN/Inf — it never calls
`.numpy()` or exercises any training-loop logging/preview code, because that code isn't part of the
model at all. A clean probe result says nothing about this class of bug. **The only thing that catches
it is SKILL.md step 9's real training run** — one more reason that step is the actual acceptance gate,
not the sweep numbers. Before enabling `trainer.precision=bf16-mixed` for training (as opposed to just
inference), grep the training step / any periodic logging helper for `.numpy()`, `.item()` calls in a
loop building a Python list per-pixel, `cv2.*`, or any other non-PyTorch consumer downstream of the
model's own outputs, and pre-emptively add `.float()` there rather than waiting to hit the crash live.

## Also watch for: an implicit-cast assumption that silently breaks

Not every bf16 bug is "this op needs a fp32 escape hatch" — sometimes it's dead/incorrect code that
assumed a specific precision string. Confirmed real case (`gaussiancar-dev`, fixed in `gcarpred-dev`'s
commit `30102b6`): a blanket batch→bf16 cast gated on `if self.cfg.trainer.precision == "bf16":` that
never fired once the config actually used Lightning's real string, `"bf16-mixed"` — inert code that
would have blindly cast label/integer tensors too if it had ever matched. When adapting a precision
code path, grep for every literal `"bf16"`/`"fp16"`/`"32-true"`-style string comparison and confirm it
actually matches what the config produces, not what the author assumed the config would say.

## Verify the fix, don't just apply it

After applying Pattern A or B, **re-run the bf16 probe** on that exact stage before moving on. A
fix that "looks right" but doesn't actually clear the NaN (wrong scope, wrong op targeted) is worse
than not fixing it at all, since it looks resolved in the report but isn't. Pattern C fixes can't be
verified by the probe at all (see its own section) — re-run the actual training step / logging code
path that crashed instead.
