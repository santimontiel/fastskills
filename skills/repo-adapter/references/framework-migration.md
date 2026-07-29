# Framework migration

Bumping a training framework across major versions is its own distinct sub-task from the infra plumbing —
budget real time for it, it's usually bigger than it looks. This file's concrete checklist is for PyTorch
Lightning 1.x → 2.5+, the most common case in this ecosystem, but the same "check the target framework's
migration guide for the version jump, then audit for these categories of breakage" approach applies to any
framework bump.

## PyTorch Lightning 1.x → 2.5+ checklist

- `import pytorch_lightning as pl` → `import lightning.pytorch as pl`. The old top-level package still
  exists as a compatibility shim in some versions but shouldn't be relied on going forward.
- `training_epoch_end(self, step_outputs)` / `validation_epoch_end(self, step_outputs)` — **removed**,
  took accumulated per-step outputs as an argument. Replace with `on_train_epoch_end(self)` /
  `on_validation_epoch_end(self)` — no argument; accumulate whatever you need in `self` across steps
  instead (or better, use `self.log(..., on_epoch=True)` and let Lightning do the accumulation).
- `self.hparams = x` → `self.save_hyperparameters(x)`. The direct-assignment form stops
  `load_from_checkpoint` from correctly restoring hyperparameters.
- `Trainer(gpus=N, accelerator='ddp', weights_summary=..., plugins=DDPPlugin(...))` →
  `Trainer(accelerator='gpu', devices=N, strategy='ddp_find_unused_parameters_true')` (or a `DDPStrategy(...)`
  instance for finer control). The old `gpus=`/string-`accelerator=`/`plugins=` triplet is gone.
- `from pytorch_lightning.metrics import ...` (pre-1.3 API, fully removed) → port `Metric` subclasses to
  `torchmetrics.Metric`. The `update()`/`compute()` method bodies usually don't need to change — just the
  base class import and any now-deleted functional helpers, which need reimplementing locally, preserving
  their exact original semantics (check against the metric's mathematical definition, not just "it runs").

## Precision

- **Use `bf16-mixed`, not `fp16-mixed` or plain integer `16`.** fp16's narrow dynamic range can silently
  NaN training partway through a run — especially with `torch.exp(...)`/softmax-heavy terms, common in
  uncertainty-weighted multi-task losses or attention. Remap any `precision: 16` config value to
  `'bf16-mixed'` explicitly; don't assume the framework's default is already safe.
- Audit any `torch.zeros(...)`/buffer creation with a hardcoded dtype (or no dtype, defaulting to fp32)
  that later receives values from a tensor which could be a different dtype under autocast — these crash
  or silently corrupt under bf16/fp16 mixed precision. Fix by deriving the buffer's dtype from the source
  tensor's `.dtype` rather than hardcoding it.

## Dependency-drift gotchas (not framework-specific, but always surface at this stage)

- `cv2.line()`/similar OpenCV drawing calls under a modern `opencv-python` reject numpy `int64` point
  tuples — cast coordinates to native Python `int` before passing them.
- `matplotlib`'s `FigureCanvas.tostring_rgb()` is removed in modern versions — use `buffer_rgba()`
  instead, and explicitly force `matplotlib.use('Agg')` for headless rendering in a container/cluster job.
- `np.float`, `np.int`, `np.bool`, `np.object` (bare Python-type aliases) were removed in numpy ≥1.24 —
  replace with `np.float64`/`int`/`bool`/`object` as appropriate. These often sit in genuinely dead code
  paths (never hit by the default config), so a plain `grep` for the old alias names across the repo finds
  more than what a single smoke-test run would surface.
- `torch.load(...)`'s `weights_only` default flipped to `True` in PyTorch 2.6 — any checkpoint-loading code
  that doesn't pass `weights_only=False` explicitly may start raising on checkpoints containing non-tensor
  objects (optimizer state, custom classes) that loaded fine before. Audit every `torch.load` call site,
  not just the obvious "load the model" one — resume/checkpoint-restore code paths are easy to miss.

## Note for hand-rolled (non-Lightning) loops

If the target repo doesn't use Lightning and isn't being ported to it (see `lightning-porting.md` for that
decision), the API-rename checklist above doesn't apply — but the precision guidance and dependency-drift
gotchas still do. A hand-rolled loop needs its own manual `torch.autocast(dtype=torch.bfloat16)` context
around the forward/loss computation instead of a `Trainer(precision=...)` config string.
