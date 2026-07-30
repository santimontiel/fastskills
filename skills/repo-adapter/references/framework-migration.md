# Framework migration

Bumping a training framework across major versions is its own distinct sub-task from the infra plumbing —
budget real time for it, it's usually bigger than it looks. This file's concrete checklist is for PyTorch
Lightning 1.x → 2.5+ (tinycar-dev's current version, and the target to match unless Before You Start says
otherwise), the most common case in this ecosystem, but the same "check the target framework's migration
guide for the version jump, then audit for these categories of breakage" approach applies to any framework
bump. The instantiation pattern reference repos actually use is Hydra's `_target_` + `hydra.utils.instantiate`
(see `config-systems.md`) — verify a migrated `Trainer`/callback/logger config still resolves through that
mechanism, not a hand-built kwargs dict.

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
- **TF32 and pairwise-distance ops don't mix silently — audit any `torch.cdist` (or similar pairwise
  distance) call under TF32.** A real, documented case (see `references/case-studies.md`'s tinycar-dev
  entry): enabling TF32 matmul alongside a `torch.cdist`-based distance computation silently produced NaNs
  for a specific input distribution, zeroing out an entire branch's contribution for a full multi-epoch
  training run with no crash and no visibly-broken loss curve. The fix was forcing full-fp32 matmul for the
  affected computation plus an explicit degenerate-case guard. Treat this as a standing instruction, not
  just historical trivia: whenever a migration changes precision/TF32 settings on a repo with a `cdist`-like
  op, explicitly check intermediate tensor statistics (not just the loss curve) for NaN/Inf before trusting
  the run — this is exactly the case `verification.md`'s "check log scalars for NaN/Inf explicitly" step
  exists for, and it's a concrete instance of the fidelity-over-idiom principle in `SKILL.md`: a "cleaner"
  precision setting that silently changes numeric output has failed the port, however plausible the curves
  look.

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
