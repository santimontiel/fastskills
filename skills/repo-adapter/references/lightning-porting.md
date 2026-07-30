# Porting a hand-rolled loop to PyTorch Lightning

This is the reference-repo-parity path: tinycar-dev and every migrated sibling repo (`fiery-radar`,
`powerbev-radar`, `JustDepth`) are actually Lightning-based, so a target repo with a hand-rolled
`torch.distributed` loop permanently diverges from the reference shape unless this port happens. It is
also the only way to get the *real* thing when the user wants "the same logger as the reference repo" —
Lightning's multi-logger fan-out (`WandbLogger` + `CSVLogger`) is a Lightning feature, not something a
hand-rolled facade can fully replicate (see `logging-facade.md` for the compensating alternative when this
port is out of scope).

**Only do this with explicit user buy-in from Before You Start.** It touches every part of the training
code — loss computation, optimizer/scheduler setup, checkpoint save/load, DDP launch, dataloader wiring —
and is materially bigger and riskier than keeping the loop and adding a facade. Don't default into it.

## The model.py / module.py split

The worked pattern below mirrors tinycar-dev's actual split between `modeling/model.py`
(`TinyCaR` — pure `nn.Module` forward wiring, no training-loop concerns at all: it just defines how a
batch flows through the network's components) and `modeling/module.py` (`TinyCaRModule` — the
`LightningModule` that *wraps* the model and owns everything training-loop-specific: loss computation,
metric logging, optimizer/scheduler config, checkpoint hooks). Keep this split when porting rather than
collapsing training-loop logic into the same class as the forward pass — it keeps the model importable and
testable independent of Lightning, and matches every Lightning-based sibling repo's actual shape. Read the
*target* reference repo's real equivalent files too, since details (metric handling, loss-weighting scheme)
vary, but this two-file split is stable across the Lightning-based sibling repos.

Architectural sub-blocks (encoders, decoders, fusers, heads) belong in a `modeling/components/` subpackage,
grouped by role — e.g. interchangeable backbones as `*_encoders/` subpackages selected via Hydra
(`module/image_encoder=...`), each with an `__init__.py` re-exporting its public classes so `_target_`
paths stay short and swapping one backbone for another is a one-line config change, not a code change.

**1. Wrap the model in a `LightningModule`:**

```python
class MyModule(L.LightningModule):
    def __init__(self, cfg, model, losses, metrics=None):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.losses = losses
        self.metrics = torch.nn.ModuleDict(metrics or {})

    def forward(self, batch):
        return self.model(batch)
```

**2. Collapse duplicated train/eval loop bodies into one `common_step`:**

A hand-rolled loop usually has near-duplicated logic for the training loop and the eval loop (forward
pass, loss computation, metric accumulation, logging — same shape, different data). Replace both with one
`common_step(batch, stage)` called from thin overrides:

```python
def common_step(self, batch, stage: str = "train"):
    outputs = self(batch)
    loss, loss_details = self.losses(outputs, batch)
    B = batch["input"].shape[0]

    self.log(f"{stage}/loss", loss.detach(), on_step=False, on_epoch=True, logger=True, batch_size=B)
    self.log_dict(
        {f"{stage}/loss/{k}": v.detach() for k, v in loss_details.items()},
        on_step=False, on_epoch=True, logger=True, batch_size=B,
    )
    return {"loss": loss}

def training_step(self, batch, batch_idx):
    return self.common_step(batch, stage="train")

def validation_step(self, batch, batch_idx):
    return self.common_step(batch, stage="val")
```

`self.log`/`self.log_dict` replace manual `.item()` accumulation + averaging + periodic print — and
critically, Lightning performs the correct DDP all-reduce across ranks automatically. **A hand-rolled loop
that only logs from rank 0 without an explicit all-reduce silently reports rank-0-skewed metrics, not the
true global average** — this is easy to miss because the numbers still look plausible, just wrong.

**3. Replace hand-rolled checkpoint state:**

A typical hand-rolled loop has a `Session`/`TrainClock`-style class doing manual `torch.save({'network':
..., 'clock': ..., 'optimizer': ..., 'lr_scheduler': ...}, path)` and manual `torch.load(path,
weights_only=False)` + `strict=True` reload on resume. Replace with:

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=self.cfg.optimizer.lr,
        total_steps=self.trainer.estimated_stepping_batches,
    )
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
```

plus a `ModelCheckpoint` callback instead of manual `torch.save` calls:

```yaml
callbacks:
  - _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: "${hydra:runtime.output_dir}/checkpoints"
    monitor: "val/metrics/<key>"
    save_top_k: 1
    mode: "max"
    save_last: True
```

wired through `hydra.utils.instantiate` from the config — the same mechanism used for loggers, see below.

**4. Replace manual DDP setup and the `torchrun` launch:**

Manual `dist.init_process_group(backend='nccl')` + `DistributedSampler` + `DDP(model, device_ids=[...])`
wrapping + external `torchrun --nproc_per_node=N tools/train.py` launch all get replaced by:

```python
trainer = L.Trainer(
    accelerator="cuda",
    devices=cfg.trainer.devices,
    strategy="ddp_find_unused_parameters_true",
    callbacks=callbacks,
    logger=loggers,
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

Lightning owns process spawning, sampler injection, and gradient sync internally.

**This is a concrete, visible operational change — flag it explicitly, don't let it surprise anyone
later**: the launch command changes from

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 tools/train.py
```

to a single-process

```bash
uv run tools/train.py trainer.devices=2
```

Update `deploy/slurm/train_slurm.sh` (drop the `torchrun`/`CUDA_VISIBLE_DEVICES` wrapping around the
`uv run` call) and every `CLAUDE.md`/`README.md` command block that documents the old launch syntax —
these go stale silently otherwise, since nothing errors, the commands just launch fewer GPUs than intended.

**5. Real multi-backend logging, for free:**

```yaml
# configs/logger/wandb.yaml
# @package loggers
- _target_: lightning.pytorch.loggers.wandb.WandbLogger
  # interpolated, not hardcoded — the wandb project should track which dataset+task the run
  # actually is, without needing to hand-edit this string per task variant
  project: "<pkg>_${data.model_config.dataset_name}_${task.key}"
  name: "${run_id}"
  save_dir: "${hydra:runtime.output_dir}"
- _target_: lightning.pytorch.loggers.CSVLogger
  save_dir: "${hydra:runtime.output_dir}"
  name: "csv"
  version: ""
```

`hydra:runtime.output_dir` should point at the unified per-run output folder (see
`infra-checklist.md`'s Output-folder convention) so the config snapshot, `train.log`, checkpoints, and
these logger subdirs all land together under one run-id-scoped directory — not split across a separate
legacy checkpoints-only tree.

instantiated in the entrypoint via a small loop over `hydra.utils.instantiate`, and passed as
`Trainer(logger=loggers)`. Every `self.log`/`self.log_dict` call is then broadcast to *every* logger in
that list automatically — this is real Lightning multi-logger fan-out, not an approximation of it. **If
porting to Lightning, `logging-facade.md`'s hand-rolled `RunLogger` is not needed at all** — the two files
are mutually exclusive alternatives for the same underlying problem (getting metrics into both a local
record and W&B), not something to apply together.

**6. Precision:**

`trainer.precision: "bf16-mixed"` as a config string replaces any manual autocast/GradScaler code the
original loop had. Cross-reference `framework-migration.md`'s bf16-over-fp16 guidance — it applies
identically here.

## Worked nuance: task-conditioned multi-head training

A pattern worth reusing rather than reinventing when a target repo needs to train against several related
but distinct label sets at once (e.g. multiple object-class groups plus several map layers, all from the
same batch): keep one shared backbone/decoder, but make the segmentation head(s) and the metrics they're
scored against config-driven per task, rather than hardcoding a fixed head/metric list into the
`LightningModule`. Concretely, this means a composite head that fans out to per-group sub-heads based on
an `object_groups`-style config list, and a matching `metric_groups` config that determines which metrics
get computed (and which checkpoint metric gets monitored) for the currently-configured task. This keeps
`module.py` itself generic across "train on vehicles only" vs. "train jointly on vehicles + bikes +
pedestrians + map" — the task config changes, not the module code. If a target repo has this kind of
multi-task shape (or is likely to grow into one), prefer this config-driven composite-head pattern over a
one-off hardcoded head for whatever the first task happens to be.

## Verification specifics for this path

- Real multi-GPU run via `Trainer(devices=N)`, not `torchrun` — confirm gradient sync actually happens
  (loss decreases consistently across ranks, not just on rank 0).
- Watch for `find_unused_parameters` needs: if any parameters (e.g. an auxiliary head, a conditionally-used
  submodule) don't receive gradients on every single step, plain `"ddp"` strategy will hang or error —
  use `"ddp_find_unused_parameters_true"` in that case.
- Consider `num_sanity_val_steps=0` if validation requires a fully-loaded dataset/index that's expensive to
  build just to sanity-check 2 batches before training even starts.
- Otherwise follow the same staged discipline in `verification.md` — import checks, dry-run config
  composition, synthetic forward pass, then the real one-epoch run against a baseline.
