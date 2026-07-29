# Porting a hand-rolled loop to PyTorch Lightning

This is the reference-repo-parity path: every known reference repo (`tinycar-dev`, `gaussiancar`,
`gcarpred-dev`, `fiery-radar`, `powerbev-radar`) is actually Lightning-based, so a target repo with a
hand-rolled `torch.distributed` loop permanently diverges from all of them unless this port happens. It is
also the only way to get the *real* thing when the user wants "the same logger as the reference repo" —
Lightning's multi-logger fan-out (`WandbLogger` + `CSVLogger`) is a Lightning feature, not something a
hand-rolled facade can fully replicate (see `logging-facade.md` for the compensating alternative when this
port is out of scope).

**Only do this with explicit user buy-in from Before You Start.** It touches every part of the training
code — loss computation, optimizer/scheduler setup, checkpoint save/load, DDP launch, dataloader wiring —
and is materially bigger and riskier than keeping the loop and adding a facade. Don't default into it.

## Concrete porting pattern

The worked pattern below mirrors `tinycar-dev`'s actual `TinyCaRModule`
(`tinycar/modeling/module.py`) — read the *target* reference repo's real equivalent file too, since
details (metric handling, loss-weighting scheme) vary, but the overall shape below is stable across the
Lightning-based sibling repos.

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
  project: "<pkg>_${data.name}"
  name: "${run_id}"
  save_dir: "${hydra:runtime.output_dir}"
- _target_: lightning.pytorch.loggers.CSVLogger
  save_dir: "${hydra:runtime.output_dir}"
  name: "csv"
  version: ""
```

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
