"""TEMPLATE: unified metrics/image logging for a hand-rolled (non-Lightning) training loop.

This is a starting scaffold, not a drop-in file — before using it in a real adaptation:
  1. Rename `RunLogger` if the target repo has its own naming convention.
  2. Replace every "<PROJECT_NAME>" placeholder with the target repo's actual name.
  3. Confirm the `WANDB_API_KEY` guard below (see the deviation note) actually matches what this
     target repo needs — it's a recommended default, not a universal rule. If the target should
     behave like the Lightning-based reference repos (no code-level guard, rely on wandb's own
     client behavior), remove it.
  4. If the target repo doesn't use Hydra/OmegaConf, swap the `DictConfig`/`cfg.get(...)` calls for
     whatever plain-dict or argparse-namespace config object it actually uses — the fan-out logic
     underneath doesn't depend on Hydra specifically.

Only build/use this if the training loop is staying hand-rolled rather than being ported to
PyTorch Lightning — see references/lightning-porting.md for the preferred, reference-repo-parity
path, which gets real `WandbLogger`/`CSVLogger` fan-out for free instead of this approximation of
it. If porting to Lightning, delete this file's usage entirely; don't run both.

Design notes (see references/logging-facade.md for the full rationale):
  - A local `metrics.csv` is written unconditionally whenever enabled, regardless of W&B
    connectivity — mirrors what Lightning's CSVLogger guarantees for the Lightning-based path.
  - W&B fan-out is selected via a `logger=wandb|csv` config-group toggle (mirroring the same
    convention the Lightning-based reference repos expose), not a hardcoded `use_wandb` flag.
  - Must be constructed with `enabled=(RANK == 0)` explicitly by the caller — raw DDP gives no
    automatic rank-zero gating the way Lightning does.
  - Deliberate, flaggable deviation from the Lightning-based reference repos: this guards on
    `$WANDB_API_KEY` being present before calling `wandb.init()`, falling back to CSV-only with a
    warning rather than risking a blocking interactive login prompt in a headless/scripted run.
    State this explicitly in the target repo's own docs if kept, since it's a real behavioral
    difference from how the reference repos work.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Swap for the target repo's own logging setup (loguru, stdlib logging, print) if it doesn't
# already depend on loguru.
from loguru import logger as text_logger
from omegaconf import DictConfig, OmegaConf


class RunLogger:
    """Fans a metrics dict / image out to a local CSV and, optionally, Weights & Biases.

    Only ever active on the process that constructs it with ``enabled=True`` -- raw DDP gives no
    rank-zero guarding for free the way Lightning does, so callers must pass
    ``enabled=(RANK == 0)`` themselves.
    """

    def __init__(self, cfg: DictConfig, log_dir: str, run_name: str, enabled: bool = True):
        # `log_dir` is expected to already BE the current run's unified output directory (config
        # snapshot + train log + checkpoints + this facade's metrics.csv all together) -- see
        # references/infra-checklist.md's Output-folder convention. Don't point this at a
        # shared/fixed path across runs.
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / "metrics.csv"
        self.use_wandb = False
        self._wandb = None

        if not self.enabled:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger_cfg = cfg.get("logger", None)
        mode = logger_cfg.mode if logger_cfg is not None else "csv"

        if mode == "wandb":
            if not os.environ.get("WANDB_API_KEY"):
                text_logger.warning(
                    "logger=wandb but $WANDB_API_KEY is not set -- falling back to CSV-only "
                    "logging (see this file's module docstring for why, and whether that's the "
                    "right behavior for this repo)."
                )
            else:
                import wandb

                self._wandb = wandb
                self._wandb.init(
                    # prefer an interpolated project name built from dataset+task (mirroring the
                    # Lightning-based reference repos' convention -- see
                    # references/lightning-porting.md step 5) over a single fixed string, so
                    # the config passed in should already resolve "<PROJECT_NAME>" to something
                    # like "<pkg>_<dataset>_<task>" before it reaches here.
                    project=logger_cfg.get("project", "<PROJECT_NAME>"),
                    entity=logger_cfg.get("entity", None),
                    name=run_name,
                    tags=list(logger_cfg.get("tags", []) or []),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    dir=str(self.log_dir),
                )
                self.use_wandb = True

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        row = {"step": step, **metrics}
        self._append_csv(row)
        if self.use_wandb:
            self._wandb.log(metrics, step=step)

    def log_image(self, name: str, path: Any, step: Optional[int] = None) -> None:
        if not self.enabled or not self.use_wandb:
            return
        self._wandb.log({name: self._wandb.Image(str(path))}, step=step)

    def _append_csv(self, row: Dict[str, Any]) -> None:
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def finish(self) -> None:
        if self.enabled and self.use_wandb:
            self._wandb.finish()


# --- Usage sketch (delete once adapted into the target repo's actual training loop) ---
#
# run_logger = RunLogger(cfg, log_dir=log_dir, run_name=run_name, enabled=(RANK == 0))
#
# run_logger.log_metrics({
#     "train/loss": avg_loss,
#     "train/lr": lr_scheduler.get_last_lr()[0],
# }, step=current_step)
# run_logger.log_image("train/preview", "./train_log/preview.png", step=current_step)
#
# run_logger.finish()
#
# Verify all four paths before considering this done (see references/logging-facade.md):
# logger=csv writes metrics.csv; logger=wandb with WANDB_API_KEY set actually activates W&B
# (test with WANDB_MODE=offline to avoid touching the real project); logger=wandb with
# WANDB_API_KEY unset falls back to CSV-only with a warning, not a crash/hang; enabled=False is a
# complete no-op with zero files/directories created.
