# Logging facade (fallback path)

This is the fallback when the training loop is **not** being ported to Lightning (see
`lightning-porting.md` for the preferred, reference-repo-parity path — read that first if the port is
even plausibly in scope, since it gets real `WandbLogger`/`CSVLogger` fan-out for free instead of an
approximation of it).

A hand-rolled `torch.distributed` loop gets none of what Lightning gives Lightning-based repos for
free: no automatic multi-logger fan-out, no automatic rank-zero gating for logging calls. This needs an
explicit small facade class, built once per target repo (start from `assets/run_logger_template.py`
rather than writing from scratch).

## Design requirements

- **Write a local `metrics.csv` unconditionally**, whenever the facade is enabled — this mirrors what
  Lightning's `CSVLogger` guarantees (metric history survives regardless of W&B connectivity). Don't make
  this conditional on W&B being active; it should always happen.
- **Fan out to `wandb.log()`/`wandb.Image()` optionally**, gated on a config toggle. Mirror the *same*
  `logger=wandb|csv` config-group selection convention the Lightning-based sibling repos already expose
  (see `lightning-porting.md`'s step 5), even though the implementation underneath is necessarily
  different — a consistent toggle interface across repos matters more than the two implementations being
  identical.
- **Must be constructed with `enabled=(RANK == 0)` explicitly** by the caller. Raw DDP gives no automatic
  rank-zero gating the way Lightning does — every rank runs the same Python code, so without this the
  facade would write N copies of the same metrics from N processes.
- **Recommended, but a judgment call, not a universal rule**: guard on `$WANDB_API_KEY` being present
  before calling `wandb.init()`, falling back to CSV-only with a logged warning if it's absent, rather than
  letting an unauthenticated `wandb.init()` potentially block on an interactive login prompt in a
  headless/scripted run. **State this explicitly as a deliberate deviation** from how the Lightning-based
  sibling repos behave (they rely entirely on `wandb`'s own client library behavior, with no code-level
  guard) — don't silently diverge without calling it out in a comment or in the CLAUDE.md/README update.

## Usage shape

```python
run_logger = RunLogger(cfg, log_dir=log_dir, run_name=run_name, enabled=(RANK == 0))

# periodically during training:
run_logger.log_metrics({
    "train/loss": avg_loss,
    "train/loss/depth": avg_depth_loss,
    "train/lr": lr_scheduler.get_last_lr()[0],
}, step=current_step)
run_logger.log_image("train/preview", "./train_log/preview.png", step=current_step)

# at the end:
run_logger.finish()
```

## Verification

Test all of these paths before considering the facade done — each is a distinct failure mode:
1. `logger=csv` mode: confirm `metrics.csv` gets written with the expected columns/values, no wandb import
   attempted.
2. `logger=wandb` mode with `WANDB_API_KEY` set: confirm `wandb.init()` actually activates
   (`rl.use_wandb is True`) and a real `wandb.log()` call succeeds — test with `WANDB_MODE=offline` to
   avoid creating a real cloud run against the actual W&B project while just verifying the facade works.
3. `logger=wandb` mode with `WANDB_API_KEY` **unset**: confirm graceful fallback to CSV-only with a
   warning, not a crash or a hang.
4. `enabled=False` (simulating a non-zero rank): confirm the facade is a complete no-op — no directory or
   file gets created at all.

See `assets/run_logger_template.py` for a starting implementation covering all of the above.
