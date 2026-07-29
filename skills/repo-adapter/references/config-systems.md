# Config systems

## Decision rule

**Preserve the existing config system by default.** Only migrate to Hydra if the source system is
genuinely ad hoc/fragile — e.g. a bespoke hand-rolled INI parser with a manual CLI-overrides-beat-config
precedence helper reimplementing what Hydra's native `key=value` CLI overrides already give for free —
**and** the user explicitly agrees during Before You Start. Never migrate a working yacs, OmegaConf, or
argparse-based config system just because Hydra is the more familiar tool; `fiery-radar`/`powerbev-radar`
both explicitly kept their yacs config as-is, only bumping its dependency version, and that was the right
call for those repos.

If migrating: design the new config tree mirroring the reference repo's actual structure
(`configs/{data,module,task}/*.yaml` + top-level `train.yaml`/`eval.yaml` with a `defaults:` list is the
common shape), translating every existing config key one-for-one rather than inventing a new structure
from scratch — this keeps behavior identical while only changing the mechanism.

## The DDP + Hydra chdir bug

If the target keeps (or has) a `torchrun`-launched, non-Lightning DDP training loop, this bug will bite
the moment Hydra is introduced: `torchrun --nproc_per_node=N` launches **N independent processes**, each
of which independently invokes `@hydra.main`. Hydra's default behavior is to `chdir` into a fresh,
timestamped `outputs/<date>/<time>/` directory *per invocation* — with N processes launched
near-simultaneously, this can produce N different output directories, or race on directory creation. This
silently breaks any assumption the training loop has about all ranks sharing one working directory /
checkpoint tree.

Fix: disable Hydra's per-run chdir entirely in the relevant top-level configs (`train.yaml`, `eval.yaml`):

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
```

This keeps Hydra as a pure config-composition layer — it never manages output directories — leaving
whatever directory-layout logic the training loop already has (e.g. a `Session`/checkpoint-manager class)
as the sole source of truth for where things get written.

**Verify this explicitly, don't just apply the fix and assume it works**: launch via real `torchrun` with
2+ ranks and confirm no stray `outputs/` directory gets created:

```bash
rm -rf outputs
CUDA_VISIBLE_DEVICES="" uv run torchrun --nproc_per_node=2 tools/train.py --cfg job local=false
ls outputs 2>&1 || echo "no outputs/ dir created (expected)"
```

(If porting the training loop to Lightning instead — see `lightning-porting.md` — Lightning manages its
own DDP process spawning internally rather than via external `torchrun`, so this specific race doesn't
apply; Lightning-based reference repos set `hydra.run.dir` to a real per-run timestamped path precisely
because they don't have this problem.)

## Dataset paths: env var with a Docker-mount fallback

Use an OmegaConf environment-variable resolver so the *same* config value resolves correctly whether the
code runs directly on a host (outside Docker) or inside the container launched by `make run`/Slurm:

```yaml
# configs/data/nuscenes.yaml
nuscenes_root: ${oc.env:NUSCENES_DATA_ROOT,/data/nuscenes}
data_root: ${.nuscenes_root}/samples
```

This resolves to the real host path when `$NUSCENES_DATA_ROOT` is set (host-direct execution), and falls
back to `/data/nuscenes` — the fixed in-container mount point the Makefile's `-v` flag targets — when the
env var isn't present (container execution, since the Makefile mounts the host path there but does not
forward the env var itself; see `infra-checklist.md`). One config value, both execution modes, no
host/container config variants needed.

Verify by dry-running config resolution for every dataset variant:

```bash
uv run python tools/train.py --cfg job --resolve data=<variant>
```

## The `os._exit()` + buffered-stdout gotcha

Entrypoint scripts that call `os._exit(0)` after `main()` (a common pattern to skip slow interpreter
teardown, especially useful for DDP training that can otherwise hang on exit) can silently discard
output — including a `--cfg job` config dump — if stdout is piped to a non-tty and therefore
block-buffered. Fix by flushing explicitly before the exit call:

```python
if __name__ == '__main__':
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exit.")
        sys.stdout.flush()
        os._exit(0)
```

If a `--cfg job` dry-run produces suspiciously empty output but exits with code 0, this is the first thing
to check — it's a silent failure mode, not an obvious crash.
