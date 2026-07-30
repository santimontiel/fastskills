# Hydra wiring

## The comma-separated `compile_stages` convention

Confirmed identical (down to the exact wording of the CLI-quoting warning) in both `tinycar-dev` and
`gcarpred-dev`'s `configs/train.yaml` and `configs/benchmark.yaml`:

```yaml
# torch.compile: comma-separated subset of COMPILE_SAFE_STAGES (see tools/train.py), or "none".
compile_stages: "image_encoder,camera_unprojection,decoder"
compile_mode: "default"  # default | reduce-overhead | max-autotune -- only "default" is safe with >1 stage compiled
```

parsed manually in Python, never as a Hydra/OmegaConf list type:

```python
compile_stages = {
    s.strip() for s in str(cfg.get("compile_stages", "none")).split(",")
    if s.strip() and s.strip() != "none"
}
```

**Why a plain string instead of a YAML list**: Hydra's CLI override syntax treats a bare comma as
list syntax and rejects `compile_stages=image_encoder,decoder` as ambiguous. Keeping the field a
plain string sidesteps this entirely — at the cost of needing single-quoting when overriding more
than one stage from the CLI:

```bash
uv run tools/train.py compile_stages=\'image_encoder,decoder\'
```

Document this exact caveat in the config comment itself (both lineage repos do) — it's a real,
repeatedly-hit gotcha, not a one-off footnote.

Validate the requested set against a `COMPILE_SAFE_STAGES` allowlist (not `KNOWN_COMPILE_HOSTILE_STAGES`
directly — the allowlist is the *positive* list of what's been measured safe, which is a stronger
guarantee than just "not on the hostile list"):

```python
invalid = compile_stages - set(COMPILE_SAFE_STAGES)
if invalid:
    raise ValueError(f"compile_stages {invalid} not supported for training -- only {COMPILE_SAFE_STAGES} are safe.")
```

## The bf16-mixed precision field

Standard Lightning `trainer.precision: "bf16-mixed"` — not a custom convention, just note the exact
string (`"bf16-mixed"`, not `"bf16"`) since a literal-string mismatch here is the exact bug documented
in `references/case-studies.md`. If the repo's default radar/point encoder doesn't support bf16 (see
`references/precision-escape-hatches.md`), document the coupling explicitly in the config comment —
`tinycar-dev`'s convention: *"If you switch back to `module/radar_encoder=points_to_gaussians`, also
set `trainer.precision=32-true`."* Two config values that must move together deserve a comment saying
so, not a silent trap for the next person who changes one without the other.

## Non-negotiable: compile after loading checkpoint weights, never before

```python
# Load pretrained weights into the plain (uncompiled) module first -- torch.compile wraps
# compiled submodules in an OptimizedModule whose state_dict keys gain an "_orig_mod."
# segment, which won't match a checkpoint saved without compile applied. Compiling after
# loading avoids that mismatch; compiling doesn't change parameter values, only forward().
if cfg.get("pretrained_weights", None):
    load_partial_weights(module, cfg.pretrained_weights)

compile_stages = { ... }
if compile_stages:
    apply_compile(module.model, compile_stages, cfg.get("compile_mode", "default"))
```

This is a real, hit-in-production ordering bug (`gcarpred-dev`) — get the order right the first time
rather than debugging a checkpoint-loading failure that looks unrelated to compile at all (a partial
weight load with suspiciously low "tensors loaded" count is the actual symptom, not an explicit
compile-related error).

## The attribute-swap compile trick, one more time for config-time application

The same `setattr(model, attr, torch.compile(getattr(model, attr), **compile_kwargs))` pattern used
by the benchmark sweep applies unchanged at real training/inference time — `torch.compile` accepts
both `nn.Module` children and bound methods identically (see `references/stage-discovery.md`). No
separate mechanism needed between "benchmarking compile" and "production compile" — that's the whole
point of the sweep: it measures exactly the thing that gets wired into the real training entrypoint.
