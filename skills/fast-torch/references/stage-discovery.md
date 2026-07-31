# Stage discovery

"Stage" = one top-level piece of the model that gets its own eager-vs-compiled, fp32-vs-bf16
measurement. Getting the stage list right matters more than any other single step — every downstream
measurement is per-stage.

## What's automatic: `model.named_children()`

Every top-level `nn.Module` attribute assigned in the model's `__init__` shows up here, in
assignment order, with zero repo-specific code needed:

```python
stage_names = [name for name, _ in model.named_children()]
```

This is enough to auto-discover, e.g., `image_encoder`, `radar_encoder`, `fuser`, `decoder`, `head` —
the common shape across `tinycar-dev`/`gcarpred-dev`. Forward hooks (`register_forward_hook`,
`register_forward_pre_hook`) attach to these automatically too, which is what lets
`assets/compile_bf16_sweep.py` do both the bf16 probe and the latency timing **without**
hand-duplicating the model's `forward()` control flow into a parallel `forward_with_timing()` the way
every existing lineage `tools/benchmark.py` does. That duplication is a real, accepted cost in those
tools (each carries an explicit "keep in sync with `forward()`" comment) — this tool's hook-based
design avoids it for anything that's a real `nn.Module` child.

## What isn't: bound-method stages

Some stages aren't `nn.Module` children at all — they're plain bound methods called directly inside
`forward()`. Confirmed real examples across the lineage:

- `tinycar-dev`/`gcarpred-dev`'s `camera_unprojection` stage is actually the bound method
  `model._pred_depth`.
- `gcarpred-dev`'s `gaussian_evolution` stage is three bound methods together
  (`_evolve_covariances`, `_evolve_opacities`, `_evolve_features`).
- `gcarpred-dev`'s `query_init` stage resolves to *one of* `_grid_sample_query_init` /
  `_knn_gaussian_query_init` / `_hybrid_query_init`, depending on a config value
  (`model.query_init_mode`) — the attribute to compile isn't even fixed, it's picked at runtime.

`named_children()` cannot find these — a bound method isn't a submodule, and no forward hook fires
on it independently of whatever `nn.Module` it happens to call internally. **These have to be added
by hand** to the tool's `EXTRA_METHOD_STAGES` mapping (name → attribute name, or a callable that
resolves the attribute name from the model's own config, for the `query_init`-style case), following
the same attribute-swap pattern used everywhere else in this lineage:

```python
model.some_attr = torch.compile(model.some_attr, **compile_kwargs)
```

`torch.compile` accepts a bound method exactly as it accepts an `nn.Module` — the swap trick works
identically either way. The only cost is that discovery isn't automatic: read the model's actual
`forward()` once, note every bound-method call that looks expensive enough to be worth measuring, and
add it explicitly. Don't skip this step just because it's manual — on `gcarpred-dev`,
`camera_unprojection` and `gaussian_evolution` were both real, measured wins (part of the seven-stage
default), and neither would show up in a `named_children()`-only sweep.

## What also isn't automatic: a container child whose own forward() is bypassed

Some top-level `nn.Module` children ARE found by `named_children()` — no discovery gap there — but
are never invoked as a whole by the model's own `forward()`. Confirmed real example: `JustDepth`'s
`graph_backbone`, an `nn.Sequential(*[Seq(Grapher, FFN) for _ in range(n_blocks)])` stack, called as:

```python
for i in range(self.n_blocks):
    x = self.graph_backbone[i](x)
```

`self.graph_backbone` itself (the container) never has `__call__` invoked on it — only its indexed
children do. This breaks both halves of the normal hook-based approach:
- A forward hook registered on `model.graph_backbone` never fires, so it silently measures nothing
  (not an error — easy to miss unless you cross-check discovered stages against the actual
  `forward()` body, not just `named_children()`'s output).
- Compiling and swapping back `model.graph_backbone` as a single attribute would break `forward()`'s
  `self.graph_backbone[i]` indexing, since `torch.compile`'s `OptimizedModule` wrapper doesn't proxy
  `__getitem__`.

**Handled by treating it as a multi-module stage**, declared in `CONTAINER_CHILD_STAGES` (name →
container attribute name):

```python
CONTAINER_CHILD_STAGES = {"graph_backbone": "graph_backbone"}
```

`discover_stage_modules()` expands this into `[container[i] for i in range(len(container))]` — every
child hooked individually, rolled up into one logical stage name (their latencies summed per forward
pass, since each child is invoked exactly once). `compiled_stage()` compiles and reassigns each child
in place (`container[i] = torch.compile(container[i])`) rather than replacing the container attribute
itself. See `use-cases.md`'s matching catalog entry and `case-studies.md`'s 2026-07-30 JustDepth entry.

## Known-hostile stages: flag before running, don't discover by crashing blind

Every lineage repo's benchmark tool pre-declares which stages wrap vendored CUDA ops with no
fake-tensor/meta kernel — `torch.compile`/Dynamo is expected to graph-break or hard-crash on these,
and that's treated as a legitimate, informative result, not a tool bug:

```python
KNOWN_COMPILE_HOSTILE_STAGES = ("radar_encoder", "gs_render_image", "gs_render_radar")
```

(exact names vary per repo — `radar_encoder` wraps spconv/PTv3, `gs_render_*` wraps the vendored
`diff_gaussian_rasterization` extension). Declare the target repo's own equivalent list up front by
reading `ops/`/vendored-extension usage (see the sibling `repo-adapter` skill's package-restructuring
guidance for how vendored code is organized) — this isn't required for correctness (the sweep's
try/except handles an undeclared crash fine) but it turns a crash from "is this a bug?" into "yes,
expected" immediately in the report, saving a debugging detour.
