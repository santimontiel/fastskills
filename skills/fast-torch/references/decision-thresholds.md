# Decision thresholds

Speedup for a stage is computed as:

```
speedup_pct = (eager_ms - compiled_ms) / eager_ms * 100
```

Positive = faster compiled. Negative = a regression (compiled is slower — this happens, and it's a
real, reported result, not a measurement error).

## The bands

| Measured speedup | Decision | Why |
|---|---|---|
| **> 20%** | Enable by default, no confirmation needed | Large enough that it's not noise, and the compile-time/complexity cost is clearly worth it |
| **10% – 20%** | `AskUserQuestion` — presenting the real number | Real but marginal; whether it's worth the added complexity (debugging a compiled module is harder — breakpoints inside it get skipped over) is a judgment call, not a fixed rule |
| **< 10%, or a regression, or a crash** | Leave eager | Not worth the complexity for a marginal or negative return |

A **crash** on a `KNOWN_COMPILE_HOSTILE_STAGES` entry (or an undeclared one) is its own bucket, not
folded into "< 10%" — report it as "compile-hostile: `<exception>`", not as a numeric zero.

**`compile_mode` other than `"default"`** (`reduce-overhead`, `max-autotune`) is a separate axis from
this table — it's confirmed to crash with CUDA-graph aliasing once more than one stage is compiled
together (independently confirmed on both `tinycar-dev` and `gcarpred-dev`: *"accessing tensor output
of CUDAGraphs that has been overwritten by a subsequent run"*). Only sweep `compile_mode` variants
per-stage in isolation if genuinely curious — never adopt a non-`"default"` mode for a multi-stage
combination without re-verifying that specific combination crash-free on the target repo.

## Worked calibration numbers (real, from the lineage — not made up)

These are the actual measured results that produced each repo's current defaults. Use them to sanity
-check whether a new measurement on a similar architecture is in the right ballpark — not as a
substitute for actually running the sweep on the target repo.

- **`fiery-radar`** (commit `0a04263`): `bev_projection` ~1.8–2.3x speedup (>>20% band — replaces a
  Python sort/mask/cumsum/scatter voxel-summing loop with a fused Inductor kernel), `encoder`
  ~2.1–2.4x (>>20% band, ~34% less peak memory too). Both enabled by default. `radar_encoder` left
  **off** by default even though compile-eligible — not a speed problem, a *correctness-risk* one:
  variable point counts per batch mean the op sees a new shape almost every step, defeating
  compilation's whole premise (see the config comment: *"unlike the encoder/BEV-projection paths'
  fixed shapes"*).
- **`powerbev-radar`** (commit `75103b9`, RTX 5090): `bev_projection` **13–14x** (dispatch-overhead-
  bound, not compute-bound — many small sequential ops), `encoder` **2.1–2.2x**. Same two-module
  conclusion as `fiery-radar`, reached independently.
- **`gcarpred-dev`** (RTX 5090, fp32, batch=1): seven stages combined for **-18.2%** whole-model
  latency (123.2ms → 100.8ms) fp32, **-45.9%** combined with bf16-mixed (123.2ms → 66.7ms). Per-stage
  range **-2.9% to -46.1%** among the kept stages. Two real exclusions worth noting as *negative*
  calibration: `scene_context` was **+166%** (a genuine regression — compiled dispatch overhead
  dominating a small latent-attention op) and `aux_seg_head` was **+1.4%** (in the "not worth it"
  band, correctly excluded even though technically compile-eligible and crash-free).
- **`tinycar-dev`** (`docs/efficiency-frontier.md`): `image_encoder` **-23%** (>20% band, enabled),
  `decoder` **-15%**, `camera_unprojection` **-12%** (both in the 10–20% ask-user band — both ended up
  enabled, meaning the user was asked and said yes for both). Regressions: `fuser` **+8%**,
  `aux_head` **+15%**, `gs_render_image` **+29%** — all correctly excluded from the default.

The pattern across all four: **encoder-style stages and stages replacing many small sequential ops
with a fused kernel are the reliable big wins; small per-item ops (attention on a short sequence,
lightweight aux heads) are the reliable regressions.** This isn't a shortcut around actually running
the sweep, but it's a reason to be suspicious of a measurement that contradicts it — e.g. if an
encoder measures as a regression, double-check the measurement (warmup iterations, GPU contention
from another job) before trusting it.
