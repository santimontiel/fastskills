# Case studies (Self-Improve log)

Append-only, same discipline as the sibling `repo-adapter` skill's `case-studies.md`: read this near
the start of a new invocation, append a new entry at the end of every invocation (even a clean one —
"nothing unusual" is still signal).

## Entry template

```
### YYYY-MM-DD — <target repo>

**Measured numbers**: which stages, what speedup %, which precision(s).

**Escape hatches needed**: which stages needed a bf16 fix, which pattern (A or B, see
precision-escape-hatches.md) fixed it, and how it was verified.

**Decisions made**: which stages auto-enabled (>20%), which were asked about (10-20%) and what the
user chose, which were left eager and why.

**New gotchas discovered**: anything not already covered by these reference docs.

**Follow-ups**: anything deferred, or a suspicion worth checking on a future similar repo.
```

## Entries

### 2026-07-30 — self-review of assets/compile_bf16_sweep.py before its first real use

**Measured numbers**: none — this entry predates any real invocation; it's a code-review pass on the
template itself, done because it was never executed against a real GPU/model.

**New gotchas discovered** (fixed where fixable, flagged where not):
- The bf16 probe's output-capturing hook originally only handled `Tensor`/`tuple`/`list` outputs. A
  stage returning a **dict** of named tensors (the actual, common shape for an `image_encoder` stage
  across this lineage) silently produced "not observed" — false confidence on exactly the modules most
  likely to need a fix. Fixed via a recursive `_iter_tensors()` walk over nested dict/list/tuple output.
- The same hook only kept the *last* captured tensor per stage, so a stage invoked more than once per
  forward (e.g. a per-timestep render call inside a `DT_STEPS`-style loop, confirmed real in
  `gcarpred-dev`) could have an earlier NaN masked by a clean final call. Fixed via an "any call, any
  NaN" accumulator instead of "last call only".
- `build_synthetic_batch` was originally called separately per precision, so the fp32 and bf16 passes
  compared against different random inputs. Fixed by building the batch once and reusing it.
- **Not fixed, flagged as an open risk**: whether `torch.compile` reliably preserves Python-level
  forward hooks across torch versions is asserted in a code comment as "reasonable, not verified" — if
  a compiled stage's probe/timing numbers come back identical to eager, check this first.
- **Not fixed, structural limitation**: the bf16 probe runs on synthetic random data. The real
  historical TF32/`cdist` bug (see the entry above) needed a specific self-distance geometry that
  `torch.randn` may never reproduce — a clean probe result is evidence, not proof, for that class of bug.

**Follow-ups**: the first real invocation of this tool should specifically watch for whether the
compile+hooks interaction behaves as assumed, and should not treat a clean bf16 probe as license to
skip watching real training logs for NaN/Inf per `repo-adapter`'s verification discipline.

### (retroactive) — cross-repo gotchas already known before this skill's first real invocation

These aren't from one single invocation of this skill — they're the real incidents across the
lineage that this skill was built to stop happening a fifth time. Recorded here so they're not
rediscovered.

**The TF32/`torch.cdist` NaN bug** (`tinycar-dev`, independently also found in `gcarpred-dev`'s
`pillar_encoder.py`): enabling `torch.set_float32_matmul_precision("high")` (TF32) alongside a
`torch.cdist`-based pairwise distance computation made `cdist`'s internal identity
`d² = |x|² + |y|² − 2⟨x,y⟩` return pure rounding error for self-distance — at nuScenes' ±50m extent,
enough error (~2.9m) to push points outside their own ball-query radius, producing fully-masked
softmax rows → NaN across 100% of the affected Gaussians. The NaN then failed a downstream
`opacities > threshold` test, so the renderer silently received **zero** Gaussians and the model
trained camera-only with a finite, plausible-looking loss — **undetected through a full 3-epoch run**.
Fix: a `_full_fp32_matmul()` context manager around just the `cdist` call, plus an identity-mask OR
against the degenerate case. **This is not a bf16-autocast bug at all — it's a TF32 bug** — meaning a
bf16-compatibility probe alone would NOT have caught it; if the target repo uses `torch.cdist` (or any
pairwise-distance op) anywhere, explicitly check its behavior under
`torch.set_float32_matmul_precision("high")` too, not just under bf16 autocast.

**The dead `"bf16"` vs `"bf16-mixed"` string-match bug** (`gaussiancar-dev`, fixed in `gcarpred-dev`'s
commit `30102b6`): a blanket batch→bf16 cast in the training step was gated on
`if self.cfg.trainer.precision == "bf16":`, but Lightning's actual precision string for mixed training
is `"bf16-mixed"` — the check never fired, so the code was silently inert. Harmless by luck (it would
have blindly cast label/integer tensors too if it had ever matched) — but a reminder to grep every
literal precision-string comparison when adapting a precision code path, not just assume the author's
string matches what the framework actually produces.

**PTv3/spconv has no mixed-precision backward pass at all** (`gaussiancar`): the repo's own
`docs/getting_started.md` states training is FP32-only *specifically* because the PTv3 radar encoder's
backward pass doesn't support bf16/fp16. This is the actual root motivation behind `gcarpred-dev` and
`tinycar-dev` both defaulting to the alternative `PillarsToGaussians` radar encoder (a plain-PyTorch
implementation with no fp32-only ops) instead — sometimes the real fix for "this module fails the bf16
probe" isn't an escape hatch at all, it's swapping the module for a bf16-native alternative
implementation, if one exists or can be built. Don't assume every probe failure needs Pattern A/B from
`precision-escape-hatches.md` — check whether an alternative implementation already exists in the
repo/lineage first.

**CUDA-graph aliasing crash on multi-stage `reduce-overhead`/`max-autotune`** (independently confirmed
on both `tinycar-dev` and `gcarpred-dev`): `compile_mode` values other than `"default"` crash with
*"accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"* once more than
one stage is compiled together. `"default"` is the only mode confirmed safe for a multi-stage
combination — see `decision-thresholds.md`.

**Compile-after-checkpoint-load ordering** (`gcarpred-dev`): `torch.compile` wraps a module in
`OptimizedModule`, prefixing its state-dict keys with `_orig_mod.` — compiling before loading a
checkpoint silently breaks key matching (symptom: suspiciously few tensors loaded, not an explicit
error). Always compile after loading weights. See `hydra-wiring.md`.

**bf16 tensors crashing NumPy/preview-logging code, not the model itself** (confirmed recurring
across repos in this lineage per direct user report — happened again on JustDepth, 2026-07-30, see
that entry below, and per the user "more than once" before this): a training loop's periodic preview/
visualization logging calls `.detach().cpu().numpy()` on the model's raw output tensors, which are
bfloat16 under bf16-mixed autocast — NumPy has no native bfloat16 dtype, so this is an unconditional
hard crash (`TypeError: Got unsupported ScalarType BFloat16`), not a numerical issue. The bf16
-compatibility probe does NOT catch this (it only exercises `forward()`, never any downstream logging
code) — this class of bug is only ever caught by SKILL.md step 9's real training run, which is exactly
why that step is the acceptance gate and not optional. Fix: `.float()` right before the `.cpu()`/
`.numpy()` boundary on the model's own output tensors specifically (not on raw dataloader inputs,
which are already fp32 and untouched by autocast). See `precision-escape-hatches.md`'s new Pattern C
for the full writeup and a checklist of what to grep for pre-emptively.

### 2026-07-30 — JustDepth (first real GPU invocation of assets/compile_bf16_sweep.py)

**Target repo state**: no compile/bf16 prior art at all (already catalogued in `use-cases.md`), no
`train_log/` checkpoint in the new Lightning format — only pre-Lightning-port `{network, clock,
optimizer, lr_scheduler}`-format checkpoints existed (`train_log/models/latest.ckpt`). Converted one
(`latest_converted.ckpt`) by re-keying `network.*` state dict entries to `model.*` (confirmed an exact
650/650 key match against a freshly-constructed `JustDepthModule`, i.e. the architecture hadn't
drifted since the pre-port checkpoint was saved) and synthesizing a minimal Lightning-loadable dict
(`state_dict`, `pytorch-lightning_version`, `epoch`, `global_step`). Worth checking for on any repo
mid-migration to a new training-harness format before assuming "no usable checkpoint."

**Measured numbers** (RTX 5090, batch=1, nuScenes 896x1600, fp32 vs bf16-mixed, num_iters=50):
all 5 discovered stages (`radar_encoder`, `image_encoder`, `fusion_block`, `graph_backbone`,
`depth_decoder`) cleared the >20% default-enable threshold at **both** precisions — fp32: +24.0%,
+23.1%, +49.3%, +51.5%, +20.4% respectively; bf16: +22.9%, +53.1%, +44.6%, +49.3%, +36.6%. No stage
fell in the 10-20% ask band, so no `AskUserQuestion` was needed. bf16 probe: "ok" for all 5 stages
(no NaN/Inf on the synthetic batch). See `dev/results.html` / `dev/sweep_results.csv` in the target repo.

**Escape hatches needed**: none — every stage's forward pass is plain conv/batchnorm/GELU/attention/
matmul-topk-based GNN ops, nothing resembling PTv3/spconv or a vendored rasterizer.

**Decisions made**: `compile_stages` set to all 5 stages by default in both `configs/eval.yaml` and
`configs/train.yaml`. `trainer.precision` for training left at `"32-true"` despite bf16 also passing
cleanly — deliberately NOT flipped to `"bf16-mixed"`, since only a forward-pass probe and an
inference-only `eval.py` comparison were run this session, not a full training epoch verifying
gradient-stability under bf16 over time. Documented as an available, not adopted, option.

**New gotchas discovered** (both fixed in `assets/compile_bf16_sweep.py` itself, not just worked
around in the target repo's adapted copy):

1. **Hooking the current torch.compile target corrupts its own timing.** The previous entry below
   flagged this as an open, unverified risk ("reasonable expectation... not verified on a GPU"). Now
   confirmed wrong in a specific, serious way: `OptimizedModule.__call__` invokes the wrapped module's
   `__call__` (hooks included), so Dynamo traces the pre/post hook's own side effects (a Python list
   append changing length every call, then a `torch.cuda.Event.elapsed_time` call) as part of the
   compiled graph. Symptom for 4 of 5 stages: silently wrong numbers (garbage negative-tens-of-
   thousands-percent "speedups", e.g. `radar_encoder` reporting -46369.6%). Symptom for the 5th
   (`graph_backbone`, compiled as 8 separate child modules): Dynamo hit its recompile limit
   (guard `len(self.samples_ms[...]) == N` changing every call) and then crashed outright with
   `RuntimeError: expected other to be a torch.Event object`. **Fixed** by never hooking the stage
   currently wrapped in `torch.compile`; its compiled latency is instead derived from the whole-model
   wall-clock delta (`compiled_whole_pass_ms - eager_whole_pass_ms + eager_stage_ms`), measured via
   CUDA events wrapping the entire `model(batch)` call from outside any hook. This is now the *only*
   path for every stage's compiled number (previously this fallback was believed needed only for
   bound-method stages that can't be hooked at all — it turns out hookable stages need it too, the
   instant they're the compile target).
2. **A top-level `nn.Sequential`/`nn.ModuleList` child whose own `forward()` the parent bypasses**,
   calling `self.<attr>[i](x)` in a manual Python loop instead of `self.<attr>(x)` as a whole
   (JustDepth's `graph_backbone`, an 8-block GNN stack). A hook on the container itself never fires
   (its `__call__` is never invoked), and compiling+swapping the container attribute would break the
   `[i]` indexing (`OptimizedModule` doesn't proxy `__getitem__`). New `CONTAINER_CHILD_STAGES`
   extension point added (alongside `EXTRA_METHOD_STAGES`) — see `stage-discovery.md`'s new section
   and `use-cases.md`'s new catalog entry.
3. **`thop.profile()` (or any per-module hook-registering FLOPs/param counter) breaks if the model is
   already partially compiled.** `justdepth/engine/eval.py` calls `thop.profile(net, ...)` right after
   loading the checkpoint, before the eval loop. A compiled submodule is enumerated twice by
   `net.modules()` — once as itself, once via the `OptimizedModule` wrapper's attribute delegation —
   so `thop`'s per-module buffer registration (`total_ops`, `total_params`) raised `KeyError:
   "attribute 'total_ops' already exists"`. Fixed by moving `apply_compile_stages()` to strictly after
   the `thop.profile()` call, not just after checkpoint loading. General lesson: "compile after
   checkpoint load" isn't the only ordering constraint — anything else that walks `model.modules()`
   and mutates/registers per-module state (profilers, some quantization/pruning tooling) should also
   run before compile is applied, not just before or interleaved with it.

**Follow-ups**: DDP + `resume_ckpt_path` + `compile_stages` together was NOT re-verified in this
session (only single-GPU `eval.py` was exercised) — flagged explicitly in `configs/train.yaml`'s own
comment. `graph_backbone`'s GNN ops (`pairwise_distance` in `gcn_lib/torch_edge.py`) compute self-
distances via `x_square + x_inner + x_square.T` (matmul-based), structurally the same shape as the
historical TF32/`cdist` NaN bug above — `torch.set_float32_matmul_precision("high")` was deliberately
NOT enabled in the target repo's production configs (only inside the benchmarking tool itself, for a
fair compiled-speed comparison) specifically because of this resemblance; a future invocation touching
TF32 defaults on this repo should re-check `graph_backbone`'s KNN neighbor selection accuracy first.

**Addendum, same day — step 10 (combined whole-model benchmark) added and exercised**: the user asked
directly for "old and new inference time in bf16 if all is compiled" after the above sweep, which the
per-stage isolated grid literally cannot answer (it never compiles more than one stage together). This
gap is why step 10 and `whole_model_benchmark`/`run_combined_benchmark`/`compiled_combined` were added
to the skill itself, not just answered ad hoc for this repo. Measured on JustDepth (whole model, real
checkpoint weights, all 5 stages compiled together, `compile_mode="default"`):

| Precision | Eager | Compiled | Speedup |
|---|---|---|---|
| fp32-true | 6.764 ms/sample (147.8 Hz) | 4.502 ms/sample (222.1 Hz) | +33.4% |
| bf16-mixed | 6.066 ms/sample (164.9 Hz) | 3.312 ms/sample (302.0 Hz) | +45.4% |

No CUDA-graph aliasing crash with all 5 stages compiled together under `"default"` mode — consistent
with `tinycar-dev`/`gcarpred-dev`'s finding that only `"default"` is safe for >1 combined stage.

**New gotcha discovered**: individual per-stage sweep classifications are noisy near the 20%/10%
bands run-to-run on the same idle GPU — `radar_encoder` measured +24.0% on the first sweep, then
+13.8% and +19.3% on two immediate repeats, flipping its classification between "default" and "ask"
each time. The combined benchmark was therefore given its own explicit `compile_stages` override
(`configs/fast_torch.yaml`'s `compile_stages` field, distinct from the sweep's own per-run
`default_stages` classification) so it measures the actually-shipped, already-verified stage set
rather than re-deriving a possibly-different set from whatever this particular run classified. See
`decision-thresholds.md`'s new note on this.

**Also new**: the combined benchmark's headline numbers are surfaced as stat-tile cards at the very
top of the HTML report, below the title, per the dataviz skill's stat-tile contract (label / value /
signed delta, color = direction × whether up is good). Requested explicitly by the user after seeing
the numbers reported as plain text in conversation — the lesson: a report's headline number belongs
where a reader looks first, not buried in the last section.

**Second addendum, same day — both precisions as cards, and a `num_repeats` loop for stable numbers**:
the user asked for two more things after seeing the first version: (1) show fp32 cards too, not just
the bf16-preferred single group, and (2) actually run the sweep in a loop to get more stable numbers,
picking up on the run-to-run variance noted just above rather than leaving it as an unaddressed
caveat. Both are now permanent parts of the skill, not one-off answers:

- `render_stat_cards` now renders one `_stat_group` of 4 cards per precision that produced a real
  comparison (fp32-true first, then bf16-mixed), instead of picking a single "best" precision.
- `run_sweep_repeated` wraps `run_sweep()` `num_repeats` times, aggregating each (stage, precision)
  cell's speedup as mean ± stdev via Python's stdlib `statistics` module and re-classifying from the
  MEAN, not any single run. `run_combined_benchmark` gained the same `num_repeats` parameter, but
  cheaper: `torch.compile` is entered ONCE per precision and only the timed forward-pass loop repeats
  inside that single compiled context (recompiling per repeat would be pure waste — compilation
  doesn't vary run to run the way GPU clock/thermal state does). Both new `configs/fast_torch.yaml`
  fields (`num_repeats: 3`, `compile_stages: none`) were added alongside the tool.

Final, 3-repeat-mean numbers on JustDepth (replacing the single-run numbers in the entry above — same
whole-model, all-5-stages-compiled-together configuration):

| Precision | Eager | Compiled | Speedup |
|---|---|---|---|
| fp32-true | 6.489 ± 0.22 ms/sample (154.2 ± 5.2 Hz) | 4.314 ± 0.16 ms/sample (232.0 ± 8.6 Hz) | +33.5% |
| bf16-mixed | 5.832 ± 0.03 ms/sample (171.5 ± 0.8 Hz) | 3.204 ± 0.10 ms/sample (312.3 ± 9.8 Hz) | +45.1% |

**New methodological finding**: the combined benchmark's stdev is small relative to its mean (fp32
eager ±3.4%, compiled ±3.7%) because it times the whole model directly. `run_sweep`'s per-stage
numbers are far noisier in comparison — over the same 3 repeats, per-stage fp32 speedups ranged from
±3.5 points (`graph_backbone`, stable) to ±39.2 points (`fusion_block`) of their own mean — because
each per-stage number is a *derived delta* (`compiled_whole_ms − eager_whole_ms + eager_stage_ms`,
see `StageLatencyProbe`'s docstring for why it can't be measured directly), which compounds noise
from three separate timing measurements instead of one. Trust the combined benchmark's numbers over
the per-stage sweep's for anything precision-sensitive; use the per-stage sweep only for the
auto/ask/skip *decision*, and even then only after `num_repeats` averaging.

**Third addendum, same day — a real training run, a real bug, and `trainer.precision="bf16-mixed"`
adopted as the default**: the user asked directly for a monitored 3-epoch training run (nuScenes,
`loader.num_workers=4`, all 5 `compile_stages`, `trainer.precision=bf16-mixed`) with explicit
instructions to watch memory carefully and cut the run if it approached OOM. This is the step 9
"Verify" gate actually exercised for real, not just described — and it earned its place in the
workflow immediately: the first attempt crashed at ~step 150 with `TypeError: Got unsupported
ScalarType BFloat16` inside `JustDepthModule._log_previews`, exactly the Pattern C bug documented
above, previously only a probe-level theoretical gap. Memory was never the problem (GPU peaked at
just 33%, host RAM never dropped below ~11GB free) — the crash was purely the bf16/NumPy
incompatibility, caught only because a real training step actually ran the preview-logging code path
for the first time this session. Fixed with `.float()` casts (see Pattern C), re-ran the same 3-epoch
config end-to-end: **completed cleanly**, loss 9.21 → 3.41, MAE/RMSE 8.03/15.01 → 2.71/6.16, no
NaN/Inf, peak GPU memory 34% of a 32GB card (11,044/32,607 MiB), minimum host free RAM 13.5GB, total
wall time ~1h14m for 10,548 steps. On the strength of this real result, `trainer.precision` in
`configs/train.yaml` was flipped from `"32-true"` to `"bf16-mixed"` as the shipped default — the
earlier entries in this file deliberately withheld that change pending exactly this kind of
end-to-end evidence; this is what "worth trying explicitly, not adopted silently" (see above) becoming
"adopted" actually looks like in practice.

**Process note**: a memory-safety wrapper script (bash loop polling `nvidia-smi`/`free -m` every 10s,
killing the training process if GPU crossed 90% or host free RAM dropped below 1.5GB) was written
ad hoc for this session rather than being part of the skill's own tooling — the fast-torch skill's
scope is compile/precision benchmarking and verification, not general training-job memory-safety
supervision. Worth a note here in case a future invocation is asked for the same "watch RAM, cut if
critical" request again: the pattern (background bash loop, one Bash `run_in_background` call
wrapping both the training launch and the monitor loop so there's a single completion notification)
worked cleanly and needs no skill-level change, just re-deriving the same script.
