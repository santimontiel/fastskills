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

### 2026-08-10/11 — bevpredformer-radar (two model variants, first BEVFormer-style deformable-attention target)

**Target repo state**: no compile/bf16 prior art at all (see `use-cases.md`'s existing entry for this
bucket). Two model variants share most of the network: `BEVPredFormerPredictor` (camera-only, has a
real checkpoint with a documented IoU/VPQ baseline) and `BEVPredFormerPredictorCamRadar` (adds a
`radar_encoder`/`fuser` pair, no checkpoint yet — random-init only). Both swept separately
(`model=BEVPredFormerPredictorCamRadar with_radar=True` selects the second). Also the first target in
this lineage with **no bound-method stages and no manually-indexed top-level containers at all** —
every stage in `forward()` is a genuine `nn.Module.__call__`, so `EXTRA_METHOD_STAGES` and
`CONTAINER_CHILD_STAGES` stayed empty. (One level *inside* `DefAttnVT.forward()`, several
`nn.ModuleList`s *are* manually indexed in a Python loop — the classic `CONTAINER_CHILD_STAGES`
shape — but this doesn't block treating `view_transform` as one atomic top-level stage, since its own
`forward()` is what gets called; only relevant if finer per-layer granularity is ever wanted later.)

**Escape hatches needed**: `view_transform` (`DefAttnVT`, wrapping `MSDeformAttn`/`MSDeformAttn3D`, a
vendored CUDA extension at `ops/defattn/`) hard-crashed under bf16 autocast:
`NotImplementedError: "ms_deform_attn_forward_cuda" not implemented for 'BFloat16'`. **Confirmed via
the CUDA source itself before even running anything** — `ops/defattn/src/cuda/ms_deform_attn_cuda.cu`
dispatches via `AT_DISPATCH_FLOATING_TYPES` (float/double only, no `_AND_HALF`/`BFloat16` variant).
Fixed with Pattern A, scoped tightly: `torch.autocast(device_type=..., enabled=False)` wrapping just
the `self.deformable_attention(...)` call inside `SADefnAttn.forward`/`CADefnAttn.forward`
(`bevpredformer/models/layers/attention.py`), with explicit `.float()` casts on every tensor argument
passed in (disabling autocast alone does NOT retroactively un-cast tensors already computed as bf16
upstream — the input tensors themselves needed casting too, not just the ambient context). One
follow-on dtype fix was needed in `CADefnAttn`: the fp32 output had to be cast back to the ambient
(possibly bf16) dtype before an in-place `+=` into a pre-allocated `bf16` accumulator, or it raises
a dtype-mismatch error — `SADefnAttn` didn't need this since its own residual add uses `+` (out-of-place,
happy to mix dtypes) not `+=`. **Verified by re-running the probe after the fix**: `view_transform`
went from `crashed` to `ok`, no NaN. `decoder` (`SparseUNet`, spconv-based) needed **no escape hatch
at all** — passed the bf16 probe cleanly despite being flagged `KNOWN_COMPILE_HOSTILE_STAGES` — see
the new use-cases.md entry below; this genuinely surprised the "spconv/PTv3 = no bf16" assumption
carried over from `tinycar-dev`/`gcarpred-dev`, and turned out to be a *compile*-only problem, not a
precision one.

**Decisions made** (clean, uncontended runs — see the methodological finding below for why "clean" is
load-bearing here): both variants agree on `backbone` (+47.2%/+45.0% fp32) and `neck`
(+20.6%/+21.6% fp32) as consistent, strong, low-variance wins — shipped as the shared default
(`compile_stages: "backbone,neck"` in `configs/train.yaml`/`configs/val.yaml`). `decoder` is a
consistent, severe regression in both (-56.4%/-84.8%, high variance from constant
`torch._dynamo` recompilation on spconv's data-dependent sparse shapes) — excluded. `projector`,
`query_gen`, `view_transform` are consistent regressions in both (cheap/small ops — compile dispatch
overhead exceeds any fusion gain) — excluded, no `AskUserQuestion` needed, unambiguous. `heads`
(+10.8%/+8.3%) and `temporal` (+12.0%/+9.7%) landed right at the ask-band boundary in BOTH variants,
consistently, with tight std (not noise) — asked via `AskUserQuestion`, user chose to exclude both
(marginal gain not worth added compile/startup latency). CAMRADAR-only: `radar_encoder` is a clean
win (+31.4%, ±6.8) — notably **not** pre-excluded the way every sibling repo's own radar encoder is
(see `use-cases.md`'s new entry below), and the sweep confirmed it was right not to assume; `fuser`
is a small, consistent regression (-6.1%) — excluded. CAMRADAR override documented in the config
comment: `compile_stages=backbone\,neck\,radar_encoder`. Combined whole-model benchmark (final,
shipped stage set, camera-only): +25.1% fp32 / +21.0% bf16.

**New methodological finding, not yet in any reference doc — GPU contention from an unrelated
concurrent job corrupts the sweep's timing numbers, even though it doesn't affect an accuracy-only
job's correctness**: a first pass at the camera-only sweep was run on a shared GPU already hosting
two long-running sibling-project containers, producing per-stage speedups as absurd as
`query_gen: +22824.5% (±26349.4)` and `neck: +1124.0% (±1497.9)` — several stages later confirmed
(clean re-run) to actually be **regressions**. Root cause: `run_sweep`'s per-stage "compiled" number
is a *derived delta* (`compiled_whole_ms − eager_whole_ms + eager_stage_ms`, see
`StageLatencyProbe`'s docstring), and for a cheap/fast stage this delta is the difference of two large,
noisy numbers — when a concurrent process is stealing GPU cycles unpredictably, that noise floor can
dwarf the tiny stage's true signal, producing nonsense percentages with the sign essentially random.
A **second**, self-inflicted instance of the same problem happened mid-session: the CAMRADAR sweep was
accidentally launched concurrently with an unrelated `tools/val.py` accuracy-comparison run (val.py's
own *output* — IoU/VPQ — is unaffected by contention, since accuracy is deterministic regardless of
wall-clock speed, but the **sweep's** output is nothing but wall-clock timing, so it was corrupted
by the very same mechanism). **Both times, the fix was the same: kill everything else on the GPU and
re-run the sweep alone.** `run_sweep_repeated`'s existing `num_repeats` averaging does NOT fix this —
it did nothing to save the first contended run, since all 3 repeats were contended identically (not
independent noise that averages out, but a shared confound biasing every repeat the same direction).
**Practical rule going forward**: before trusting sweep numbers, `nvidia-smi`/`docker ps` should be
used to confirm nothing else is running on the target GPU — not just once before starting, but ideally
spot-checked again after, since another process can start mid-sweep. The combined benchmark is
somewhat more robust to brief contention (it's 1-3 direct measurements, not a chain of deltas) but is
not immune either — treat it with the same "was anything else running" scrutiny.

**Follow-ups**: neither variant's `compile_stages` default was tested with DDP (`trainer.devices>1`)
— only `trainer.devices=1` was exercised, matching this repo's own single-GPU verification convention
(`num_workers` capped at 2). The CAMRADAR variant's `compile_stages` recommendation is based on
random-init weights only (no checkpoint exists yet) — its accuracy-comparison leg of step 9 could only
check "loss stays finite, roughly consistent with an earlier uncompiled smoke-test run," not a real
IoU/VPQ baseline; re-verify once a CAMRADAR checkpoint exists.

### 2026-08-30 — rapidradar-dev (RapidLiDAR: LiDAR scene completion; first nested-stage target)

**Target state**: no compile/bf16 prior art. Run in the same session as the `/repo-adapter` port, so
the model had just been split into `model.py`/`module.py`. Hardware: RTX 5090 (sm_120), torch
2.8.0+cu129. Note the host's NVIDIA driver was broken all session (kernel module 595.71.05 vs
userspace 595.84, so `nvidia-container-cli` failed and no new GPU container could start); all GPU
work ran via `docker exec` into a container started *before* the upgrade, which still held matching
driver libs. Worth remembering as a workaround — and as a hazard, since stopping such a container is
irreversible until the driver is fixed.

**Measured numbers** (trained HF weights, batch 1, 18000 input points → 180000 queries, num_iters=30,
num_repeats=3, idle GPU):

| stage | fp32 | bf16 | decision |
|---|---|---|---|
| `voxel_encoder` | **+36.8%** (±0.8) | **+37.6%** (±0.6) | default |
| `bev_projections` | **+36.8%** (±0.4) | **+46.2%** (±0.9) | default |
| `fe_extract` (bound method) | +8.0% (±0.5) | +7.6% (±0.2) | skip |
| `reconstruction` | +2.3% (±0.5) | +1.9% (±0.2) | skip |
| `adaptive_init` | +1.8% (±4.7) | −3.8% (±6.1) | skip |
| `feature_proj` | −0.1% (±1.4) | −1.9% (±0.5) | regression |
| `bev_head` | **−39.6%** (±6.1) | **−48.8%** (±12.6) | regression |

Nothing landed in the 10–20% ask band, so no `AskUserQuestion` was needed. Shipped
`compile_stages: "voxel_encoder,bev_projections"`.

Combined whole-model (both stages together, `mode="default"`): fp32 57.836→49.044 ms/sample
(17.3→20.4 Hz, **+15.2%**); bf16 59.736→50.239 ms (16.7→19.9 Hz, +15.9%). Peak memory essentially
unchanged (3590→3590 MB fp32). Note again how much smaller the honest whole-model number is than the
per-stage figures suggest.

**Escape hatches needed**: `reconstruction` and `refine_model`'s attention, both wrapping the
vendored `MultiScaleDeformableAttention` kernel. Pattern A, exactly as `bevpredformer-radar`'s
`view_transform`: the `.cu` source's `AT_DISPATCH_FLOATING_TYPES` predicted the crash before running
anything, and the probe reproduced it verbatim (`NotImplementedError: "ms_deform_attn_forward_cuda"
not implemented for 'BFloat16'`). Fixed with `torch.autocast(enabled=False)` plus explicit `.float()`
on every tensor arg and a cast back to the caller's dtype. Verified the fix left fp32 **bit-identical**
(sha256 digest of a fixed-seed forward pass), then re-probed: crashed → ok. The chamfer op is
*also* fp32-only but needed nothing — its wrapper already casts `.float()` on the way in.

**bf16 is measured but NOT adopted — the important finding.** After the hatch, bf16 runs clean with
no NaN/Inf, and the sweep shows it is marginally *slower* whole-model than fp32 eager
(59.7 vs 57.8 ms). More importantly, on **trained** weights a bf16 forward diverges from fp32 by a
chamfer distance of **0.338 m** — the same order as the metric the model reports. Per-point
differences (mean 23 m) are a red herring: the network emits an *unordered* cloud, so point identity
is meaningless and only a set-level metric is informative. The lesson worth carrying: **for a model
whose output is an unordered set, never judge bf16 fidelity by elementwise diff — use the task
metric.** `trainer.precision` stays `"32-true"` pending a real eval against a dataset baseline.

**Three template bugs found and fixed** (all upstreamed into `use-cases.md`): hooks leaked whenever a
pass raised (`probe.remove()` outside a `finally`), producing a misleading first-failure several rows
downstream; nested stages did not exclude their descendants from hooking; and my own first attempt at
a bound-method-safe restore used `__dict__.pop` for *all* stages, which silently left `nn.Module`
stages compiled (children live in `_modules`, not `__dict__`) so every later stage was measured
against a partly-compiled model.

**Follow-ups**: `bev_head`'s −40%/−49% is the largest regression seen in this lineage — attention over
only 41×41=1681 tokens, so dispatch overhead swamps fusion. Worth checking whether a bigger BEV grid
flips it. The refinement network was not swept at all: its gradient-checkpointing and chunking
branches key off `self.training`/`requires_grad`/chunk size, so expect far more compile hostility.
And every number here is on synthetic uniform input — re-run once SemanticKITTI is available, since
`voxelize()`'s occupancy pattern (and hence the 3D UNet's cost) is data-dependent.
