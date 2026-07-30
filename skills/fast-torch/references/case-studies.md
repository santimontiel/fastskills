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
