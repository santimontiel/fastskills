---
name: fast-torch
description: Introduces torch.compile + bf16-mixed precision into a research repo's network for
  faster training/inference with no accuracy regression, via a per-module bf16-compatibility probe,
  a full precision x compile sweep, and threshold-driven auto/ask/skip decisions wired into Hydra
  config using the family's comma-separated compile_stages convention. Use this whenever the user
  explicitly runs /fast-torch, or asks to "add torch.compile", "enable bf16", "speed up
  training/inference", "mixed precision", "compile sweep", or "make this repo faster without losing
  accuracy". Do not trigger automatically just because a repo looks slow — this is a heavy,
  GPU-bound, highly interactive workflow that must be explicitly requested.
disable-model-invocation: true
user-invocable: true
---

# fast-torch

## Purpose

Codify the repeatable process — already run independently on `tinycar-dev`, `gcarpred-dev`,
`fiery-radar`, and `powerbev-radar` — of speeding up a research repo's network via `torch.compile`
and bf16-mixed precision, with the same acceptance bar as every other change in this lineage: no
accuracy regression, verified numerically, not eyeballed — the same fidelity-over-idiom principle
the sibling `repo-adapter` skill names in its own `SKILL.md`.

**The lineage already converged on one answer, independently, more than once**: compile each
top-level submodule of the model in isolation (never the whole model as one graph), measure the
real per-module speedup, and only keep the ones that are an actual, reproducible win.
`fiery-radar`'s commit `0a04263` and `powerbev-radar`'s `75103b9` reached the identical two-module
conclusion (encoder + BEV-projection/decoder-style stage) independently, on different hardware —
`powerbev-radar`'s own `CLAUDE.md` says so explicitly: *"Same conclusion fiery-radar reached
independently."* `tinycar-dev` and `gcarpred-dev` reached the same shape via Hydra instead of yacs.
This skill exists so the *next* repo doesn't have to rediscover it a fifth time.

## What's new here vs. the existing lineage tools

Every reference repo already has (or had) a `tools/benchmark.py` doing per-stage isolated
compile+precision timing — but in all of them, **which modules are bf16-safe was discovered the
hard way**: a real NaN bug in production training (`tinycar-dev`'s TF32/`cdist` bug,
`gaussiancar-dev`'s dead `"bf16"` vs `"bf16-mixed"` string-match bug, `gaussiancar`'s PTv3 backward
pass never supporting mixed precision at all — see `references/case-studies.md`). This skill adds
the missing automated step: **a bf16-compatibility probe that runs before the speed sweep**, so a
module needing an explicit fp32 escape hatch is *found automatically*, not discovered by a silent
NaN three epochs into a real training run.

It also automates model-agnostic **stage discovery** via `model.named_children()` and PyTorch
forward hooks, rather than hand-duplicating the model's `forward()` control flow into a parallel
`forward_with_timing()` (the pattern every existing `tools/benchmark.py` uses, each carrying an
explicit "keep this in sync with `forward()`" comment — a known, accepted maintenance cost this
tool avoids for the parts that can be avoided). See `references/stage-discovery.md` for exactly
what's automatic and what still needs a human/Claude to fill in.

## Principles

- **Isolate, measure, then decide — never compile the whole model as one graph** and never adopt a
  compile/precision default without a number behind it.
- **A crash on a known-hostile stage (vendored CUDA op, sparse-conv backend) is a legitimate,
  informative result, not a tool bug.** Report it and move on; don't treat it as something to fix
  by force.
- **Decide by measured threshold, not vibes** (see `references/decision-thresholds.md`): **>20%**
  faster compiled → enable by default; **10–20%** → ask the user explicitly, don't default either
  way; **<10%, a regression, or a crash** → leave eager.
- **`compile_mode="default"` only, once more than one stage is compiled together.**
  `reduce-overhead`/`max-autotune` are confirmed (independently, in both `tinycar-dev` and
  `gcarpred-dev`) to crash on CUDA-graph aliasing across multiple compiled stages. Don't reach for
  them without re-verifying this from scratch on the target repo.
- **Compile *after* loading checkpoint weights, never before.** `torch.compile` wraps a module in
  `OptimizedModule`, adding an `_orig_mod.` prefix to its state-dict keys — compiling first breaks
  checkpoint key matching. This bit `gcarpred-dev` for real; it's now a non-negotiable here.
  See `references/hydra-wiring.md`.
- **This is fidelity-over-idiom applied to speed work**: a compile/precision change that makes
  training/inference faster but silently drifts accuracy has failed, no matter how good the
  benchmark numbers look. The Verify step against a real baseline is not optional.

## Workflow

1. **Scope & Confirm** (`AskUserQuestion`): target repo; a real (or debug-scale) checkpoint to
   benchmark with real weights rather than random init; does the repo already have a
   `tools/benchmark.py`-style per-stage tool (if so, **extend it** — add the bf16 probe + the
   sweep-driver loop + decision thresholds + HTML upgrade — rather than replacing proven, existing
   tooling); accuracy baseline to verify against at the end; commit at the end? (default: no).

2. **Stage discovery** — enumerate the model's top-level submodules automatically via
   `model.named_children()`. Cross-check against the model's actual `forward()` to catch any
   stage that's a **bound method** rather than an `nn.Module` child (e.g. a depth-prediction or
   Gaussian-evolution helper) — these need to be added by hand to the tool's stage list, since
   they can't be auto-discovered. See `references/stage-discovery.md`. **Check
   `references/use-cases.md` first** — most structural quirks a new repo hits (dict-output stages,
   repeated-invocation stages, config-resolved attribute names, a yacs instead of Hydra config
   system) have already been seen and catalogued; don't rediscover one from scratch.

3. **Copy and adapt** `assets/compile_bf16_sweep.py` into the target repo's **`dev/` folder**
   (deliberately `dev/`, not `tools/`, even though this tool needs the full model stack unlike a
   dependency-light analyzer — it's a one-off research/tuning tool, not a permanent training
   entrypoint; see the note at the top of the template). Wire it to the target's actual Hydra
   model instantiation (`hydra.utils.instantiate(cfg.module.model)`) and a synthetic-batch builder
   shaped like the target's real inputs.

4. **Run the bf16-compatibility probe** — one fp32 forward pass and one bf16-autocast forward pass
   over the same synthetic batch, comparing every stage's hooked output for NaN/Inf or an
   exception. Any stage that fails gets excluded from the bf16 half of the sweep and flagged for a
   fix — read the actual exception/NaN pattern to pick the right one from
   `references/precision-escape-hatches.md` (a whole-submodule `autocast(enabled=False)` wrap for
   "no bf16 kernel at all," vs. narrower explicit `.float()` casts for "this specific op needs
   fp32 regardless of ambient autocast").

5. **Apply the escape hatch(es)** identified in step 4 to the target repo's actual model code, then
   re-run the probe to confirm the fix actually clears it (don't assume — verify).

6. **Run the full sweep**: precision (`fp32` × `bf16-mixed`) × compile (`eager` × `compiled`,
   per-stage in isolation, restoring each stage to eager before moving to the next). A stage crash
   here (see Principles) is caught, reported, and the sweep continues — this must be an unattended,
   complete pass, not one CLI invocation per cell of the grid.

7. **Classify each stage** against `references/decision-thresholds.md`'s bands. For every
   **10–20%** stage, use `AskUserQuestion` — present the actual measured number, don't just say
   "some stages are borderline."

8. **Wire the decision into Hydra config** — the family's comma-separated `compile_stages` string
   convention (`references/hydra-wiring.md`), including the CLI single-quoting caveat and the
   compile-after-checkpoint-load ordering non-negotiable above.

9. **Verify** — a real training/inference smoke run (or a full epoch, matching `repo-adapter`'s
   verification bar) with the new defaults, comparing the resulting accuracy metric against the
   baseline lined up in step 1. No NaN/Inf in logged scalars across the run. This is the actual
   acceptance gate, not the benchmark numbers alone.

10. **Generate the HTML report** (`dev/results.html` or wherever the target's benchmark output
    already lives) — the deliverable: per-stage eager-vs-compiled latency under both precisions,
    memory gains, the bf16-probe result per stage, and the final auto/ask/skip decision table with
    the exact `compile_stages=...` line to paste.

11. **Self-Improve** — two separate updates, not one:
    - Append a dated entry to `references/case-studies.md`: target repo, measured numbers, which
      stages needed an escape hatch and which pattern fixed them, decisions made, anything not
      covered by these reference docs. Same discipline `repo-adapter` uses — don't skip it.
    - **If this invocation hit a structural edge case not already in `references/use-cases.md`**
      (a new kind of stage shape, a new config-system variant, a new class of bf16/compile
      limitation) — add a new catalog entry there, in the indexed format, not just narratively in
      `case-studies.md`. If an edge case *was* already catalogued and this repo just confirms it
      again, add the repo's name to that entry's "Confirmed present in" list instead of duplicating
      the entry. This is what makes step 2's "check `use-cases.md` first" actually pay off over
      time instead of becoming stale.

## Reference index

- `references/stage-discovery.md` — what `named_children()` auto-discovers vs. what needs a
  manual bound-method stage list; cross-repo examples of both kinds.
- `references/decision-thresholds.md` — the >20%/10–20%/<10% bands, worked real numbers from all
  four lineage repos as calibration, and the crash/regression handling.
- `references/precision-escape-hatches.md` — the two escape-hatch patterns (`autocast(enabled=False)`
  vs. explicit `.float()` casts), when to use which, and how to tell from the probe's failure mode.
- `references/hydra-wiring.md` — the comma-separated `compile_stages` convention, the CLI
  single-quoting caveat, and the compile-after-checkpoint-load ordering rule.
- `references/use-cases.md` — the **indexed catalog of structural edge cases** (dict-output stages,
  repeated-invocation stages, config-resolved attribute names, yacs-vs-Hydra, architecturally-blocked
  mixed precision, TF32-not-bf16 bugs) — check this *before* stage discovery on a new repo, and add
  to it whenever a new kind of edge case appears.
- `references/case-studies.md` — the **chronological log** of real invocations: measured numbers,
  decisions made, and the cross-repo gotchas already discovered (TF32/`cdist` NaN, the dead
  `"bf16"`-string-match bug, PTv3's no-mixed-precision-backward limitation, CUDA-graph aliasing on
  multi-stage `reduce-overhead`/`max-autotune`).
- `assets/compile_bf16_sweep.py` — the `dev/` tool template: hook-based stage discovery, bf16
  probe, sweep driver, CSV history, and the HTML report generator.
