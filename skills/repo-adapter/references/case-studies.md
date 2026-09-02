# Case studies (Self-Improve log)

This is an **append-only** log of nuances, gotchas, and tradeoffs discovered during real invocations of
this skill. It exists so the Systematic/Nuance passes on the *next* adaptation start smarter instead of
rediscovering the same surprise — read it near the start of the Systematic Pass, and always append a new
entry at the end of the Self-Improve workflow step, even for adaptations that went smoothly (a "nothing
unusual, followed the checklist exactly" entry is still useful signal).

## Entry template

```
### YYYY-MM-DD — <target repo>

**What was ported**: one or two sentences — scope of the adaptation (infra only? code restructuring?
config migration? Lightning port?).

**Deviations from the tinycar-dev shape, and why**: what didn't match `references/module-checklist.md`'s
canonical shape, and whether that divergence was justified (real repo-specific need) or should be flagged
as drift to fix later.

**Fidelity-vs-Pythonic tradeoffs made**: any place idiomatic style was deliberately *not* chosen because it
risked changing numeric output — what was kept ugly-but-faithful, and how it was verified to still match.

**New gotchas / nuances discovered**: anything not already covered by the reference docs that future
invocations should know about.

**Follow-ups for next time**: anything left deliberately deferred, or a suspicion worth checking on a
future adaptation of a similar repo.
```

## Entries

### 2026 (undated) — tinycar-dev (retroactive, from its own documented history)

**What was ported**: n/a — this is tinycar-dev itself, not an adaptation of it, but its own `CLAUDE.md`
records a precision regression worth treating as a standing cautionary case study for any future
precision/framework migration performed by this skill.

**Deviations from the tinycar-dev shape, and why**: n/a.

**Fidelity-vs-Pythonic tradeoffs made**: n/a.

**New gotchas / nuances discovered**: switching the radar encoder path and enabling TF32 matmul exposed a
latent bug where `torch.cdist` under TF32 silently produced NaNs for a specific input distribution,
zeroing out the entire radar branch's contribution for a full multi-epoch training run without crashing or
raising — the loss curves still looked plausible. The fix required forcing full-fp32 matmul for the
affected distance computation (a `_full_fp32_matmul()` helper) and OR'ing in an identity mask to guard
against the degenerate case. The training run had already completed and logged metrics before this was
caught — nothing in the loss/metric curves themselves signaled the corruption; it was only caught by
inspecting intermediate tensor statistics directly.

**Follow-ups for next time**: when porting or bumping precision on any repo with a `cdist`-style pairwise
distance computation, don't rely on "loss curve looks reasonable" as a fidelity signal — explicitly check
intermediate tensors for NaN/Inf under the target precision mode before trusting a full training run's
results. This is why `references/verification.md`'s "check log scalars for NaN/Inf explicitly" step exists
and why the fidelity-over-idiom principle in `SKILL.md` treats numeric-parity verification as non-optional,
not a nice-to-have.

### 2026-08-30 — rapidradar-dev (RapidLiDAR, ECCV 2026)

**What was ported**: full adaptation of an externally-authored LiDAR scene-completion repo — uv +
Docker + Slurm infra, package rename, a Hydra migration, a model/module/datamodule split, and two
vendored CUDA extensions. `/fast-torch` ran in the same session on top of it. The repo was already
Lightning-based, so the biggest risk in the workflow (porting a hand-rolled loop) did not apply; what
it lacked was the `model.py`/`module.py` separation, since `RapidLiDAR` was one class holding the
network, the steps, the optimizer and all three dataloaders.

**Deviations from the tinycar-dev shape, and why**:
- `bevpredformer-radar` was used as the reference repo, not tinycar-dev. It is the most recently
  adapted sibling and — decisively — already vendors the *same* `MultiScaleDeformableAttention` CUDA
  op this target needed. Copying tinycar-dev would have meant re-deriving that from scratch.
- `faststart` was deliberately **not** used. Its installed template was verified in sync with its
  repo, but every real repo in the lineage has hand-diverged from it (different dataset env-var
  names, a `jupyter` target none of them kept, different slurm dataset-check logic), so it would have
  produced the generic shape and then needed manual re-customization anyway.
- The Python package was renamed `rapidlidar` → `rapidradar` at the user's request, but **class
  names were kept** (`RapidLiDAR`, `RapidLiDARHubModel`) so the published HF model still loads.

**Fidelity-vs-Pythonic tradeoffs made**:
- `RapidLiDAR.__init__` still accepts `config_path`, `learning_rate` and `batch_size` even though it
  is now a pure `nn.Module` with no use for them. The Hub's `config.json` stores the original
  constructor kwargs and `from_pretrained` replays them, so dropping them breaks the published model
  for everyone. **Check a published `config.json` before trimming any constructor signature.**
- Preserved without "fixing": `set_float32_matmul_precision("medium")` (more aggressive than the
  family's usual `"high"`); the validation dataloader's `shuffle=True`; a second, redundant-looking
  `attn.init_weights()` call that actually consumes RNG and so changes every weight after it;
  `train_refine.py` setting `gradient_clip_algorithm` with no `gradient_clip_val`, which disables
  clipping entirely.
- The ops' *Python frontend* was rewritten for readability at the user's request while the `.cu`/`.cpp`
  sources stayed byte-for-byte upstream — a reasonable split of the "keep vendored code diffable"
  rule, and safe only because numeric equivalence tests existed to back it.

**New gotchas / nuances discovered**:
- **A digest-based forward reference is the single highest-value artifact of a restructure.** Before
  moving anything, a fixed-seed forward pass was captured as sha256 digests of every output
  (`docs/forward_reference.json`, 12 KB — the raw tensors were ~400 MB). It then proved, bit-for-bit,
  that the monolith→split and the bf16 escape hatch changed nothing. It needs no dataset, so it works
  long before the one-epoch gate is reachable. Do this on every restructure.
- **The upstream repo was not installable as written**, in two independent ways: `mmcv==2.2.0` only
  via `mim` (hard-pinning torch 2.4.x), and a `ChamferDistancePytorch` git submodule the README told
  you to init that **never existed** — no `.gitmodules`, no `[submodule]` in `.git/config`. Check
  `git submodule status` early; a README is not evidence.
- **Verify a vendored op three ways**, not one: against the upstream library's own pure-Python
  reference (forward *and* backward), and by asserting the published checkpoint loads with 0
  missing / 0 unexpected keys. The key check is what proves the swap is a genuine drop-in; the
  reference check is what proves the kernel computes the same thing.
- **A blanket `sed` rename needs auditing.** `\brapidlidar\b` correctly skipped
  `rapidlidar_vox_0.3_best.pth` (underscore is a word char) — which was the *right* outcome, since
  that is a published artifact name, not the package. It also silently missed
  `rapidlidar_refine.yaml` for the same reason. Both directions need checking.
- **`outputs/` in `.gitignore` matches at any depth**, so it also swallowed `dev/outputs/`, the
  compile sweep's deliverable. Anchor run-output ignores with a leading slash.
- **A broken host NVIDIA driver does not necessarily block GPU work.** `nvidia-smi` and
  `nvidia-container-cli` both failed all session (kernel module vs userspace version mismatch), so no
  *new* GPU container could start — but containers started *before* the upgrade kept working through
  `docker exec`, holding the matching driver libs. That carried the entire session's GPU verification.
  Flag loudly that stopping such a container is irreversible until the driver is fixed.

**Follow-ups for next time**: the acceptance gate is **not** met — SemanticKITTI is not on the machine,
so Baselines A/B and the one-epoch run are all outstanding and `docs/baseline.md` is a scaffold with
empty rows. Everything below that bar (op equivalence, checkpoint key match, bit-identical
restructure, a real 1-epoch fit + eval on synthetic data) is done. Also unverified: multi-GPU DDP,
and the refinement path end-to-end (only its construction and forward were exercised).
