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
