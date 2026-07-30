"""TEMPLATE: automated torch.compile + bf16-mixed precision sweep for a research model.

Lives under the target repo's `dev/` folder by deliberate choice (not `tools/`) -- this is a one-off
tuning tool run when introducing/re-checking compile+precision defaults, not a permanent training
entrypoint, even though (unlike a dependency-light `dev/analyze_run.py`-style tool) it needs the full
model stack to actually run forward passes. See skills/fast-torch/SKILL.md for the workflow this is one
step of, and skills/fast-torch/references/*.md for the reasoning behind every decision this file makes.

NOT a drop-in file. Before using it on a real repo:
  1. Fill in `build_synthetic_batch()` to shape a batch like this repo's actual `model.forward()`
     input (see any existing tools/benchmark.py in the target repo for the exact shapes if one
     exists -- reuse its batch-building logic rather than guessing).
  2. Wire `instantiate_model()` to the target's actual Hydra module config.
  3. Fill in `EXTRA_METHOD_STAGES` with any bound-method stage (not an nn.Module child) this repo's
     forward() calls directly -- named_children() finds everything else automatically. See
     references/stage-discovery.md for real examples (camera_unprojection, gaussian_evolution,
     query_init across the lineage) and why these can't be auto-discovered.
  4. Fill in `KNOWN_COMPILE_HOSTILE_STAGES` if you already know which stage(s) wrap a vendored CUDA
     op / sparse-conv backend with no fake-tensor kernel (optional -- an undeclared crash is still
     caught and reported, this just labels it immediately). See references/stage-discovery.md.
  5. If the target repo already has a tools/benchmark.py-style per-stage tool (tinycar-dev and
     gcarpred-dev both do), prefer EXTENDING it with this file's bf16-probe + sweep-driver + HTML
     upgrade rather than replacing proven, hand-tuned tooling -- see SKILL.md step 1.

Usage (inside the container, matching the lineage's uv-first convention):
    uv run dev/compile_bf16_sweep.py
    uv run dev/compile_bf16_sweep.py checkpoint_path=/workspace/checkpoints/best.ckpt
    uv run dev/compile_bf16_sweep.py num_iters=100 compile_mode=default

Design notes:
  - Stage discovery is automatic for nn.Module children (via model.named_children()) plus whatever
    bound-method stages are added to EXTRA_METHOD_STAGES -- see references/stage-discovery.md for
    exactly what's automatic and what needs a human to fill in.
  - The bf16-compatibility probe and the per-stage latency timing both use forward hooks rather than
    hand-duplicating the model's forward() control flow into a parallel "forward_with_timing()" --
    every existing lineage tools/benchmark.py does that duplication and carries an explicit "keep in
    sync with forward()" comment as an accepted cost. Hooks avoid that cost for anything that's a
    real nn.Module child, at the price of not being able to auto-discover bound-method stages (item 3
    above) or precisely isolate a crash to one submodule inside a single forward() call (see the
    bf16 probe's docstring).
  - Each stage is compiled and timed in isolation, restoring it to eager (in a `finally:`) before
    moving to the next -- matching the safer, catch-and-continue pattern from powerbev-radar's
    tools/benchmark_compile.py, not tinycar-dev/gcarpred-dev's hard-crash-is-fine-for-one-invocation
    convention. This tool is meant to run the *entire* sweep unattended in one invocation, so a crash
    on one stage must not kill the rest of the grid.
"""
import csv
import datetime
import logging
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, Optional, Union

import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

# --- Fill in for the target repo (see module docstring, and references/stage-discovery.md) -------
# name -> attribute name, or a callable(model) -> attribute name for a config-resolved stage (the
# query_init-style case in gcarpred-dev, where the attribute to compile depends on a config value).
EXTRA_METHOD_STAGES: dict[str, Union[str, Callable[[nn.Module], str]]] = {
    # "camera_unprojection": "_pred_depth",
    # "query_init": lambda model: {
    #     "grid_sample": "_grid_sample_query_init", "knn_gaussian": "_knn_gaussian_query_init",
    # }[model.query_init_mode],
}
KNOWN_COMPILE_HOSTILE_STAGES: tuple[str, ...] = ()
# ---------------------------------------------------------------------------------------------

THRESHOLD_DEFAULT_PCT = 20.0  # >this measured speedup -> enable by default
THRESHOLD_ASK_PCT = 10.0  # >this (and <=THRESHOLD_DEFAULT_PCT) -> ask the user; else -> leave eager


def instantiate_model(cfg: DictConfig) -> nn.Module:
    """Placeholder -- wire to the target repo's actual Hydra module config, e.g.:
    return hydra.utils.instantiate(cfg.module.model)
    """
    raise NotImplementedError("Wire this to the target repo's Hydra model instantiation.")


def build_synthetic_batch(cfg: DictConfig, device: str) -> dict:
    """Placeholder -- shape a random batch matching the target model's real forward() input. See
    an existing tools/benchmark.py's build_random_batch()/build_synthetic_batch() in this repo or a
    sibling repo for the exact tensor shapes -- reuse that logic rather than guessing from scratch.
    """
    raise NotImplementedError("Wire this to the target repo's actual input shapes.")


def discover_stages(model: nn.Module) -> dict[str, str]:
    """Auto-discovered nn.Module children, plus the manually-declared bound-method stages."""
    stages = {name: name for name, _ in model.named_children()}
    for name, attr in EXTRA_METHOD_STAGES.items():
        stages[name] = attr(model) if callable(attr) else attr
    return stages


@contextmanager
def compiled_stage(model: nn.Module, attr: str, mode: str):
    """Attribute-swap torch.compile, restoring the original (eager) callable in a `finally:` block
    regardless of success or failure -- so one stage crashing never leaves the model in a partially
    -compiled, inconsistent state for the next stage's measurement.
    """
    compile_kwargs = {} if mode == "default" else {"mode": mode}
    original = getattr(model, attr)
    setattr(model, attr, torch.compile(original, **compile_kwargs))
    try:
        yield
    finally:
        setattr(model, attr, original)
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()


class StageLatencyProbe:
    """Forward hooks on every nn.Module stage, recording per-stage GPU latency across iterations via
    paired torch.cuda.Event, synced individually (so a compiled stage's async kernels are timed
    correctly, and one stage's timing never leaks into another's). Attached once per timed pass and
    covers every discovered nn.Module stage simultaneously, regardless of which single stage is
    currently compiled -- this is what lets one pass produce the full per-stage breakdown, instead of
    needing one script invocation per stage the way every existing lineage tools/benchmark.py does.

    Bound-method stages (EXTRA_METHOD_STAGES) are NOT covered here -- a forward hook needs an
    nn.Module to attach to. Their isolated timing instead falls back to the whole-model wall-clock
    delta between an eager baseline pass and the pass where that stage alone is compiled (see
    run_sweep()) -- an approximation (assumes all else held equal), not a precise isolated number.
    """

    def __init__(self, stage_modules: dict[str, nn.Module]):
        self.samples_ms: dict[str, list[float]] = defaultdict(list)
        self._starts: dict[str, torch.cuda.Event] = {}
        self._handles = []
        for name, module in stage_modules.items():
            self._handles.append(module.register_forward_pre_hook(self._pre(name)))
            self._handles.append(module.register_forward_hook(self._post(name)))

    def _pre(self, name: str):
        def _hook(module, args):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._starts[name] = ev
        return _hook

    def _post(self, name: str):
        def _hook(module, args, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            self.samples_ms[name].append(self._starts[name].elapsed_time(end))
        return _hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for name, samples in self.samples_ms.items():
            t = torch.tensor(samples)
            if t.numel() == 0:
                continue
            out[name] = {
                "mean_ms": t.mean().item(),
                "std_ms": t.std().item() if t.numel() > 1 else 0.0,
            }
        return out


def _iter_tensors(obj):
    """Yield every torch.Tensor leaf inside an arbitrarily nested dict/list/tuple output structure.
    Several real stages (e.g. an image encoder) return a dict of named tensors, not a bare tensor or
    a plain tuple -- a hook that only checks isinstance(output, Tensor) silently observes nothing for
    those and would falsely report full coverage. Walk the whole structure instead of assuming a shape.
    """
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)


def probe_bf16_compatibility(model: nn.Module, stage_modules: dict[str, nn.Module], batch: dict) -> dict[str, str]:
    """One forward pass under bf16 autocast, checking every stage's hooked output(s) for NaN/Inf.

    Returns {stage_name: "ok" | "nan" | "crashed: <message>"}.

    Two things this deliberately does NOT try to paper over:
    - If the whole-model forward call raises, this can report the exception (which usually names the
      failing op/module in its message) but cannot always mechanically isolate WHICH stage caused it,
      since stages aren't independently callable in isolation from inside a single forward(). Read the
      exception message and cross-reference references/precision-escape-hatches.md in that case -- a
      genuine hand-off point to human/Claude judgment, not a gap to force-automate around.
    - This runs on ONE synthetic batch. A clean result is evidence a stage is bf16-safe on the inputs
      actually exercised, not proof for every input distribution -- the real TF32/cdist bug (see
      references/case-studies.md) needed a specific self-distance geometry at a large spatial extent
      that a plain torch.randn batch may never reproduce. Treat "ok" here as "no problem found yet,"
      and still watch for NaN/Inf in real training logs per the sibling repo-adapter skill's
      verification discipline.

    Bound-method stages (EXTRA_METHOD_STAGES) aren't hookable for output capture either -- inspect
    them manually if the whole-model probe raises or shows NaN downstream of one.
    """
    results: dict[str, str] = {}
    # A stage called more than once per forward (e.g. a per-timestep render call invoked once per
    # DT_STEPS-style loop) must have EVERY call checked, not just the last -- an earlier NaN must not
    # be masked by a clean final call. Track "any bad tensor seen" per stage, not "last tensor seen".
    saw_nan: dict[str, bool] = defaultdict(bool)
    saw_any: dict[str, bool] = defaultdict(bool)
    handles = []

    def _capture(name):
        def _hook(module, args, output):
            for t in _iter_tensors(output):
                saw_any[name] = True
                if torch.isnan(t).any() or torch.isinf(t).any():
                    saw_nan[name] = True
        return _hook

    for name, module in stage_modules.items():
        handles.append(module.register_forward_hook(_capture(name)))

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            model(batch)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: report and continue, don't crash the sweep
        for name in stage_modules:
            results[name] = f"crashed: {type(e).__name__}: {e}"
        return results
    finally:
        for h in handles:
            h.remove()

    for name in stage_modules:
        if not saw_any[name]:
            results[name] = "not observed (no tensor output captured)"
        elif saw_nan[name]:
            results[name] = "nan"
        else:
            results[name] = "ok"
    return results


def run_sweep(cfg: DictConfig, model: nn.Module, stage_attrs: dict[str, str], bf16_ok: dict[str, str]) -> list[dict]:
    """The full precision x compile grid: for each precision in (fp32, bf16-mixed) -- skipping bf16
    for any stage the probe flagged as not "ok" -- run one eager baseline pass (all stages, via
    StageLatencyProbe) plus one isolated-compile pass per stage, computing per-stage speedup.

    Returns a list of result rows, one per (stage, precision), ready to append to the CSV history and
    feed into the HTML report.
    """
    stage_modules = {name: getattr(model, attr) for name, attr in stage_attrs.items() if isinstance(getattr(model, attr), nn.Module)}
    rows: list[dict] = []

    # Built ONCE, reused for both precisions -- comparing fp32 vs bf16 on different random batches
    # would add noise the comparison doesn't need; the input doesn't depend on precision.
    batch = build_synthetic_batch(cfg, cfg.device)

    for precision in ("fp32", "bf16"):
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" else nullcontext()
        )

        def _timed_pass(iters: int) -> tuple[dict[str, dict[str, float]], float]:
            # NOTE (unverified): torch.compile applied to model.<attr> wraps that attribute in an
            # OptimizedModule; forward hooks registered on the ORIGINAL module object (captured in
            # stage_modules before compilation) are expected to still fire, since PyTorch's
            # nn.Module.__call__ (where hooks live) sits outside the compiled forward() body Dynamo
            # traces. This is a reasonable expectation, not something actually verified on a GPU in
            # this environment -- if a compiled stage's timing/probe numbers come back suspiciously
            # identical to its eager baseline, hook-skipping under compile is the first thing to
            # check, not a red herring.
            probe = StageLatencyProbe(stage_modules)
            torch.cuda.reset_peak_memory_stats(cfg.device)
            with torch.no_grad(), autocast_ctx:
                for _ in range(cfg.num_warmup_iters):
                    model(batch)
                torch.cuda.synchronize(cfg.device)
                for _ in range(iters):
                    model(batch)
                torch.cuda.synchronize(cfg.device)
            summary = probe.summary()
            probe.remove()
            peak_mb = torch.cuda.max_memory_allocated(cfg.device) / 1024**2
            return summary, peak_mb

        log.info(f"[{precision}] eager baseline pass ({cfg.num_iters} iterations)...")
        eager_summary, eager_peak_mb = _timed_pass(cfg.num_iters)

        for name, attr in stage_attrs.items():
            if precision == "bf16" and bf16_ok.get(name) not in ("ok",):
                rows.append({
                    "stage": name, "precision": precision, "eager_ms": eager_summary.get(name, {}).get("mean_ms"),
                    "compiled_ms": None, "speedup_pct": None,
                    "decision": f"skipped (bf16 probe: {bf16_ok.get(name, 'unknown')})",
                    "peak_mem_eager_mb": round(eager_peak_mb, 1), "peak_mem_compiled_mb": None,
                })
                continue

            if name in KNOWN_COMPILE_HOSTILE_STAGES:
                log.info(f"[{precision}] {name}: known compile-hostile stage, expect a crash or graph break.")

            try:
                with compiled_stage(model, attr, cfg.compile_mode):
                    compiled_summary, compiled_peak_mb = _timed_pass(cfg.num_iters)
                eager_ms = eager_summary.get(name, {}).get("mean_ms")
                compiled_ms = compiled_summary.get(name, {}).get("mean_ms")
                if eager_ms and compiled_ms:
                    speedup_pct = (eager_ms - compiled_ms) / eager_ms * 100
                    decision = classify(speedup_pct)
                else:
                    # bound-method stage: no hook-based per-stage number, fall back to whole-model delta
                    speedup_pct, decision = None, "manual (bound-method stage, see stage-discovery.md)"
                rows.append({
                    "stage": name, "precision": precision, "eager_ms": eager_ms, "compiled_ms": compiled_ms,
                    "speedup_pct": round(speedup_pct, 1) if speedup_pct is not None else None,
                    "decision": decision,
                    "peak_mem_eager_mb": round(eager_peak_mb, 1), "peak_mem_compiled_mb": round(compiled_peak_mb, 1),
                })
            except Exception as e:  # noqa: BLE001 -- a per-stage crash must not kill the rest of the sweep
                log.warning(f"[{precision}] {name}: compile crashed -- {type(e).__name__}: {e}")
                rows.append({
                    "stage": name, "precision": precision, "eager_ms": eager_summary.get(name, {}).get("mean_ms"),
                    "compiled_ms": None, "speedup_pct": None, "decision": f"crashed: {type(e).__name__}: {e}",
                    "peak_mem_eager_mb": round(eager_peak_mb, 1), "peak_mem_compiled_mb": None,
                })
    return rows


def classify(speedup_pct: float) -> str:
    if speedup_pct > THRESHOLD_DEFAULT_PCT:
        return "default"
    if speedup_pct > THRESHOLD_ASK_PCT:
        return "ask"
    if speedup_pct < 0:
        return "regression"
    return "skip"


def append_csv_rows(csv_path: Path, timestamp: str, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp"] + list(rows[0].keys()) if rows else []
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({"timestamp": timestamp, **row})


def read_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_STATUS = {
    "default": ("good", "✅ default"),
    "ask": ("warning", "⚠️ ask"),
    "regression": ("critical", "\U0001f53a regression"),
    "skip": ("muted", "– skip"),
}


def _status_for(row: dict) -> tuple[str, str]:
    decision = row.get("decision") or ""
    if decision.startswith("crashed"):
        return "serious", "\U0001f4a5 crash"
    if decision.startswith("skipped"):
        return "muted", "– bf16 skip"
    return _STATUS.get(decision, ("muted", decision or "?"))


def render_html_report(bf16_probe: dict[str, str], rows: list[dict], compile_stages_line: str) -> str:
    """Self-contained HTML report (no external assets), following the dataviz skill's method: a
    two-shade single-hue bar pair per stage (eager = tint, compiled = the full categorical-slot-1
    blue) for the "before -> after" job, and the fixed 4-role status palette (good/warning/serious/
    critical) -- never reused for anything else -- for the auto/ask/crash/regression decision, always
    paired with an icon + label so the color is never the only signal.
    """
    max_ms = max((r["eager_ms"] for r in rows if r.get("eager_ms")), default=1.0) or 1.0

    def bar_pair(r: dict) -> str:
        eager_w = 100.0 * (r["eager_ms"] or 0) / max_ms
        compiled_w = 100.0 * (r["compiled_ms"] or 0) / max_ms if r.get("compiled_ms") else 0.0
        status, label = _status_for(r)
        speedup = f"{r['speedup_pct']:+.1f}%" if r.get("speedup_pct") is not None else "–"
        return f"""
        <div class="stage-row">
          <div class="stage-name">{_esc(r['stage'])} <span class="badge badge-{status}">{label}</span></div>
          <div class="bar-track"><div class="bar-fill bar-eager" style="width:{eager_w:.1f}%"></div></div>
          <div class="bar-track"><div class="bar-fill bar-compiled" style="width:{compiled_w:.1f}%"></div></div>
          <div class="stage-value">{(r['eager_ms'] or 0):.2f}ms → {(r['compiled_ms'] or 0):.2f}ms <span class="status-{status}">({speedup})</span></div>
        </div>"""

    probe_rows = "\n".join(
        f"""<tr><td>{_esc(name)}</td><td class="status-{'good' if res == 'ok' else ('serious' if res.startswith('crashed') else 'critical')}">
          {'✅ ok' if res == 'ok' else ('\U0001f4a5 ' + _esc(res) if res.startswith('crashed') else '\U0001f53a ' + _esc(res))}
        </td></tr>"""
        for name, res in bf16_probe.items()
    )

    by_precision: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_precision[r["precision"]].append(r)

    sections = "\n".join(
        f"""<h2>{'fp32-true' if prec == 'fp32' else 'bf16-mixed'}</h2>
        <div class="card">{"".join(bar_pair(r) for r in sorted(prs, key=lambda r: -(r.get('eager_ms') or 0)))}</div>"""
        for prec, prs in by_precision.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>torch.compile / bf16-mixed sweep results</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
    --gridline: #e1e0d9; --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
      --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #e66767;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #e66767;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    margin: 0; padding: 2rem 1.5rem 4rem;
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.05rem; margin: 2.5rem 0 0.75rem; }}
  p.sub {{ color: var(--text-secondary); margin-top: 0; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--gridline); }}
  th {{ color: var(--text-muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
  .stage-row {{ display: grid; grid-template-columns: 220px 1fr 1fr 220px; align-items: center; gap: 0.5rem; padding: 0.4rem 0; }}
  .stage-name {{ font-size: 0.82rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar-track {{ background: var(--gridline); border-radius: 4px; height: 10px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .bar-eager {{ background: color-mix(in srgb, var(--series-1) 40%, var(--surface-1)); }}
  .bar-compiled {{ background: var(--series-1); }}
  .stage-value {{ font-size: 0.78rem; text-align: right; font-variant-numeric: tabular-nums; }}
  .badge {{ font-size: 0.65rem; padding: 0.05rem 0.4rem; border-radius: 6px; margin-left: 0.35rem; font-weight: 600; }}
  .badge-good {{ background: color-mix(in srgb, var(--good) 20%, transparent); color: var(--good); }}
  .badge-warning {{ background: color-mix(in srgb, var(--warning) 25%, transparent); color: color-mix(in srgb, var(--warning) 70%, black); }}
  .badge-serious {{ background: color-mix(in srgb, var(--serious) 20%, transparent); color: var(--serious); }}
  .badge-critical {{ background: color-mix(in srgb, var(--critical) 20%, transparent); color: var(--critical); }}
  .badge-muted {{ background: var(--gridline); color: var(--text-muted); }}
  .status-good {{ color: var(--good); }} .status-warning {{ color: color-mix(in srgb, var(--warning) 70%, black); }}
  .status-serious {{ color: var(--serious); }} .status-critical {{ color: var(--critical); }} .status-muted {{ color: var(--text-muted); }}
  code {{ background: var(--gridline); padding: 0.15rem 0.4rem; border-radius: 4px; }}
</style>
</head>
<body data-theme-root>
<div class="wrap">
  <h1>torch.compile / bf16-mixed sweep results</h1>
  <p class="sub">Generated {_esc(datetime.datetime.now().isoformat(timespec='seconds'))}</p>

  <h2>bf16-compatibility probe</h2>
  <div class="card"><table><thead><tr><th>Stage</th><th>Result</th></tr></thead><tbody>{probe_rows}</tbody></table></div>

  {sections}

  <h2>Recommended Hydra config</h2>
  <div class="card"><code>compile_stages={_esc(compile_stages_line)}</code></div>
</div>
</body>
</html>"""


@hydra.main(version_base="1.3", config_path="../configs", config_name="fast_torch.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)  # keep the fp32 and bf16 passes' synthetic batches comparable
    if "cuda" not in cfg.device:
        raise ValueError("This sweep requires a CUDA device.")
    if cfg.compile_mode != "default":
        log.warning(
            "compile_mode != 'default' is confirmed to crash with CUDA-graph aliasing once more than "
            "one stage is compiled together (see references/decision-thresholds.md) -- proceeding "
            "anyway since this sweep compiles one stage at a time, but do not adopt this mode for a "
            "multi-stage production config without re-verifying that combination separately."
        )

    log.info("Instantiating model...")
    model = instantiate_model(cfg).to(cfg.device).eval()
    stage_attrs = discover_stages(model)
    log.info(f"Discovered stages: {sorted(stage_attrs)}")

    batch = build_synthetic_batch(cfg, cfg.device)
    stage_modules = {name: getattr(model, attr) for name, attr in stage_attrs.items() if isinstance(getattr(model, attr), nn.Module)}

    log.info("Running bf16-compatibility probe...")
    bf16_probe = probe_bf16_compatibility(model, stage_modules, batch)
    for name, result in bf16_probe.items():
        if result != "ok":
            log.warning(f"bf16 probe: {name} -> {result} (see references/precision-escape-hatches.md)")

    log.info("Running the full precision x compile sweep...")
    rows = run_sweep(cfg, model, stage_attrs, bf16_probe)

    default_stages = sorted(r["stage"] for r in rows if r["decision"] == "default" and r["precision"] == "fp32")
    ask_stages = sorted(r["stage"] for r in rows if r["decision"] == "ask" and r["precision"] == "fp32")
    compile_stages_line = ",".join(default_stages) if default_stages else "none"

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.rule("[bold red]fast-torch sweep results")
    table = Table(title="Per-stage decision (fp32)")
    table.add_column("Stage"); table.add_column("Speedup"); table.add_column("Decision")
    for r in rows:
        if r["precision"] != "fp32":
            continue
        speedup = f"{r['speedup_pct']:+.1f}%" if r.get("speedup_pct") is not None else "-"
        table.add_row(r["stage"], speedup, r["decision"])
    console.print(table)
    if ask_stages:
        console.print(f"[yellow]Ask the user about: {ask_stages}[/yellow]")
    console.print(f"Recommended: compile_stages={compile_stages_line!r}")

    output_dir = Path(cfg.output_dir)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    if rows:
        append_csv_rows(output_dir / "sweep_results.csv", timestamp, rows)
    (output_dir / "results.html").write_text(render_html_report(bf16_probe, rows, compile_stages_line))
    log.info(f"Results appended to <{output_dir / 'sweep_results.csv'}>.")
    log.info(f"HTML report written to <{output_dir / 'results.html'}>.")
    log.info("Sweep completed. If any stage is in the 10-20% band, use AskUserQuestion before wiring "
              "the config -- see SKILL.md workflow step 7.")


if __name__ == "__main__":
    main()
