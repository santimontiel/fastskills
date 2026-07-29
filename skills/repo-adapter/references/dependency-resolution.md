# Dependency resolution

Bumping an old repo's dependencies is real research, not guessing. Check actual current stable versions
(PyPI, GitHub releases) before picking numbers — don't default to "whatever's newest" without checking
what that actually is, and don't reuse stale version numbers from memory.

## uv's resolver is stricter than pip's

`uv` will refuse to resolve a dependency set that `pip` would have silently installed with a live
conflict. This is a feature, not a nuisance — but it means bumping a dependency in an old repo will
surface real transitive conflicts that were previously invisible.

Concrete worked example: an old data-toolkit dependency (e.g. a specific `nuscenes-devkit` release)
declares `matplotlib<3.6.0`. Pinning `matplotlib==3.11.1` at the top level makes `uv sync` fail outright
with a resolver error, even though the toolkit's actual runtime usage (basic `pyplot` calls) works fine on
the newer version. Fix this with an explicit, explained override — not a downgrade of the thing you
actually wanted to bump:

```toml
[tool.uv]
# <toolkit>==<old version> declares matplotlib<3.6.0, but only uses stable, long-unchanged
# pyplot APIs that work fine on 3.11.x; this bypasses its stale upper bound rather than
# downgrading matplotlib.
override-dependencies = [
    "matplotlib==3.11.1",
]
```

Always write the comment explaining *why* the override is safe — a future reader (or you, in six months)
needs to know this wasn't a blind "make the error go away" edit.

## Python-version wheel availability gaps

An old transitive dependency may lack a prebuilt wheel for the newest Python, forcing `uv` to build it
from source — which can fail outright on removed stdlib APIs (a common one: `pkgutil.ImpImporter`, removed
in Python 3.12, needed by old `setuptools`/`pkg_resources`-based source builds). Don't fight this by
patching the dependency or vendoring a fix. Instead, pin the venv to an older-but-still-current Python
that has a prebuilt wheel available:

```bash
uv python list          # see what's available/downloadable
uv sync --python 3.11    # pin the venv explicitly
```

Check this before assuming a resolver/build failure means the dependency itself is broken — it's often
just a missing wheel for the specific Python version `uv` happened to default to.

## GPU/CUDA index selection

Always verify the actual GPU driver on the deploy target via `nvidia-smi` (max supported CUDA version is
in the output) before picking a `pytorch-cuXXX` uv index — don't default blindly to whatever the reference
repo uses if the target might run on different hardware. Once chosen, **keep the Dockerfile's CUDA base
image tag in lockstep with the chosen index** — e.g. a `cu130` `pyproject.toml` index pairs with an
`nvidia/cuda:13.0.x-devel-*` base image, not a `12.x` one. A mismatch here can silently degrade to CPU
execution or fail at import time inside the container, not at build time — verify by actually running
`torch.cuda.is_available()` inside the built container, not just confirming the image builds.
