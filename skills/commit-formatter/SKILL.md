---
name: commit-formatter
description: Formats git commit messages using Gitmoji + Conventional Commits — emoji + type + short imperative description, optionally followed by a small set of labeled self-documentation bullets, no scope, no trailers — following Santi's established convention. Use whenever the user asks to write, suggest, review, tidy up, or format a commit message for any repo, including casual asks like "génerame el commit", "cómo commiteo esto", "dame un mensaje para este cambio", or "how should I commit this" — even if they don't mention gitmoji or conventional commits by name.
---

# Commit Formatter

## Why this exists

A commit log is read far more often than it's written — mostly by future-you
running `git log --oneline`, or an LLM skimming history to understand what
happened before touching related code. The whole point of this format is to
make both of those fast: the emoji signals the *kind* of change at a glance,
the type keyword makes it filterable/greppable, the subject says what
happened, and an optional body captures anything that would otherwise force
someone back into the diff to understand *why*. Every rule below protects one
of those two things — skimmability or self-documentation — not because rules
are inherently good.

## Subject line

```
<emoji> <type>: <Imperative description>
```

- **No scope** (`feat(encoder):`): these are small research repos without
  separate subsystem ownership, so a scope adds ceremony without adding
  information. Worth reconsidering only if a repo grows enough
  maintainers/modules that scoping actually disambiguates something.
- **No trailers, ever** (`Co-Authored-By:`, `Signed-off-by:`, etc.): these
  exist for multi-author collaboration tooling (GitHub's PR co-author
  feature, bots). Drop them if a bot or IDE auto-inserts one — don't
  propagate a trailer just because a template elsewhere had it.
- Imperative, verb-first, capitalized, no trailing period, ideally ≤50 chars.
  Say what changed — the body (if present) is where "why" goes.

## Body (optional self-documentation)

Most commits are fully explained by the subject alone — default to no body.
Add one only when it captures something a reader would otherwise have to
re-derive from the diff or ask you directly.

When you do add a body, use this fixed set of labeled bullets instead of free
prose, so every commit body is scannable the same way:

```
<emoji> <type>: <Imperative description>

- Why: <the problem/context that motivated this, if not obvious from the subject>
- Approach: <the key decision or technique, especially anything non-obvious>
- Result: <measured impact, if this is an experiment/perf/fix — a metric delta, a benchmark number>
- Note: <a caveat, limitation, or explicit follow-up left for later>
```

Include only the bullets that add real information — most bodies need one or
two, not all four. A bullet that just restates the subject in different words
("Why: because it needed fixing") is worse than no body at all; cut it.

Trailers stay forbidden even with a body — self-documentation bullets and
attribution footers are separate concerns, and the "no trailers" rule above
still applies.

## Emoji ↔ type map

Official [Gitmoji](https://gitmoji.dev) entries only, each pinned to one
Conventional Commits type so the pairing is deterministic instead of a matter
of taste each time.

| Emoji | Gitmoji                  | Type       | Use for |
|-------|---------------------------|------------|---------|
| ✨    | `:sparkles:`               | `feat`     | New feature or capability |
| 🐛    | `:bug:`                    | `fix`      | Bug fix |
| 🩹    | `:adhesive_bandage:`       | `fix`      | Minor/non-critical fix |
| ⚡️    | `:zap:`                    | `perf`     | Performance improvement |
| ♻️    | `:recycle:`                | `refactor` | Refactor, no behavior change |
| 🎨    | `:art:`                    | `refactor` | Structure/formatting only |
| 🏗️    | `:building_construction:`  | `refactor` | Architectural changes |
| ✅    | `:white_check_mark:`       | `test`     | Add/update tests |
| 📝    | `:memo:`                   | `docs`     | Documentation |
| 👷    | `:construction_worker:`    | `build`    | CI / build system |
| 🧱    | `:bricks:`                 | `build`    | Infra (Docker, Slurm, deploy, cluster config) |
| 🔧    | `:wrench:`                 | `chore`    | Config file changes |
| ➕    | `:heavy_plus_sign:`        | `chore`    | Add a dependency |
| ➖    | `:heavy_minus_sign:`       | `chore`    | Remove a dependency |
| ⬆️    | `:arrow_up:`               | `chore`    | Upgrade dependencies |
| ⬇️    | `:arrow_down:`             | `chore`    | Downgrade dependencies |
| 🔥    | `:fire:`                   | `chore`    | Remove code or files |
| 🚚    | `:truck:`                  | `chore`    | Move or rename files/paths |
| ⚗️    | `:alembic:`                | `chore`    | Experiments (configs, hyperparams, ablations) |
| 🗃️    | `:card_file_box:`          | `chore`    | Dataset/database-related changes |
| ⏪️    | `:rewind:`                 | `revert`   | Revert a previous commit |
| 🎉    | `:tada:`                   | *(none)*   | A repo's literal first commit — written bare as `🎉 Initial commit`, no type |

If nothing fits well, say so instead of forcing the closest row — a slightly
wrong emoji is worse than asking.

## Decision protocol

1. **Look at the actual diff**, not just the request text.
2. Reverting a prior commit wholesale? → ⏪️ `revert`. Stop.
3. Repo's first-ever commit? → 🎉 `Initial commit` (no type). Stop.
4. New functionality/capability? → ✨ `feat`.
5. Fixes incorrect behavior? → 🐛 `fix` (🩹 if trivial).
6. Restructures/optimizes with no behavior change? → ♻️/🎨/🏗️/⚡️ (most
   specific row).
7. CI, build tooling, deployment, cluster infra? → 👷/🧱 `build`.
8. Docs-only? → 📝 `docs`. Tests-only? → ✅ `test`.
9. Otherwise (deps, config, cleanup, renames, experiments, data) → closest
   `chore` row.
10. Several categories genuinely apply? Prioritize by what a reader would
    search history for later: `feat` > `fix` > `perf` > `refactor` >
    `build` > `test` > `docs` > `chore`. If the change set is truly a
    grab-bag of unrelated things, suggest splitting into separate commits
    instead.
11. **Decide on a body**: does the subject alone tell the whole story? If
    yes, stop there. If not, add only the labeled bullets that carry real
    information (see Body section).

## Worked examples

**No body needed** — the subject says it all:

> Change: added WaffleIron as a radar encoder option.

`✨ feat: Add WaffleIron as radar encoder`

**Body earns its place** — the subject can't carry the motivation or the result:

> Change: swapped the camera backbone from ResNet-50 to EfficientViT-L2 to
> close the gap with camera-only baselines in the scaling comparison.

```
✨ feat: Replace ResNet-50 with EfficientViT-L2 camera encoder

- Why: ResNet-50 was the bottleneck vs. camera-only baselines in the scaling comparison
- Result: Vehicle IoU 54.1 → 57.3, inference 92ms → 75.6ms
```

## Self-improvement

This skill should stay current with how Santi actually corrects or confirms
commit messages. When a session reveals a new edge case (a diff shape the
decision protocol misjudged, an emoji row that was wrong, a correction Santi
gave), append one line under a `## Learned` section below — a rule or
example, not a transcript. Keep each entry to a single line; fold it into
the main tables/protocol above instead if it generalizes past one case.

## Learned

<!-- One line per finding. Prune once folded into the sections above. -->
- The `<emoji> <type>: ` prefix costs ~12 chars, so the ≤50 target leaves ~38 for the description — budget for it rather than trimming after the fact.
- When splitting one large change into per-feature commits, a rename/move commit must stage the *deletions* of the old paths too (`git add -A <old> <new>`), or the intermediate tree carries both copies and the history is unbuildable.
