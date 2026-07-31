#!/usr/bin/env python3
"""Aggregate Claude Code token usage from local session transcripts.

Scans ~/.claude/projects/**/*.jsonl (main session files and their nested
subagents/*.jsonl transcripts), attributes subagent token usage back to the
parent conversation, and emits one row per conversation (session) with
input/output/cache-creation/cache-read token totals.

Usage:
    python3 collect_usage.py [--since YYYY-MM-DD] [--project SUBSTRING]
                              [--json OUT.json] [--csv OUT.csv]

With no output flags, prints a summary to stdout.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_file(fp):
    cwd = None
    first_ts = None
    last_ts = None
    msg_count = 0
    models = set()
    tot = defaultdict(int)
    try:
        with open(fp, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("cwd"):
                    cwd = obj.get("cwd")
                ts = obj.get("timestamp")
                if ts:
                    if first_ts is None or ts < first_ts:
                        first_ts = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts
                msg = obj.get("message")
                if not isinstance(msg, dict):
                    continue
                usage = msg.get("usage")
                if not usage:
                    continue
                model = msg.get("model", "unknown")
                if model and model != "<synthetic>":
                    models.add(model)
                tot["input"] += usage.get("input_tokens", 0) or 0
                tot["output"] += usage.get("output_tokens", 0) or 0
                tot["cache_creation"] += usage.get("cache_creation_input_tokens", 0) or 0
                tot["cache_read"] += usage.get("cache_read_input_tokens", 0) or 0
                msg_count += 1
    except (PermissionError, OSError):
        return None
    return {
        "cwd": cwd, "first_ts": first_ts, "last_ts": last_ts,
        "messages": msg_count, "models": models, "tot": tot,
    }


def collect(base_dir):
    all_files = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True)
    sessions = {}
    skipped = []

    for fp in all_files:
        rel = os.path.relpath(fp, base_dir)
        parts = rel.split(os.sep)
        if len(parts) == 2:
            project_dir, sessionid = parts[0], parts[1][:-6]
            is_main = True
        elif len(parts) >= 3 and parts[2] == "subagents":
            project_dir, sessionid = parts[0], parts[1]
            is_main = False
        else:
            continue

        parsed = parse_file(fp)
        if parsed is None:
            skipped.append(fp)
            continue
        if parsed["messages"] == 0:
            continue

        key = (project_dir, sessionid)
        if key not in sessions:
            sessions[key] = {
                "project": project_dir, "sessionid": sessionid,
                "cwd": None, "first_ts": None, "last_ts": None,
                "messages": 0, "models": set(),
                "tot": defaultdict(int), "has_main": False,
            }
        s = sessions[key]
        if is_main:
            s["has_main"] = True
        if parsed["cwd"]:
            s["cwd"] = parsed["cwd"]
        if parsed["first_ts"] and (s["first_ts"] is None or parsed["first_ts"] < s["first_ts"]):
            s["first_ts"] = parsed["first_ts"]
        if parsed["last_ts"] and (s["last_ts"] is None or parsed["last_ts"] > s["last_ts"]):
            s["last_ts"] = parsed["last_ts"]
        s["messages"] += parsed["messages"]
        s["models"] |= parsed["models"]
        for k, v in parsed["tot"].items():
            s["tot"][k] += v

    home = os.path.expanduser("~")
    rows = []
    for s in sessions.values():
        tot = s["tot"]
        total = tot["input"] + tot["output"] + tot["cache_creation"] + tot["cache_read"]
        billable = tot["input"] + tot["output"] + tot["cache_creation"]
        project = s["project"]
        if s["cwd"]:
            project = s["cwd"].replace(home, "~")
        rows.append({
            "date": (s["first_ts"] or "")[:10],
            "sessionid": s["sessionid"],
            "project": project,
            "models": "|".join(sorted(s["models"])),
            "model_primary": sorted(s["models"])[0] if s["models"] else "unknown",
            "messages": s["messages"],
            "input": tot["input"],
            "output": tot["output"],
            "cache_creation": tot["cache_creation"],
            "cache_read": tot["cache_read"],
            "total": total,
            "billable": billable,
        })
    rows.sort(key=lambda r: r["date"])
    return rows, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default=os.path.expanduser("~/.claude/projects/"))
    ap.add_argument("--since", help="Only include conversations on/after this date (YYYY-MM-DD)")
    ap.add_argument("--project", help="Only include conversations whose project path contains this substring")
    ap.add_argument("--json", help="Write rows as JSON to this path")
    ap.add_argument("--csv", help="Write rows as CSV to this path")
    args = ap.parse_args()

    rows, skipped = collect(args.base_dir)

    if args.since:
        rows = [r for r in rows if r["date"] >= args.since]
    if args.project:
        rows = [r for r in rows if args.project.lower() in r["project"].lower()]

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "sessionid", "project", "models", "messages",
                        "input", "output", "cache_creation", "cache_read", "total", "billable"])
            for r in rows:
                w.writerow([r["date"], r["sessionid"], r["project"], r["models"], r["messages"],
                            r["input"], r["output"], r["cache_creation"], r["cache_read"],
                            r["total"], r["billable"]])

    if not args.json and not args.csv:
        total = sum(r["total"] for r in rows)
        billable = sum(r["billable"] for r in rows)
        output = sum(r["output"] for r in rows)
        print(f"Conversations: {len(rows)}")
        if rows:
            print(f"Date range: {rows[0]['date']} -> {rows[-1]['date']}")
        print(f"Output tokens: {output:,}")
        print(f"Billable-ish total (input+output+cache writes): {billable:,}")
        print(f"All-in total (incl. cache reads): {total:,}")
        if skipped:
            print(f"Skipped (permission denied): {len(skipped)}")

    # Always emit a machine-parseable skipped count, for callers (e.g. the report renderer)
    import sys
    print(f"SKIPPED_COUNT={len(skipped)}", file=sys.stderr)

    if skipped and (args.json or args.csv):
        import sys
        print(f"Note: skipped {len(skipped)} unreadable file(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
