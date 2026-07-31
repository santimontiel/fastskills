#!/usr/bin/env python3
"""Render the token-usage HTML report from a rows JSON file (produced by collect_usage.py).

Usage:
    python3 render_report.py ROWS.json OUT.html [--skipped N]
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows_json")
    ap.add_argument("out_html")
    ap.add_argument("--skipped", type=int, default=0)
    args = ap.parse_args()

    with open(args.rows_json) as f:
        rows = json.load(f)

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")
    with open(template_path) as f:
        html = f.read()

    html = html.replace("__ROWS_JSON__", json.dumps(rows))
    html = html.replace("__SKIPPED_COUNT__", str(args.skipped))

    with open(args.out_html, "w") as f:
        f.write(html)

    print(f"Wrote {args.out_html} ({len(rows)} conversations)")


if __name__ == "__main__":
    main()
