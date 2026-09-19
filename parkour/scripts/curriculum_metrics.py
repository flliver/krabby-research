#!/usr/bin/env python3
"""Metrics extraction/history for the crab-hex training curriculum (see
``.claude/skills/crab-hex-train/SKILL.md``). Pure stdlib -- does not need the
isaac_venv, works with any python3.

Parses RSL-RL's per-iteration stdout blocks (``Learning iteration N/M`` followed
by ``Label: value`` lines) out of a training log file, and provides:

  extract  -- windowed-average metrics from the end of one log file (one batch)
  trend    -- metrics sampled every --step iterations across one log file
  save     -- append a batch's final-window metrics to that stage's history
  compare  -- print a stage's history (most recent vs. previous vs. best)
  list     -- list all stages with saved history

History is stored at ``<repo>/parkour/logs/rsl_rl/metrics_history/<stage>.jsonl``
(one JSON object per line, oldest first) -- independent of any specific training
run's own log directory, so it survives across separate invocations of the skill.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
# "<label>: <number>" with nothing trailing (excludes "Computation: 1518 steps/s (...)",
# "Iteration time: 4.05s", "ETA: 00:33:43", etc.)
METRIC_RE = re.compile(r"^(.+?):\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*$")

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = REPO_ROOT / "parkour" / "logs" / "rsl_rl" / "metrics_history"


def parse_blocks(log_path: Path) -> list[dict]:
    """Return one dict per ``Learning iteration`` block: {"iteration": int, <label>: float, ...}."""
    text = log_path.read_text(errors="replace")
    chunks = re.split(r"(?=Learning iteration \d+/\d+)", text)
    blocks = []
    for chunk in chunks:
        m = ITER_RE.search(chunk)
        if not m:
            continue
        block = {"iteration": int(m.group(1)), "target_iteration": int(m.group(2))}
        for line in chunk.splitlines():
            stripped = line.strip()
            # Skip "ETA: 00:00:06" / "Time elapsed: 00:10:40" / "Iteration time: 4.05s" style
            # lines -- MM:SS timestamps have >1 colon and confuse the label:value split via
            # regex backtracking; plain "s"-suffixed durations aren't bare numbers either but
            # are cheaper to just exclude by colon count up front.
            if stripped.count(":") != 1:
                continue
            mm = METRIC_RE.match(stripped)
            if mm:
                label, value = mm.group(1).strip(), mm.group(2)
                try:
                    block[label] = float(value)
                except ValueError:
                    continue
        blocks.append(block)
    return blocks


def windowed_average(blocks: list[dict], window: int) -> dict:
    """Average every numeric key over the last ``window`` blocks (skips iteration/target_iteration)."""
    if not blocks:
        return {}
    tail = blocks[-window:]
    keys = set()
    for b in tail:
        keys.update(b.keys())
    keys -= {"iteration", "target_iteration"}
    out = {}
    for k in sorted(keys):
        vals = [b[k] for b in tail if k in b]
        if vals:
            out[k] = round(statistics.fmean(vals), 6)
    out["_window_blocks"] = len(tail)
    out["_iteration_start"] = tail[0]["iteration"]
    out["_iteration_end"] = tail[-1]["iteration"]
    return out


def cmd_extract(args) -> None:
    blocks = parse_blocks(Path(args.log))
    if not blocks:
        print(json.dumps({"error": "no iteration blocks found in log"}))
        sys.exit(1)
    print(json.dumps(windowed_average(blocks, args.window), indent=2, sort_keys=True))


def cmd_trend(args) -> None:
    blocks = parse_blocks(Path(args.log))
    if not blocks:
        print("no iteration blocks found in log", file=sys.stderr)
        sys.exit(1)
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    first_it, last_it = blocks[0]["iteration"], blocks[-1]["iteration"]
    sample_at = list(range(first_it, last_it + 1, args.step)) + [last_it]
    seen = set()
    for target in sample_at:
        if target in seen:
            continue
        seen.add(target)
        row = min(blocks, key=lambda b: abs(b["iteration"] - target))
        if keys:
            filtered = {"iteration": row["iteration"]}
            for k in keys:
                if k in row:
                    filtered[k] = row[k]
            print(json.dumps(filtered))
        else:
            print(json.dumps(row, sort_keys=True))


def cmd_save(args) -> None:
    blocks = parse_blocks(Path(args.log))
    if not blocks:
        print("no iteration blocks found in log -- nothing saved", file=sys.stderr)
        sys.exit(1)
    metrics = windowed_average(blocks, args.window)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "checkpoint": args.checkpoint,
        "log": str(Path(args.log).resolve()),
        "note": args.note or "",
        "metrics": metrics,
    }
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = HISTORY_DIR / f"{args.stage}.jsonl"
    with hist_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"saved to {hist_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _load_history(stage: str) -> list[dict]:
    hist_path = HISTORY_DIR / f"{stage}.jsonl"
    if not hist_path.exists():
        return []
    records = []
    with hist_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cmd_compare(args) -> None:
    records = _load_history(args.stage)
    if not records:
        print(f"no history for stage '{args.stage}' yet")
        return
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    print(f"=== {args.stage}: {len(records)} saved run(s) ===")
    for i, rec in enumerate(records):
        label = "latest" if i == len(records) - 1 else f"run {i + 1}"
        m = rec["metrics"]
        print(f"\n[{label}] {rec['timestamp']}  checkpoint={rec['checkpoint']}")
        if rec.get("note"):
            print(f"  note: {rec['note']}")
        show_keys = keys or sorted(k for k in m if not k.startswith("_"))
        for k in show_keys:
            if k in m:
                prev = records[i - 1]["metrics"].get(k) if i > 0 else None
                delta = f"  (Δ {m[k] - prev:+.4g} vs previous run)" if prev is not None else ""
                print(f"  {k}: {m[k]:.4g}{delta}")


def cmd_list(args) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stages = sorted(p.stem for p in HISTORY_DIR.glob("*.jsonl"))
    if not stages:
        print("no stage history saved yet")
        return
    for stage in stages:
        n = len(_load_history(stage))
        print(f"{stage}: {n} saved run(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="windowed-average metrics from the end of a log file")
    p.add_argument("--log", required=True)
    p.add_argument("--window", type=int, default=10, help="number of trailing iteration-blocks to average")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("trend", help="sample metrics every --step iterations across a log file")
    p.add_argument("--log", required=True)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--keys", help="comma-separated metric labels to show (default: all)")
    p.set_defaults(func=cmd_trend)

    p = sub.add_parser("save", help="append a batch's final-window metrics to stage history")
    p.add_argument("--stage", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_save)

    p = sub.add_parser("compare", help="print a stage's saved history")
    p.add_argument("--stage", required=True)
    p.add_argument("--keys", help="comma-separated metric labels to show (default: all)")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("list", help="list stages with saved history")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
