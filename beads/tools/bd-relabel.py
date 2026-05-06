#!/usr/bin/env python3
"""Rebuild bd's truncated DOT labels from full titles in the JSONL export.

Usage:
  bd graph <id> --dot | bd-relabel.py <issues.jsonl> [--dark] [--vertical]
                                                     [--with-acceptance]
                                                     [--max-width N]

Writes the transformed DOT to stdout.

The script:
  - Reads the DOT from stdin
  - Looks up each `m11-XXX` (or any prefix-XXX) node by ID in the JSONL
  - Replaces the label with: full title (wrapped) + status badge + priority
    + optional acceptance criteria summary
  - Optionally applies dark-mode palette and rankdir=TB
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

STATUS_ICONS = {
    "open": "○",
    "in_progress": "◐",
    "blocked": "●",
    "closed": "✓",
    "deferred": "❄",
}

# Light → dark palette for fills + edges + text.
DARK_PALETTE = {
    "#e8f4fd": "#1f3a5f",  # open: light blue → deep blue
    "#d4edda": "#1f4d2e",  # closed: light green → deep green
    "#fff3cd": "#5a4216",  # in_progress: light amber → dark amber
    "#f8d7da": "#5a1d2c",  # blocked: light red → dark red
    "#e2e3e5": "#2d2d2d",  # deferred: light grey → dark grey
    "#1a1a1a": "#e6e6e6",  # text near-black → near-white
    "#666666": "#888888",  # edges
}

# Status → fill color (must match bd's mapping for consistency w/ light-mode runs).
STATUS_FILL_LIGHT = {
    "open": "#e8f4fd",
    "in_progress": "#fff3cd",
    "blocked": "#f8d7da",
    "closed": "#d4edda",
    "deferred": "#e2e3e5",
}


def wrap(text: str, width: int) -> str:
    """Wrap text to lines no longer than `width` chars; join with \\n for DOT."""
    lines = textwrap.wrap(text, width=width, break_long_words=False)
    return "\\n".join(lines) if lines else text


def first_sentence(text: str, max_chars: int = 200) -> str:
    """Pull the first sentence (or first paragraph) up to max_chars."""
    if not text:
        return ""
    # Strip markdown emphasis and headers
    cleaned = re.sub(r"[*_`#]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # First sentence: first '.' followed by space/end
    m = re.search(r"^(.*?[.!?])(\s|$)", cleaned)
    out = m.group(1) if m else cleaned
    if len(out) > max_chars:
        out = out[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return out


def build_label(
    issue: dict, width: int, with_acceptance: bool, with_first_line: bool
) -> str:
    icon = STATUS_ICONS.get(issue.get("status", "open"), "○")
    priority = f"P{issue.get('priority', 2)}"
    title = issue.get("title", "")
    parts = [
        f"{icon} {issue['id']}",
        f"{priority} | {wrap(title, width)}",
    ]
    if with_first_line:
        snippet = first_sentence(issue.get("description", ""), max_chars=width * 2)
        if snippet:
            parts.append(wrap(snippet, width))
    if with_acceptance:
        ac = issue.get("acceptance_criteria") or ""
        if ac:
            parts.append(f"AC: {wrap(first_sentence(ac, max_chars=width * 2), width)}")
    return "\\n".join(parts)


def transform(dot_text: str, issues_by_id: dict, *, dark: bool, vertical: bool,
              width: int, with_acceptance: bool, with_first_line: bool) -> str:
    # 1. Substitute each node's label using the issues map.
    #    bd's pattern: "m11-xxx" [label="...", fillcolor="#...", fontcolor="#..."];
    node_re = re.compile(
        r'(\s*"(?P<id>[a-z0-9-]+-[a-z0-9]+)"\s*\[)'
        r'label="[^"]*"(?P<rest>[^]]*)\]'
    )

    def repl(m):
        bid = m.group("id")
        issue = issues_by_id.get(bid)
        if not issue:
            return m.group(0)
        label = build_label(issue, width, with_acceptance, with_first_line)
        # Re-emit with our label and (optionally) recolored fill.
        rest = m.group("rest")
        # Strip existing fill/font to let our palette decide.
        rest = re.sub(r',?\s*fillcolor="[^"]*"', "", rest)
        rest = re.sub(r',?\s*fontcolor="[^"]*"', "", rest)
        fill = STATUS_FILL_LIGHT.get(issue.get("status", "open"), "#e8f4fd")
        if dark:
            fill = DARK_PALETTE.get(fill, fill)
        text_color = "#e6e6e6" if dark else "#1a1a1a"
        return f'{m.group(1)}label="{label}", fillcolor="{fill}", fontcolor="{text_color}"{rest}]'

    out = node_re.sub(repl, dot_text)

    # 2. rankdir override
    if vertical:
        out = re.sub(r"rankdir\s*=\s*\w+\s*;", "rankdir=TB;", out)

    # 3. Dark-mode globals (bg + edge color)
    if dark:
        # Replace global bg and edge colors. Inject if not present.
        if "bgcolor=" not in out:
            out = re.sub(
                r"(digraph\s+\w+\s*\{)",
                r'\1\n  bgcolor="#0d1117";\n  graph [bgcolor="#0d1117"];\n  edge [color="#888888"];',
                out,
                count=1,
            )
        else:
            for light, dark_c in DARK_PALETTE.items():
                out = out.replace(light, dark_c)
        # Re-substitute any remaining edge colors that came through
        out = out.replace('color="#666666"', 'color="#888888"')

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", help="Path to bd export JSONL (e.g. .beads/issues.jsonl)")
    ap.add_argument("--dark", action="store_true", help="Apply dark-mode palette")
    ap.add_argument("--vertical", action="store_true", help="rankdir=TB layout")
    ap.add_argument("--with-acceptance", action="store_true",
                    help="Include first acceptance-criteria line in node label")
    ap.add_argument("--with-description", action="store_true",
                    help="Include first description sentence in node label")
    ap.add_argument("--max-width", type=int, default=44,
                    help="Max characters per wrapped line (default 44)")
    args = ap.parse_args()

    issues = {}
    for line in Path(args.jsonl).read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if d.get("_type") == "issue":
            issues[d["id"]] = d

    dot_in = sys.stdin.read()
    sys.stdout.write(transform(
        dot_in, issues,
        dark=args.dark, vertical=args.vertical,
        width=args.max_width,
        with_acceptance=args.with_acceptance,
        with_first_line=args.with_description,
    ))


if __name__ == "__main__":
    main()
