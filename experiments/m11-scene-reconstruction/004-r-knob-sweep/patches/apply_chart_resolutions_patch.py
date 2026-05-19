"""Idempotent patcher: add chart_resolutions kwarg to align_charts_in_parallel().

Two-line change to MAtCha's matcha/dm_trainers/charts_alignment.py so that
the alignment YAML's `chart_resolutions:` field flows through to the
MultiResChartsEncodingParams constructor.

This is safe to run multiple times — it detects the marker comments it
inserts and refuses to re-patch.

Usage:
    python3 apply_chart_resolutions_patch.py <path-to-charts_alignment.py>

Default path inside the matcha-build container:
    /opt/MAtCha/matcha/dm_trainers/charts_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_TARGET = Path("/opt/MAtCha/matcha/dm_trainers/charts_alignment.py")
MARKER = "# r-knob-sweep patch"


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    if not target.is_file():
        sys.exit(f"ERROR: {target} not found")

    text = target.read_text()
    if MARKER in text:
        print(f"  already patched ({target}); skipping")
        return

    # --- Patch 1: add chart_resolutions kwarg to function signature -------
    sig_anchor = "    use_multi_res_charts_encoding=True,\n"
    sig_replacement = (
        "    use_multi_res_charts_encoding=True,\n"
        f"    chart_resolutions=None,  {MARKER}\n"
    )
    if sig_anchor not in text:
        sys.exit(
            "ERROR: signature anchor not found "
            "('    use_multi_res_charts_encoding=True,'). "
            "Has the file been modified upstream?"
        )
    text = text.replace(sig_anchor, sig_replacement, 1)

    # --- Patch 2: thread chart_resolutions into MultiResChartsEncodingParams ---
    # AND co-vary ChartsEncodingParams + DepthEncodingParams so their
    # `encoding_dim` matches the multi-res total. Two upstream sites
    # (parallel_aligner_with_cameras.py:351 and the depth-encoding adder)
    # use these dims and break if they diverge.
    cons_anchor = (
        "    pa = ParallelAligner(\n"
        "        depths=initial_depths,\n"
        "        cameras=lowres_cameras,\n"
        "        charts_encoding_params=ChartsEncodingParams(),\n"
        "        depth_encoding_params=DepthEncodingParams(),\n"
    )
    cons_replacement = (
        "    pa = ParallelAligner(\n"
        "        depths=initial_depths,\n"
        "        cameras=lowres_cameras,\n"
        f"        # {MARKER}: when chart_resolutions is set, both encoding\n"
        "        # dims must equal 8 * len(resolutions) — see\n"
        "        # parallel_aligner_with_cameras.py:351 (uses ChartsEncodingParams)\n"
        "        # and the depth_encoding+chart_encoding sum at :367.\n"
        "        charts_encoding_params=ChartsEncodingParams(\n"
        "            encoding_dim=(8 * len(chart_resolutions)) if chart_resolutions is not None else 32\n"
        "        ),\n"
        "        depth_encoding_params=DepthEncodingParams(\n"
        "            encoding_dim=(8 * len(chart_resolutions)) if chart_resolutions is not None else 32\n"
        "        ),\n"
    )
    if cons_anchor not in text:
        sys.exit(
            "ERROR: ParallelAligner-construction anchor not found. "
            "Has the file been modified upstream?"
        )
    text = text.replace(cons_anchor, cons_replacement, 1)

    # --- Patch 3: thread chart_resolutions into MultiResChartsEncodingParams ---
    mr_anchor = "        multi_res_charts_encoding_params=MultiResChartsEncodingParams(),\n"
    mr_replacement = (
        f"        multi_res_charts_encoding_params=MultiResChartsEncodingParams(  {MARKER}\n"
        "            resolutions=chart_resolutions if chart_resolutions is not None\n"
        "                        else [0.05, 0.1, 0.2, 0.4]\n"
        "        ),\n"
    )
    if mr_anchor not in text:
        sys.exit(
            "ERROR: MultiResChartsEncodingParams construction anchor not found."
        )
    text = text.replace(mr_anchor, mr_replacement, 1)

    # Save backup once, then write
    backup = target.with_suffix(target.suffix + ".pre-r-knob-sweep")
    if not backup.exists():
        backup.write_text(target.read_text())
        print(f"  backed up: {backup}")
    target.write_text(text)
    print(f"  patched: {target}")
    print(f"  marker:  '{MARKER}'")


if __name__ == "__main__":
    main()
