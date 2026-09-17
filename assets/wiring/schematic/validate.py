#!/usr/bin/env python3

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from catalog import DIAGRAMS

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def require(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing generated output: {path}")


def validate_svg_links(svg_path: Path) -> None:
    root = ET.parse(svg_path).getroot()
    for element in root.iter():
        href = element.get("href") or element.get(XLINK_HREF)
        if not href or href.startswith("#") or "://" in href:
            continue
        target = (svg_path.parent / href).resolve()
        if not target.is_file():
            raise SystemExit(f"dead link in {svg_path}: {href}")


def validate(output_dir: Path) -> None:
    sheets_dir = output_dir / "sheets"
    for diagram in DIAGRAMS:
        for suffix in ("html", "svg", "png", "pdf"):
            require(sheets_dir / f"{diagram.name}.{suffix}")
        validate_svg_links(sheets_dir / f"{diagram.name}.svg")
    print("M16 wiring outputs and links are valid.")


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("generated")
    validate(destination)
