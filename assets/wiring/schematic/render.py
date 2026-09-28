#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

from catalog import DIAGRAMS
from theme import configure, write_inline_html


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def convert(svg: Path, png: Path, pdf: Path) -> None:
    run(
        "rsvg-convert",
        "--background-color",
        "white",
        str(svg),
        "-o",
        str(png),
    )
    run(
        "rsvg-convert",
        "--background-color",
        "white",
        "-f",
        "pdf",
        str(svg),
        "-o",
        str(pdf),
    )


def build(output_dir: Path) -> None:
    sheets_dir = output_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    configure()

    for diagram in DIAGRAMS:
        svg = sheets_dir / f"{diagram.name}.svg"
        png = sheets_dir / f"{diagram.name}.png"
        pdf = sheets_dir / f"{diagram.name}.pdf"
        html = sheets_dir / f"{diagram.name}.html"

        diagram.build(svg)
        write_inline_html(svg, html, diagram.title, diagram.hint)
        convert(svg, png, pdf)


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("generated")
    build(destination)
