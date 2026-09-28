from pathlib import Path

import schemdraw
import schemdraw.elements as elm


INK = "#172033"
MUTED = "#526178"


def configure() -> None:
    schemdraw.use("svg")
    schemdraw.svgconfig.text = "text"
    elm.style(elm.STYLE_IEEE)


def drawing(path: Path) -> schemdraw.Drawing:
    result = schemdraw.Drawing(
        file=str(path),
        show=False,
        canvas="svg",
        transparent=False,
        color=INK,
    )
    result.config(
        unit=1.0,
        lw=1.8,
        fontsize=11,
        bgcolor="white",
        margin=0.35,
    )
    return result


def add_title(
    diagram: schemdraw.Drawing,
    title: str,
    y: float = 7.2,
) -> None:
    diagram.add(
        elm.Label()
        .at((0, y))
        .label(title, fontsize=18, halign="left")
    )


def write_inline_html(
    svg_path: Path,
    html_path: Path,
    title: str,
    hint: str,
) -> None:
    svg = svg_path.read_text(encoding="utf-8")
    html_path.write_text(
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>"
        "body{margin:0;padding:24px;background:#f8fafc;font-family:sans-serif}"
        ".hint{color:#526178;margin:0 0 16px}"
        "svg{display:block;width:100%;height:auto;background:white}"
        "</style></head><body>"
        f"<p class='hint'>{hint}</p>"
        f"{svg}"
        "</body></html>\n",
        encoding="utf-8",
    )
