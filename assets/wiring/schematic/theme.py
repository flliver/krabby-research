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
