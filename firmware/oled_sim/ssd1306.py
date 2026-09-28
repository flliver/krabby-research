"""Pixel-accurate simulator for the 128x64 SparkFun Qwiic OLED."""
from __future__ import annotations

import os
import re
from pathlib import Path

WIDTH = 128
HEIGHT = 64
COLOR_BLACK = 0
COLOR_WHITE = 1

# Font lookup order: environment override, repository, then Arduino sketchbook.
_LIB_ENV = os.environ.get("QWIIC_OLED_LIB_DIR")
_REPO_RES = (
    Path(__file__).resolve().parents[1]
    / "arduino/libraries/SparkFun_Qwiic_OLED_Arduino_Library/src/res"
)
_HOME_RES = Path.home() / (
    "Documents/Arduino/libraries/SparkFun_Qwiic_OLED_Arduino_Library/src/res"
)


def _resolve_res_dir() -> "Path | None":
    candidates = []
    if _LIB_ENV:
        candidates.append(Path(_LIB_ENV) / "res")
    candidates.append(_REPO_RES)
    candidates.append(_HOME_RES)
    for d in candidates:
        if (d / "_fnt_5x7.h").exists():
            return d
    return None


_RES_DIR = _resolve_res_dir()


class Font:
    """SparkFun bitmap font parsed from its C header."""

    def __init__(self, name: str, width: int, height: int, start: int,
                 n_chars: int, map_width: int, data: list[int]):
        self.name = name
        self.width = width
        self.height = height
        self.start = start
        self.n_chars = n_chars
        self.map_width = map_width
        self.data = data
        self.margin = 1 if (height // 8 or 1) == 1 else 0  # 5x7 gets a 1px gap

    @property
    def advance(self) -> int:
        return self.width + self.margin

    @classmethod
    def from_header(cls, prefix: str, header: Path) -> "Font":
        txt = header.read_text()

        def _def(field: str) -> int:
            m = re.search(rf"#define\s+{prefix}_{field}\s+(\d+)", txt)
            if not m:
                raise ValueError(f"{prefix}_{field} not found in {header}")
            return int(m.group(1))

        body = txt.split("_data[]", 1)[1]
        body = body.split("{", 1)[1]
        data = [int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]{2}", body)]
        return cls(prefix, _def("WIDTH"), _def("HEIGHT"), _def("START"),
                   _def("NCHAR"), _def("MAP_WIDTH"), data)


def _load_fonts() -> dict[str, Font]:
    fonts = {}
    if _RES_DIR is None:
        return fonts
    for key, prefix, fname in (
        ("5x7", "FONT_5X7", "_fnt_5x7.h"),
        ("8x16", "FONT_8X16", "_fnt_8x16.h"),
    ):
        path = _RES_DIR / fname
        if path.exists():
            fonts[key] = Font.from_header(prefix, path)
    return fonts


_FONTS = _load_fonts()


class OLED:
    """128x64 1-bit framebuffer with the SparkFun primitive API."""

    def __init__(self):
        self.buf = bytearray(WIDTH * HEIGHT)  # one byte/pixel, 0/1
        self.font = _FONTS.get("5x7")

    # --- framebuffer ---
    def erase(self):
        for i in range(len(self.buf)):
            self.buf[i] = 0

    def display(self):
        # The simulated buffer needs no hardware flush.
        return

    def pixel(self, x: int, y: int, clr: int = COLOR_WHITE):
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            self.buf[y * WIDTH + x] = 1 if clr else 0

    def get(self, x: int, y: int) -> int:
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            return self.buf[y * WIDTH + x]
        return 0

    def line(self, x0, y0, x1, y1, clr: int = COLOR_WHITE):
        # Port of SparkFun's steep-swap Bresenham implementation.
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx == 0 and dy == 0:
            self.pixel(x0, y0, clr)
            return
        steep = dy > dx
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            dx, dy = dy, dx
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        err = dx // 2
        ystep = 1 if y0 < y1 else -1
        y = y0
        for x in range(x0, x1 + 1):
            self.pixel(y, x, clr) if steep else self.pixel(x, y, clr)
            err -= dy
            if err < 0:
                y += ystep
                err += dx

    def rectangle(self, x, y, w, h, clr: int = COLOR_WHITE):
        # SparkFun omits vertical sides when height is below four pixels.
        if w <= 1 or h <= 1:
            self.line(x, y, x + w - 1, y + h - 1, clr)
            return
        x1, y1 = x + w - 1, y + h - 1
        self.line(x, y, x1, y, clr)
        self.line(x, y1, x1, y1, clr)
        if y1 - y < 3:
            return
        self.line(x, y + 1, x, y1, clr)
        self.line(x1, y + 1, x1, y1, clr)

    def rectangleFill(self, x, y, w, h, clr: int = COLOR_WHITE):
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.pixel(xx, yy, clr)

    def setFont(self, key: str):
        if key not in _FONTS:
            raise ValueError(f"font {key!r} not loaded (have {list(_FONTS)})")
        self.font = _FONTS[key]

    def text(self, x0: int, y0: int, s: str, clr: int = COLOR_WHITE):
        """Faithful port of QwGrBufferDevice::drawText/drawCharacter."""
        f = self.font
        n_rows = (f.height // 8) or 1
        margin = 1 if n_rows == 1 else 0
        n_row_len = f.map_width // f.width
        row_bytes = f.map_width * n_rows
        x = x0
        for ch in s:
            off = ord(ch) - f.start
            if 0 <= off < f.n_chars:
                font_index = (off // n_row_len) * row_bytes + (off % n_row_len) * f.width
                for row in range(n_rows):
                    row_offset = row * 8
                    for i in range(f.width + margin):
                        if margin and i == f.width:
                            continue
                        idx = font_index + i + row * f.map_width
                        col = f.data[idx] if idx < len(f.data) else 0
                        for j in range(8):
                            if col & (1 << j):
                                self.pixel(x + i, y0 + j + row_offset, clr)
            x += f.width + margin

    def text_width(self, s: str) -> int:
        return len(s) * (self.font.width + self.font.margin)

    def to_ascii(self, on="#", off=".") -> str:
        rows = []
        for y in range(HEIGHT):
            rows.append("".join(on if self.buf[y * WIDTH + x] else off
                                for x in range(WIDTH)))
        return "\n".join(rows)

    def to_rows(self) -> list[str]:
        return self.to_ascii().split("\n")


def fonts_available() -> list[str]:
    return list(_FONTS)
