from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Diagram:
    name: str
    title: str
    hint: str
    build: Callable[[Path], None]
