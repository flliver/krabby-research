"""Locate the optional STM32CubeProgrammer CLI for explicit hardware tests."""
import os
from pathlib import Path
import shutil
import re


def programmer_cli():
    configured = os.environ.get('STM32_PROGRAMMER_CLI')
    if configured:
        return configured
    discovered = shutil.which('STM32_Programmer_CLI')
    if discovered:
        return discovered
    bundles = Path.home() / 'AppData/Local/stm32cube/bundles/programmer'
    candidates = sorted(bundles.glob('*/bin/STM32_Programmer_CLI.exe'),
                        key=lambda p: tuple(map(int, re.findall(r'\d+', p.parent.parent.name))))
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError('Set STM32_PROGRAMMER_CLI or add STM32CubeProgrammer bin to PATH')
