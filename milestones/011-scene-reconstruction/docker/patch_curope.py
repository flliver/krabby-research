"""Patch curope's kernels.cu for PyTorch 2.6+ compatibility.

The deprecated `.type()` API was removed; replace with `.scalar_type()`.
See: https://github.com/huggingface/transformers/issues/35976
"""
import pathlib

p = pathlib.Path('/opt/MASt3R-SLAM/thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu')
if not p.exists():
    raise SystemExit(f'curope kernels.cu not found at {p}')

txt = p.read_text()
before = txt.count('.type()')
txt = txt.replace('.type()', '.scalar_type()')
after_count = txt.count('.scalar_type()')
p.write_text(txt)
print(f'  curope: replaced {before} occurrences of .type() with .scalar_type()')
