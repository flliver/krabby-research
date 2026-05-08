"""Patch MAtCha's curope kernels.cu for PyTorch 2.6+ API.

PyTorch 2.6 removed `tensor.type()`. Replace with `.scalar_type()`.
Same patch as MASt3R-SLAM but at MAtCha's path.

Reference: https://github.com/huggingface/transformers/issues/35976
"""
import pathlib

p = pathlib.Path('/opt/MAtCha/mast3r/dust3r/croco/models/curope/kernels.cu')
if not p.exists():
    raise SystemExit(f'curope kernels.cu not found at {p}')

txt = p.read_text()
before = txt.count('.type()')
txt = txt.replace('.type()', '.scalar_type()')
after = txt.count('.scalar_type()')
p.write_text(txt)
print(f'  curope: replaced {before} occurrences of .type() with .scalar_type() (file now has {after})')
