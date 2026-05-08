"""Patch torch.load calls for PyTorch 2.6+ default weights_only=True.

PyTorch 2.6 changed torch.load() default to weights_only=True for security.
Older checkpoints (including MASt3R's official weights) contain
argparse.Namespace and other non-tensor objects, which now fail with:

  WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace
  was not an allowed global by default.

Fix: add weights_only=False to all torch.load() calls in MASt3R-SLAM and
its thirdparty dependencies.
"""
import pathlib
import re

# Walk the entire MASt3R-SLAM tree and patch every torch.load() call.
# Earlier we tried an explicit file list and missed mast3r/mast3r/model.py
# which is THE crucial loader. Better to patch them all.
roots = [pathlib.Path('/opt/MASt3R-SLAM')]
files = []
for root in roots:
    for p in root.rglob('*.py'):
        if 'torch.load' in p.read_text():
            files.append(str(p))

print(f'Found {len(files)} Python files containing torch.load:')

total = 0
for fp in files:
    p = pathlib.Path(fp)
    if not p.exists():
        continue
    txt = p.read_text()
    # Match torch.load(...) calls that don't already have weights_only=
    # Add weights_only=False before the closing paren on the matched call.
    # Be conservative: only modify when we can identify the exact call.
    new_txt = re.sub(
        r'torch\.load\(([^)]*?)\)',
        lambda m: (
            m.group(0) if 'weights_only' in m.group(1)
            else f'torch.load({m.group(1)}, weights_only=False)'
        ),
        txt
    )
    diff = new_txt.count('weights_only=False') - txt.count('weights_only=False')
    if diff > 0:
        p.write_text(new_txt)
        print(f'  {fp}: added weights_only=False to {diff} torch.load() calls')
        total += diff

print(f'Total: patched {total} torch.load() calls')
