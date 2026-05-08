"""Patch MASt3R-SLAM's dataloader for our use case:
1. Make pyrealsense2 import soft (it's optional, needed only for live RealSense capture)
2. Accept .jpg in addition to .png in the RGBFiles dataset class
"""
import pathlib

p = pathlib.Path('/opt/MASt3R-SLAM/mast3r_slam/dataloader.py')
txt = p.read_text()

# 1. Soft pyrealsense2 import
old_pyrs = 'import pyrealsense2 as rs'
new_pyrs = 'try:\n    import pyrealsense2 as rs\nexcept ImportError:\n    rs = None'
if old_pyrs in txt:
    txt = txt.replace(old_pyrs, new_pyrs)
    print('  Patched pyrealsense2 import → soft')
else:
    print('  WARN: pyrealsense2 import not found (may already be soft)')

# 2. RGBFiles glob: accept .jpg in addition to .png
old_glob = 'natsorted(list((self.dataset_path).glob("*.png")))'
new_glob = 'natsorted(list((self.dataset_path).glob("*.png")) + list((self.dataset_path).glob("*.jpg")))'
if old_glob in txt:
    txt = txt.replace(old_glob, new_glob)
    print('  Patched RGBFiles glob → accept *.jpg')
else:
    print('  WARN: RGBFiles glob pattern not found')

p.write_text(txt)
print('dataloader patched')
