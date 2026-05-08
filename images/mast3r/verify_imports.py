"""Verify all critical MASt3R-SLAM imports work.
Catches issues from:
- lietorch CUDA version mismatch (libcudart.so.11.0 not found)
- mast3r_slam_backends C extension not built (no `wheel` package)
- LD_LIBRARY_PATH not set (libc10.so not found)
- Patched dataloader still functional
"""
import torch
print(f'torch {torch.__version__} CUDA={torch.cuda.is_available()}')

import lietorch
print('lietorch OK')

import mast3r_slam_backends
print('mast3r_slam_backends OK')

from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.global_opt import FactorGraph
print('All MASt3R-SLAM imports OK')
