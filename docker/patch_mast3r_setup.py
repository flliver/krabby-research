"""Patch MASt3R-SLAM's setup.py for Docker build + modern GPU support.

Two problems:

1. setup.py uses `torch.cuda.is_available()` to decide whether to compile the CUDA
   extension. During `docker build`, no GPU is exposed, so this returns False and
   the build fails with "CUDA not found, cannot compile backend!"

2. The hardcoded gencode list only covers sm_60 through sm_86 — missing both
   Ada Lovelace (sm_89, RTX 4080) and Blackwell (sm_120, RTX 5080).

Fixes:
- Replace `torch.cuda.is_available()` with `os.environ.get('CUDA_HOME')` check
  (true at build time when CUDA toolkit is installed, regardless of GPU presence).
- Append sm_89 and sm_120 to the gencode list.
"""
import pathlib

p = pathlib.Path('/opt/MASt3R-SLAM/setup.py')
txt = p.read_text()

# 1. Force has_cuda based on CUDA_HOME being set (build-time check)
old_check = 'has_cuda = torch.cuda.is_available()'
new_check = 'has_cuda = bool(os.environ.get("CUDA_HOME"))  # patched: build-time check'
if old_check in txt:
    txt = txt.replace(old_check, new_check)
    print('  Patched has_cuda check → CUDA_HOME presence (Docker build compatible)')
else:
    raise SystemExit('FAIL: has_cuda line not found')

# 2. Replace gencode list — CUDA 13 dropped sm_60/sm_61/sm_70 support.
#    New list targets only what we actually need + run on:
#    - sm_75 (Turing baseline, broadly compatible)
#    - sm_80 (Ampere, A100)
#    - sm_86 (Ampere, RTX 30xx)
#    - sm_89 (Ada, RTX 40xx) — required for project
#    - sm_120 (Blackwell, RTX 50xx) — required for project
old_block = '''"-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",'''
new_block = '''"-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_120,code=sm_120",'''
if old_block in txt:
    txt = txt.replace(old_block, new_block)
    print('  Patched gencode list → CUDA 13 compatible (Turing+Ampere+Ada+Blackwell)')
else:
    raise SystemExit('FAIL: gencode block not found')

p.write_text(txt)
print('setup.py patched successfully')

# 3. Patch CUDA kernel sources for PyTorch 2.6+ / 2.9 API changes
print()
print('Patching CUDA kernel sources for modern PyTorch API:')

# 3a. matching_kernels.cu — same .type() → .scalar_type() fix as curope
mk = pathlib.Path('/opt/MASt3R-SLAM/mast3r_slam/backend/src/matching_kernels.cu')
if mk.exists():
    mtxt = mk.read_text()
    n = mtxt.count('.type()')
    if n:
        mtxt = mtxt.replace('.type()', '.scalar_type()')
        mk.write_text(mtxt)
        print(f'  matching_kernels.cu: replaced {n} occurrences of .type() with .scalar_type()')

# 3b. gn_kernels.cu — torch::linalg::* → at::linalg_*
#     PyTorch 2.9 removed the torch/linalg.h header entirely. The linalg API lives
#     in the ATen namespace as at::linalg_norm, at::linalg_det, etc.
#     (Earlier rename torch::linalg::linalg_norm → torch::linalg::norm was for 1.9;
#      now we need the ATen path for 2.9+.)
gk = pathlib.Path('/opt/MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu')
if gk.exists():
    gtxt = gk.read_text()
    n_norm = gtxt.count('torch::linalg::linalg_norm')
    if n_norm:
        gtxt = gtxt.replace('torch::linalg::linalg_norm', 'at::linalg_norm')
        print(f'  gn_kernels.cu: replaced {n_norm} occurrences of torch::linalg::linalg_norm → at::linalg_norm')
    n_norm2 = gtxt.count('torch::linalg::norm')
    if n_norm2:
        gtxt = gtxt.replace('torch::linalg::norm', 'at::linalg_norm')
        print(f'  gn_kernels.cu: replaced {n_norm2} occurrences of torch::linalg::norm → at::linalg_norm')
    n_det = gtxt.count('torch::linalg::linalg_det')
    if n_det:
        gtxt = gtxt.replace('torch::linalg::linalg_det', 'at::linalg_det')
        print(f'  gn_kernels.cu: replaced {n_det} occurrences of torch::linalg::linalg_det → at::linalg_det')
    n_det2 = gtxt.count('torch::linalg::det')
    if n_det2:
        gtxt = gtxt.replace('torch::linalg::det', 'at::linalg_det')
        print(f'  gn_kernels.cu: replaced {n_det2} occurrences of torch::linalg::det → at::linalg_det')
    # Also fix .type() → .scalar_type() for consistency
    n_type = gtxt.count('.type()')
    if n_type:
        gtxt = gtxt.replace('.type()', '.scalar_type()')
        print(f'  gn_kernels.cu: replaced {n_type} occurrences of .type() with .scalar_type()')
    if n_norm or n_norm2 or n_det or n_det2 or n_type:
        gk.write_text(gtxt)

print()
print('All patches applied')
