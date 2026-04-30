"""Patch tetra-triangulation's CMakeLists for non-conda + modern PyTorch builds.

Two fixes:
  1. The original CMakeLists assumes a conda environment that supplies
     CUDA headers via ${CONDA_PREFIX}/include. In a non-conda Docker build,
     cuda_runtime.h is not on the include path for .cpp files (only for .cu
     files via nvcc auto-include). Add an explicit /usr/local/cuda/include.
  2. The FindTorch.cmake forces -D_GLIBCXX_USE_CXX11_ABI=0, but PyTorch 2.7+
     wheels are built with CXX11_ABI=1. Linking against torch with the wrong
     ABI flag causes undefined-symbol errors at import time.
"""
import pathlib

# 1. Add explicit CUDA include before find_package(Torch)
cmakelists = pathlib.Path("/opt/MAtCha/2d-gaussian-splatting/submodules/tetra-triangulation/CMakeLists.txt")
if cmakelists.exists():
    txt = cmakelists.read_text()
    if "Krabby patch: explicit CUDA include" not in txt:
        patch = (
            "# Krabby patch: explicit CUDA include for non-conda builds\n"
            "include_directories(/usr/local/cuda/include)\n\n"
            "find_package(Torch REQUIRED)"
        )
        txt = txt.replace("find_package(Torch REQUIRED)", patch, 1)
        cmakelists.write_text(txt)
        print(f"  patched CMakeLists.txt (added /usr/local/cuda/include)")
    else:
        print(f"  CMakeLists.txt already patched")

# 2. Flip CXX11_ABI 0 -> 1 in FindTorch.cmake
findtorch = pathlib.Path("/opt/MAtCha/2d-gaussian-splatting/submodules/tetra-triangulation/cmake/FindTorch.cmake")
if findtorch.exists():
    txt = findtorch.read_text()
    if "_GLIBCXX_USE_CXX11_ABI=0" in txt:
        txt = txt.replace("_GLIBCXX_USE_CXX11_ABI=0", "_GLIBCXX_USE_CXX11_ABI=1")
        findtorch.write_text(txt)
        print(f"  patched FindTorch.cmake (CXX11_ABI 0 -> 1)")
    else:
        print(f"  FindTorch.cmake CXX11_ABI flag already correct or absent")
