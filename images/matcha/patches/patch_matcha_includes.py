"""Patch missing C++ standard headers in MAtCha's CUDA extensions.

Ubuntu 24.04 ships gcc 13, which stopped transitively including <cstdint> and
<cfloat> through other headers. The Inria 2D Gaussian Splatting code (~2023)
relied on those transitive includes, so we have to add them explicitly.

Affected files:
  - diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h
  - diff-surfel-rasterization/cuda_rasterizer/auxiliary.h
  - simple-knn/simple_knn.h
  - simple-knn/spatial.h
    (above 4 use uintptr_t / uint32_t / uint64_t — need <cstdint>)

  - simple-knn/simple_knn.cu
    (uses FLT_MAX — needs <cfloat>; must be prepended at line 1, NOT
    inserted after #pragma once because the file's pragma sits inside
    a leading multi-line comment)
"""
import pathlib

CSTDINT_HEADERS = [
    "/opt/MAtCha/2d-gaussian-splatting/submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h",
    "/opt/MAtCha/2d-gaussian-splatting/submodules/diff-surfel-rasterization/cuda_rasterizer/auxiliary.h",
    "/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn/simple_knn.h",
    "/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn/spatial.h",
]

CFLOAT_FILE = "/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn/simple_knn.cu"


def add_cstdint(path: str) -> None:
    p = pathlib.Path(path)
    if not p.exists():
        print(f'  SKIP (missing): {path}')
        return
    txt = p.read_text()
    if '#include <cstdint>' in txt:
        print(f'  already has <cstdint>: {path}')
        return
    # Insert after the first #pragma once
    new_txt = txt.replace('#pragma once', '#pragma once\n#include <cstdint>', 1)
    p.write_text(new_txt)
    print(f'  added <cstdint>: {path}')


def add_cfloat(path: str) -> None:
    p = pathlib.Path(path)
    if not p.exists():
        print(f'  SKIP (missing): {path}')
        return
    txt = p.read_text()
    if '#include <cfloat>' in txt:
        print(f'  already has <cfloat>: {path}')
        return
    # Prepend at line 1 (the file starts with a multi-line comment;
    # putting it inside the comment block does not work).
    p.write_text('#include <cfloat>\n' + txt)
    print(f'  prepended <cfloat>: {path}')


for h in CSTDINT_HEADERS:
    add_cstdint(h)

add_cfloat(CFLOAT_FILE)
