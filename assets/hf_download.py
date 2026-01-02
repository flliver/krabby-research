"""
Incremental full snapshot download for NVIDIA PhysicalAI SimReady Warehouse dataset.

- Downloads ALL files (no filtering) so MDL/materials/textures are included.
- Safe to re-run: only missing or updated files are fetched; existing files are kept.
- Uses real files (no symlinks) under ./PhysicalAI-Warehouse01.

Note:
- Repo id is lowercase to match the dataset slug on Hugging Face.
- If you want faster transfers, set environment variable HF_HUB_ENABLE_HF_TRANSFER=1
  after installing `hf-transfer` (optional).
"""

from pathlib import Path
from huggingface_hub import snapshot_download


def main() -> None:
    target_dir = Path(__file__).parent / "PhysicalAI-Warehouse01"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Perform an incremental snapshot download. With no allow_patterns, we fetch all files.
    # This call is idempotent: re-running will only fetch what's missing or changed.
    local_path = snapshot_download(
        repo_id="nvidia/physicalai-simready-warehouse-01",
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,  # real files instead of symlinks
        max_workers=1,  # parallelism; adjust if you hit rate limits
        # do not set allow_patterns/ignore_patterns to ensure completeness
    )

    # Lightweight summary
    p = Path(local_path)
    total_files = sum(1 for _ in p.rglob("*") if _.is_file())
    # Common extensions present
    exts = {}
    for fp in p.rglob("*"):
        if fp.is_file():
            exts[fp.suffix.lower()] = exts.get(fp.suffix.lower(), 0) + 1

    # Print a compact summary helpful for sanity-checking completeness
    print(f"Downloaded to: {local_path}")
    print(f"Total files: {total_files}")
    interesting = [".usd", ".usda", ".usdz", ".mdl", ".mtlx", ".png", ".jpg", ".exr", ".dds"]
    for ext in interesting:
        if ext in exts:
            print(f"  {ext}: {exts[ext]}")


if __name__ == "__main__":
    main()