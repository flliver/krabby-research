#!/usr/bin/env python3
"""Build script for parkour package wheel from repo root.

This script builds the parkour package as a wheel without modifying
any files in the parkour/ directory. It uses setup_parkour.py at the
repo root to build with the correct package namespace (parkour.scripts.*).
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path

# Build the wheel from repo root using setup_parkour.py
repo_root = Path(__file__).parent
parkour_dir = repo_root / "parkour"
dist_dir = parkour_dir / "dist"

# Create dist directory if it doesn't exist
dist_dir.mkdir(parents=True, exist_ok=True)

# Temporarily rename setup_parkour.py to setup.py so build can find it
setup_parkour = repo_root / "setup_parkour.py"
setup_py = repo_root / "setup.py"
original_setup_exists = setup_py.exists()

try:
    if original_setup_exists:
        # Backup original setup.py
        backup = repo_root / "setup.py.backup"
        shutil.copy2(setup_py, backup)
    
    # Copy setup_parkour.py to setup.py
    shutil.copy2(setup_parkour, setup_py)
    
    # Build wheel
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", f"--outdir={dist_dir}"],
        cwd=repo_root,
        check=True,
    )
finally:
    # Restore original setup.py or remove temporary one
    if original_setup_exists and (repo_root / "setup.py.backup").exists():
        shutil.move(repo_root / "setup.py.backup", setup_py)
    elif setup_py.exists() and not original_setup_exists:
        setup_py.unlink()
    
    # Clean up build directory
    build_dir = repo_root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

print(f"Wheel built successfully in {dist_dir}")

