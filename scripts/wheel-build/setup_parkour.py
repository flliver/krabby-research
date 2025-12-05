"""Setup script for building parkour package wheel from repo root.

This setup.py is used only for building the parkour wheel. It does not modify
any files in the parkour/ directory. The wheel is built with the correct
package namespace (parkour.scripts.*) so imports work correctly.
"""
from setuptools import setup

setup(
    name="isaaclab-parkour",
    version="0.1.0",
    description="Custom RSL-RL extensions for parkour training",
    packages=["parkour.scripts", "parkour.scripts.rsl_rl", 
              "parkour.scripts.rsl_rl.modules", 
              "parkour.scripts.rsl_rl.modules.feature_extractors"],
    package_dir={"parkour": "parkour"},
    python_requires=">=3.10",
)

