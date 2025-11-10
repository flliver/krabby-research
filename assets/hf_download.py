from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PhysicalAI-SimReady-Warehouse-01",
    repo_type="dataset",
    local_dir="./PhysicalAI-Warehouse01",
    local_dir_use_symlinks=False,           # real files instead of symlinks
    allow_patterns=["*.usd", "*.usda", "*.usdz",
                    "*.png", "*.jpg", "physical_ai_simready_warehouse_01.csv"],
    max_workers=1  # to avoid rate limiting issues,
)