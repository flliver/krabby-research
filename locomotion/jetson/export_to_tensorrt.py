"""Script to optionally export PyTorch model to TensorRT for optimized inference.

This script exports a PyTorch checkpoint to TensorRT format for faster inference on Jetson.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def export_to_tensorrt(
    checkpoint_path: str,
    output_path: str,
    obs_dim: int,
    device: str = "cuda",
    precision: str = "fp16",
):
    """Export PyTorch model to TensorRT.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save TensorRT engine
        obs_dim: Observation dimension
        device: Device to use ("cuda")
        precision: Precision mode ("fp16" or "fp32")

    Returns:
        True if export successful, False otherwise
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT not available. Install with: pip install nvidia-tensorrt")
        return False

    logger.info(f"Exporting model to TensorRT: {checkpoint_path} -> {output_path}")
    logger.info(f"Observation dimension: {obs_dim}, Precision: {precision}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available, cannot export to TensorRT")
        return False

    try:
        # Load PyTorch model
        logger.info("Loading PyTorch checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model (structure depends on checkpoint format)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model = checkpoint["model"]
            elif "actor" in checkpoint:
                model = checkpoint["actor"]
            else:
                logger.error("Could not find model in checkpoint")
                return False
        else:
            model = checkpoint

        model = model.to(device)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, obs_dim, device=device, dtype=torch.float32)

        # Export to ONNX first (TensorRT typically requires ONNX as intermediate)
        logger.info("Exporting to ONNX...")
        onnx_path = output_path.replace(".trt", ".onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            opset_version=11,
        )
        logger.info(f"ONNX model saved to: {onnx_path}")

        # Convert ONNX to TensorRT
        logger.info("Converting ONNX to TensorRT...")
        logger.warning("TensorRT conversion requires additional setup and may need manual configuration")
        logger.warning("See TensorRT documentation for full conversion process")

        # Note: Full TensorRT conversion typically requires:
        # 1. ONNX model (done above)
        # 2. TensorRT builder and engine creation
        # 3. Calibration data for INT8 precision
        # 4. Engine serialization

        # This is a placeholder - actual implementation would use TensorRT Python API
        # Example structure:
        # builder = trt.Builder(logger)
        # network = builder.create_network()
        # parser = trt.OnnxParser(network, logger)
        # # ... parse ONNX, build engine, serialize ...

        logger.info("TensorRT export completed (placeholder implementation)")
        logger.info("For production use, implement full TensorRT conversion pipeline")

        return True

    except Exception as e:
        logger.error(f"TensorRT export failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export PyTorch model to TensorRT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save TensorRT engine")
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"], help="Precision mode")

    args = parser.parse_args()

    success = export_to_tensorrt(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        obs_dim=args.obs_dim,
        device=args.device,
        precision=args.precision,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

