"""Script to test loading model checkpoint on Jetson.

This script verifies that checkpoints can be loaded successfully on Jetson hardware.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_checkpoint_loading(checkpoint_path: str, action_dim: int, obs_dim: int, device: str = "cuda"):
    """Test loading a checkpoint on Jetson.

    Args:
        checkpoint_path: Path to checkpoint file
        action_dim: Expected action dimension
        obs_dim: Expected observation dimension
        device: Device to load on ("cuda" or "cpu")

    Returns:
        True if checkpoint loaded successfully, False otherwise
    """
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False

    logger.info(f"Testing checkpoint loading: {checkpoint_path}")
    logger.info(f"Device: {device}, Action dim: {action_dim}, Obs dim: {obs_dim}")

    try:
        # Check device availability
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            else:
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")

        # Create model weights
        weights = ModelWeights(
            checkpoint_path=str(checkpoint_path),
            action_dim=action_dim,
            obs_dim=obs_dim,
        )

        # Load model
        logger.info("Loading model...")
        model = ParkourPolicyModel(weights, device=device)
        logger.info("Model loaded successfully")

        # Verify model is on correct device
        if device == "cuda":
            # Check if model parameters are on CUDA
            for name, param in model.model.named_parameters():
                if param.is_cuda:
                    logger.info(f"Parameter {name} is on CUDA")
                    break
            else:
                logger.warning("No CUDA parameters found")

        # Test model forward pass with dummy input
        logger.info("Testing model forward pass...")
        dummy_input = torch.randn(1, obs_dim, device=device, dtype=torch.float32)
        with torch.no_grad():
            output = model.model(dummy_input)
            logger.info(f"Model output shape: {output.shape if hasattr(output, 'shape') else type(output)}")

        logger.info("Checkpoint loading test PASSED")
        return True

    except FileNotFoundError as e:
        logger.error(f"Checkpoint file not found: {e}")
        return False
    except ValueError as e:
        logger.error(f"Checkpoint loading failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test checkpoint loading on Jetson")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--action_dim", type=int, required=True, help="Action dimension")
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation dimension")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")

    args = parser.parse_args()

    success = test_checkpoint_loading(
        checkpoint_path=args.checkpoint,
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        device=args.device,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

