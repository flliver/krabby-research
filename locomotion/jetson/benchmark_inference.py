"""Script to benchmark inference latency on Jetson.

This script measures inference latency to ensure it meets real-time requirements (< 15ms target).
"""

import argparse
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from hal.client.observation.types import NavigationCommand
from compute.parkour.types import ParkourModelIO, ParkourObservation, OBS_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_dummy_model_io(action_dim: int, obs_dim: int) -> ParkourModelIO:
    """Create dummy model IO for testing.

    Args:
        action_dim: Action dimension
        obs_dim: Observation dimension (should be OBS_DIM = 753)

    Returns:
        Model IO with dummy data
    """
    nav_cmd = NavigationCommand.create_now(vx=0.0, vy=0.0, yaw_rate=0.0)

    # Create complete observation array in training format
    observation_array = np.zeros(obs_dim, dtype=np.float32)

    observation = ParkourObservation(
        timestamp_ns=time.time_ns(),
        schema_version="1.0",
        observation=observation_array,
    )

    return ParkourModelIO(
        timestamp_ns=time.time_ns(),
        schema_version="1.0",
        nav_cmd=nav_cmd,
        observation=observation,
    )


def benchmark_inference(
    checkpoint_path: str,
    action_dim: int,
    obs_dim: int,
    device: str = "cuda",
    num_iterations: int = 100,
    warmup_iterations: int = 10,
):
    """Benchmark inference latency.

    Args:
        checkpoint_path: Path to checkpoint file
        action_dim: Action dimension
        obs_dim: Observation dimension
        device: Device to use ("cuda" or "cpu")
        num_iterations: Number of inference iterations to benchmark
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking inference on {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Action dim: {action_dim}, Obs dim: {obs_dim}")
    logger.info(f"Iterations: {num_iterations} (warmup: {warmup_iterations})")

    # Load model
    weights = ModelWeights(
        checkpoint_path=checkpoint_path,
        action_dim=action_dim,
        obs_dim=obs_dim,
    )
    model = ParkourPolicyModel(weights, device=device)

    # Create dummy input
    model_io = create_dummy_model_io(action_dim, obs_dim)

    # Warmup
    logger.info("Running warmup iterations...")
    for _ in range(warmup_iterations):
        _ = model.inference(model_io)

    # Synchronize if using CUDA
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    logger.info("Running benchmark iterations...")
    latencies_ms = []

    for i in range(num_iterations):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time_ns = time.time_ns()
        result = model.inference(model_io)
        end_time_ns = time.time_ns()

        if device == "cuda":
            torch.cuda.synchronize()

        latency_ns = end_time_ns - start_time_ns
        latency_ms = latency_ns / 1_000_000.0
        latencies_ms.append(latency_ms)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{num_iterations} iterations")

    # Calculate statistics
    results = {
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": np.percentile(latencies_ms, 95),
        "p99_ms": np.percentile(latencies_ms, 99),
        "num_iterations": num_iterations,
        "device": device,
    }

    # Log results
    logger.info("=" * 60)
    logger.info("Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Mean latency: {results['mean_ms']:.3f} ms")
    logger.info(f"Median latency: {results['median_ms']:.3f} ms")
    logger.info(f"Min latency: {results['min_ms']:.3f} ms")
    logger.info(f"Max latency: {results['max_ms']:.3f} ms")
    logger.info(f"Std deviation: {results['std_ms']:.3f} ms")
    logger.info(f"P50 (median): {results['p50_ms']:.3f} ms")
    logger.info(f"P95: {results['p95_ms']:.3f} ms")
    logger.info(f"P99: {results['p99_ms']:.3f} ms")
    logger.info("=" * 60)

    # Check if meets target
    target_ms = 15.0
    if results["mean_ms"] < target_ms:
        logger.info(f"✓ Mean latency ({results['mean_ms']:.3f} ms) < target ({target_ms} ms)")
    else:
        logger.warning(f"✗ Mean latency ({results['mean_ms']:.3f} ms) >= target ({target_ms} ms)")

    if results["p95_ms"] < target_ms:
        logger.info(f"✓ P95 latency ({results['p95_ms']:.3f} ms) < target ({target_ms} ms)")
    else:
        logger.warning(f"✗ P95 latency ({results['p95_ms']:.3f} ms) >= target ({target_ms} ms)")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark inference latency on Jetson")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--action_dim", type=int, required=True, help="Action dimension")
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation dimension")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")

    args = parser.parse_args()

    results = benchmark_inference(
        checkpoint_path=args.checkpoint,
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        device=args.device,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )

    # Exit with error if latency too high
    if results["mean_ms"] >= 15.0 or results["p95_ms"] >= 15.0:
        logger.error("Inference latency does not meet target (< 15ms)")
        exit(1)


if __name__ == "__main__":
    main()

