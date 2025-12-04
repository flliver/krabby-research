# Krabby Compute - Parkour

Parkour policy inference client and utilities for the Krabby quadruped robot.

## Overview

This package provides the parkour policy inference client that:
- Connects to HAL server via ZMQ
- Polls hardware observations
- Runs parkour policy inference
- Sends joint commands back to HAL server

### Components

#### Inference Client (`compute.parkour.inference_client.ParkourInferenceClient`)
- Runs in a separate thread
- Manages HAL client connection
- Handles inference loop (poll → infer → command)

#### Policy Interface (`compute.parkour.policy_interface.ParkourPolicyModel`)
- Loads parkour policy checkpoints
- Runs inference on observations
- Uses OnPolicyRunnerWithExtractor for model loading

#### Mappers (`compute.parkour.mappers`)
- **hardware_to_model**: Maps Krabby hardware observations to parkour model format
- **model_to_hardware**: Maps parkour model actions to Krabby joint positions

#### Types (`compute.parkour.parkour_types`)
- `ParkourObservation`: Observation in training format
- `ParkourModelIO`: Combined input for policy inference
- `InferenceResponse`: Policy inference output with action tensor

## Installation

### From source (development)

```bash
cd compute/parkour
pip install -e .
```

### From wheel

```bash
pip install krabby-compute-parkour-0.1.0-py3-none-any.whl
```

## Usage

### As a library

```python
from compute.parkour.inference_client import ParkourInferenceClient
from compute.parkour.policy_interface import ModelWeights
from hal.client.config import HalClientConfig

# Configure HAL client
hal_config = HalClientConfig(
    observation_endpoint="inproc://hal_observation",
    command_endpoint="inproc://hal_commands",
)

# Configure model
model_weights = ModelWeights(
    checkpoint_path="/path/to/model.pt",
    action_dim=12,
    obs_dim=753,
)

# Create client
client = ParkourInferenceClient(
    hal_client_config=hal_config,
    model_weights=model_weights,
    control_rate=100.0,
    device="cuda",
    transport_context=transport_context,  # From HAL server
)

# Initialize and start
client.initialize()
client.start_thread(running_flag=lambda: True)
```

## Architecture

```
┌──────────────────────────────────────┐
│  ParkourInferenceClient              │
│                                      │
│  ┌────────────┐    ┌──────────────┐ │
│  │ HAL Client │───▶│ Policy Model │ │
│  └────────────┘    └──────────────┘ │
│        │                  │          │
│        │ observations     │ actions  │
│        ▼                  ▼          │
│  ┌──────────────────────────────┐   │
│  │   Hardware ↔ Model Mappers   │   │
│  └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

## Dependencies

- `krabby-hal-client`: For HAL communication
- `torch`: For policy inference
- `numpy`: For numerical operations

## Development

### Running Tests

```bash
pytest tests/
```

### Building Wheel

```bash
python -m build
```

## Notes

- Designed to run in a separate thread from HAL server
- Supports both inproc (same-process) and TCP (distributed) communication
- Zero-copy operations where possible for performance
