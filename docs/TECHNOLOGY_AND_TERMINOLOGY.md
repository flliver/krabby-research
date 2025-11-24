# Krabby Research - Technology and Terminology

## Project Overview

**Krabby Research** specifically focuses on training legged robots to perform extreme parkour locomotion. The project is built on top of NVIDIA's Isaac Lab framework and implements a teacher-student reinforcement learning approach to enable robots to navigate complex, dynamically generated parkour terrains.

The project is based on the Extreme-Parkour research (https://extreme-parkour.github.io/), which demonstrates how legged robots can learn to perform challenging parkour maneuvers through simulation-based training. The system trains two types of policies:

1. **Teacher Policy**: A privileged policy that has access to full state information and terrain knowledge
2. **Student Policy**: A more realistic policy that uses only visual observations (depth images) and proprioceptive data, trained via knowledge distillation from the teacher

Key files:
- Main environment: `parkour/parkour_isaaclab/envs/parkour_manager_based_rl_env.py`
- Training script: `parkour/scripts/rsl_rl/train.py`
- README: `parkour/README.md`

## Technologies Used

### Core Technologies

#### **Python 3.11**
- **What it is**: A high-level programming language
- **Usage**: The entire project is written in Python
- **Why**: Python is the standard language for machine learning and robotics research, with extensive library support

#### **PyTorch 2.7.0**
- **What it is**: An open-source deep learning framework developed by Meta (Facebook)
- **Usage**: Used for all neural network operations, including:
  - Actor-critic networks for reinforcement learning
  - Feature extractors for processing depth images
  - State encoders and history encoders
- **Where it fits**: 
  - Neural network definitions: `parkour/scripts/rsl_rl/modules/actor_critic_with_encoder.py`
  - Training loops: `parkour/scripts/rsl_rl/train.py`
  - PPO implementation: `parkour/scripts/rsl_rl/modules/ppo_with_extractor.py`
- **Why**: PyTorch provides efficient GPU-accelerated tensor operations and automatic differentiation, essential for training deep RL policies

#### **PyTorch Checkpoint Files (.pt, .pth)**
- **What it is**: PyTorch's native format for saving and loading trained model weights, optimizer states, and training metadata
- **Usage**: Used to store trained parkour policy models:
  - Model weights (neural network parameters)
  - Optimizer state (for resuming training)
  - Training metadata (epoch, loss, hyperparameters)
  - Normalization statistics (for observation preprocessing)
- **File formats**:
  - `.pt` - PyTorch checkpoint file (commonly used)
  - `.pth` - Alternative PyTorch checkpoint extension (same format)
- **Where it fits**: 
  - Training checkpoints: Saved during training via `OnPolicyRunnerWithExtractor`
  - Model loading: Used by `compute/parkour/policy_interface.py` to load trained policies for inference
  - Checkpoint management: `compute/parkour/model_loader.py` handles loading and validation
- **Why**: PyTorch checkpoints preserve the complete model state, enabling:
  - Resuming training from a specific point
  - Deploying trained models for inference
  - Transferring models between environments
  - Version control of model weights and training state

#### **ONNX (Open Neural Network Exchange)**
- **What it is**: An open standard format for representing machine learning models, enabling interoperability between different frameworks and inference engines
- **Usage**: Optional format for exporting PyTorch models for optimized inference:
  - Export PyTorch models to ONNX format for deployment
  - Use with TensorRT on Jetson for optimized inference
  - Cross-platform model deployment (not limited to PyTorch)
- **File format**: `.onnx` - Binary format containing model graph and weights
- **Where it fits**: 
  - Model export: Convert PyTorch checkpoints to ONNX for deployment
  - Jetson optimization: ONNX models can be converted to TensorRT engines for faster inference
  - Cross-framework compatibility: Models can be loaded in different inference engines
- **Why**: ONNX provides:
  - Framework-agnostic model format
  - Optimized inference paths (TensorRT, ONNX Runtime)
  - Smaller model size and faster loading compared to PyTorch checkpoints
  - Better performance on edge devices (Jetson) when converted to TensorRT
- **Note**: While ONNX export is optional currently, it may be beneficial for Jetson deployment to achieve lower latency and better resource utilization

#### **CUDA 13.0**
- **What it is**: NVIDIA's parallel computing platform and programming model for GPU acceleration
- **Usage**: Enables GPU-accelerated computation for:
  - Neural network training (forward and backward passes)
  - Physics simulation in Isaac Sim
  - Parallel environment execution (running thousands of simulations simultaneously)
- **Where it fits**: Required for both PyTorch operations and Isaac Sim physics simulation
- **Why**: GPU acceleration is critical for training RL policies efficiently, as the project runs thousands of parallel environments (up to 12,288 environments simultaneously)

### Simulation & Physics

#### **Isaac Sim 5.1.0**
- **What it is**: NVIDIA's high-performance robotics simulation platform built on Omniverse
- **Usage**: Provides the physics engine and rendering capabilities for:
  - Simulating robot dynamics and physics
  - Rendering visual observations (RGB and depth cameras)
  - Managing scene assets and environments
- **Where it fits**: 
  - Environment setup: `parkour/parkour_isaaclab/envs/parkour_manager_based_env.py`
  - Scene configuration: `parkour/parkour_tasks/parkour_tasks/default_cfg.py`
- **Why**: Isaac Sim provides highly accurate physics simulation and GPU-accelerated rendering, enabling fast parallel simulation of many robot instances

#### **Isaac Lab**
- **What it is**: A unified framework for robot learning built on top of Isaac Sim, providing standardized interfaces for RL environments
- **Usage**: Provides:
  - Manager-based environment architecture (ActionManager, ObservationManager, CommandManager, etc.)
  - Standardized RL environment interfaces
  - Terrain generation utilities
  - Scene management and asset loading
- **Where it fits**: 
  - Base environment class: `parkour/parkour_isaaclab/envs/parkour_manager_based_env.py` (inherits from `ManagerBasedEnv`)
  - Terrain generation: `parkour/parkour_isaaclab/terrains/parkour_terrain_generator.py` (inherits from `TerrainGenerator`)
- **Why**: Isaac Lab standardizes the interface between RL algorithms and simulation, making it easier to develop and test robot learning algorithms

#### **USD (Universal Scene Description)**
- **What it is**: An open-source framework and file format developed by Pixar for describing, composing, simulating, and collaborating on 3D scenes. USD provides a hierarchical, non-destructive editing model for complex 3D data.
- **Usage**: Used by Isaac Sim as the underlying scene format for:
  - Robot model descriptions (geometry, materials, physics properties)
  - Scene composition and hierarchy
  - Environment and terrain definitions
  - Asset references and instancing
  - Non-destructive scene editing and layering
- **Where it fits**: 
  - Isaac Sim uses USD files (`.usd`, `.usda`, `.usdc`) to represent all scene data
  - Robot models can be defined in USD format (in addition to URDF)
  - Terrain and environment assets are stored as USD files
  - Scene composition uses USD's composition arcs (references, inherits, variants, etc.)
- **Why**: USD provides a powerful, scalable format for complex 3D scenes that enables:
  - Efficient scene composition and collaboration
  - Non-destructive editing workflows
  - High-performance scene traversal and rendering
  - Support for large, complex scenes with many assets
  - Standardized format that works across different tools (Isaac Sim, Omniverse, Blender, etc.)
- **File Formats**:
  - `.usd` - Binary format (compact, fast loading)
  - `.usda` - ASCII format (human-readable, editable)
  - `.usdc` - Crate format (optimized binary, fastest loading)

### Reinforcement Learning

#### **RSL-RL (Rapid Motor Adaptation - Reinforcement Learning)**
- **What it is**: A reinforcement learning library originally developed for the RMA (Rapid Motor Adaptation) algorithm, now extended for general RL training
- **Usage**: Provides:
  - PPO (Proximal Policy Optimization) algorithm implementation
  - On-policy training runners
  - Vectorized environment wrappers
  - Checkpointing and logging utilities
- **Where it fits**: 
  - Training runner: `parkour/scripts/rsl_rl/modules/on_policy_runner_with_extractor.py`
  - PPO algorithm: `parkour/scripts/rsl_rl/modules/ppo_with_extractor.py`
  - Vectorized environment wrapper: `parkour/scripts/rsl_rl/vecenv_wrapper.py`
- **Why**: RSL-RL provides efficient implementations of on-policy RL algorithms optimized for robotics applications, with support for privileged learning and feature extraction

#### **PPO (Proximal Policy Optimization)**
- **What it is**: A policy gradient algorithm for reinforcement learning that uses clipped surrogate objectives to ensure stable policy updates
- **Usage**: The core RL algorithm used to train both teacher and student policies
- **Where it fits**: 
  - Algorithm implementation: `parkour/scripts/rsl_rl/modules/ppo_with_extractor.py`
  - Configuration: `parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_teacher_ppo_cfg.py`
- **Why**: PPO is a state-of-the-art on-policy algorithm that balances sample efficiency with training stability, making it well-suited for robotics applications

#### **Knowledge Distillation**
- **What it is**: A machine learning technique where a smaller "student" model learns from a larger "teacher" model by mimicking its behavior
- **Usage**: The student policy (which only has access to visual observations) learns from the teacher policy (which has privileged information)
- **Where it fits**: 
  - Distillation implementation: `parkour/scripts/rsl_rl/modules/distillation_with_extractor.py`
  - Student configuration: `parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_ppo_cfg.py`
- **Why**: Allows training a more realistic policy (student) that doesn't require privileged information, while leveraging the knowledge of a more capable teacher policy

### Environment & Interface

#### **Gymnasium**
- **What it is**: The standard API for reinforcement learning environments (formerly OpenAI Gym)
- **Usage**: Provides the standard `gym.Env` interface that the RL environment implements
- **Where it fits**: 
  - Environment class: `parkour/parkour_isaaclab/envs/parkour_manager_based_rl_env.py` (inherits from `gym.Env`)
- **Why**: Gymnasium provides a standardized interface that makes the environment compatible with various RL libraries and tools

### Robot Modeling

#### **URDF (Unified Robot Description Format)**
- **What it is**: An XML-based file format used to describe the physical properties of robots, including their geometry, dynamics, and kinematics
- **Usage**: Defines the robot model structure:
  - Links (rigid bodies) and their inertial properties
  - Joints (connections between links) and their limits
  - Visual and collision geometries
  - Actuator specifications
- **Where it fits**: 
  - Robot model: `assets/crab_hex.urdf`
- **Why**: URDF is the standard format for robot descriptions in ROS and many simulation platforms, allowing robots to be easily imported into simulators

#### **Xacro (XML Macros)**
- **What it is**: An XML macro language that extends URDF, allowing parameterized and reusable robot descriptions
- **Usage**: Used in the URDF file to define reusable leg macros, making it easier to define multiple identical legs
- **Where it fits**: 
  - Robot model: `assets/crab_hex.urdf` (uses `xacro:macro` for leg definitions)
- **Why**: Xacro reduces code duplication and makes robot descriptions more maintainable

### Terrain Generation

#### **Trimesh**
- **What it is**: A Python library for loading and manipulating triangular meshes
- **Usage**: Used for generating 3D terrain meshes for parkour obstacles:
  - Creating geometric shapes for obstacles
  - Combining meshes to form complex terrain
  - Exporting meshes for physics simulation
- **Where it fits**: 
  - Terrain generator: `parkour/parkour_isaaclab/terrains/parkour_terrain_generator.py`
- **Why**: Trimesh provides efficient mesh manipulation capabilities needed to programmatically generate diverse parkour terrains

#### **NumPy**
- **What it is**: A fundamental library for numerical computing in Python, providing N-dimensional array objects and mathematical functions
- **Usage**: Used throughout the project for:
  - Array operations and mathematical computations
  - Terrain generation calculations
  - Data manipulation and preprocessing
- **Where it fits**: Used extensively across all modules
- **Why**: NumPy provides efficient array operations that are the foundation of scientific computing in Python

### Visualization & Monitoring

#### **TensorBoard**
- **What it is**: A visualization tool for TensorFlow and PyTorch that provides real-time monitoring of training metrics
- **Usage**: Tracks and visualizes:
  - Training loss curves
  - Episode rewards
  - Policy performance metrics
  - Learning rate schedules
- **Where it fits**: Logs are generated during training and can be viewed with TensorBoard
- **Why**: Essential for monitoring training progress and debugging RL training issues

### Communication & Hardware Abstraction

#### **ZMQ (ZeroMQ)**
- **What it is**: A high-performance asynchronous messaging library that provides sockets for various messaging patterns (PUB/SUB, REQ/REP, PUSH/PULL, etc.)
- **Usage**: Used as the communication layer for the Hardware Abstraction Layer (HAL) runtime, enabling:
  - Decoupled communication between the parkour policy runtime and hardware/simulation backends
  - Real-time exchange of camera frames, robot state, and joint commands
  - Support for both in-process (`inproc://`) and network (`tcp://`) transports
  - Minimal, explicit IPC contract that works identically with IsaacSim and Jetson hardware
- **Where it fits**: 
  - HAL client/server implementation (current)
  - Game loop container that drives the parkour policy
  - IsaacSim HAL server for simulation-based control
  - Jetson HAL server for real hardware control
- **Why**: ZMQ was chosen over ROS2 because it:
  - Keeps the HAL contract extremely small and explicit (a handful of sockets, topics, and flat tensors)
  - Is easy to embed in both IsaacLab and lightweight Jetson containers without bringing in a full robotics middleware stack
  - Is friendlier to automated agents modifying the codebase
  - Provides low-latency, high-throughput messaging suitable for real-time control loops (100+ Hz)
  - Supports both local (inproc) and distributed (tcp) communication patterns
- **ZMQ Architecture**: The HAL uses three main ZMQ channels:
  - **Camera telemetry** (PUB/SUB): Publishes depth/RGBD frames or pre-encoded depth features
  - **State telemetry** (PUB/SUB): Publishes robot/base pose and joint state
  - **Joint command service** (REQ/REP): Receives desired joint positions from the game loop and sends acknowledgements back
- **Message Format**: Messages use topic-prefixed multipart format with flat float32 numpy arrays for efficient serialization and minimal overhead

### Embedded Hardware

#### **JetPack**
- **What it is**: NVIDIA's software development kit (SDK) for Jetson embedded computing platforms, providing a complete Linux-based development environment
- **Usage**: Used for deploying the parkour policy runtime and HAL server on Jetson Orin hardware:
  - Provides Ubuntu-based Linux for Tegra (L4T) operating system
  - Includes CUDA, cuDNN, TensorRT, and other NVIDIA libraries pre-installed
  - Enables GPU-accelerated inference on embedded hardware
- **Where it fits**: 
  - Jetson HAL server deployment (`locomotion/backends/jetson/`)
  - Container images for Jetson (`images/locomotion/`)
- **Why**: JetPack provides a complete, optimized software stack for Jetson devices, enabling efficient deployment of AI/ML workloads on embedded hardware without manual library installation and configuration

#### **L4T (Linux for Tegra)**
- **What it is**: NVIDIA's custom Linux distribution optimized for Tegra SoCs (System-on-Chips) used in Jetson devices
- **Usage**: The base operating system layer of JetPack that runs on Jetson hardware:
  - Provides the Linux kernel, device drivers, and system libraries
  - Optimized for ARM64 architecture and NVIDIA GPU integration
  - Includes hardware-specific drivers for cameras, sensors, and peripherals
- **Where it fits**: 
  - Base OS for Jetson Orin when running JetPack
  - Required for all Jetson hardware deployments
- **Why**: L4T provides a stable, optimized Linux foundation specifically designed for Jetson hardware, ensuring proper hardware access and performance for robotics applications. JetPack builds on top of L4T to provide the complete development environment.

### Development Tools

#### **Hydra**
- **What it is**: A framework for elegantly configuring complex applications, allowing hierarchical configuration management
- **Usage**: Used for managing configuration files for:
  - Environment settings
  - Training hyperparameters
  - Task-specific parameters
- **Where it fits**: Configuration management throughout the project
- **Why**: Hydra simplifies managing complex configurations and enables easy experimentation with different hyperparameters

#### **Setuptools**
- **What it is**: A Python package management tool for building and distributing Python packages
- **Usage**: Used to package the project modules for installation
- **Where it fits**: 
  - Package configuration: `parkour/setup.py`
  - Build configuration: `parkour/pyproject.toml`
- **Why**: Enables the project to be installed as a Python package, making imports and dependencies manageable

## Architecture Overview

The project follows a modular architecture with clear separation between simulation, training, and deployment:

### Core Training Architecture

1. **Environment Layer** (`parkour/parkour_isaaclab/envs/`): Defines the RL environment with managers for actions, observations, rewards, and terminations
2. **Terrain Layer** (`parkour/parkour_isaaclab/terrains/`): Generates diverse parkour terrains programmatically
3. **RL Layer** (`parkour/scripts/rsl_rl/`): Implements the training algorithms, neural networks, and training loops
4. **Task Configuration** (`parkour/parkour_tasks/`): Defines specific task configurations for different robot models and training scenarios
5. **Assets** (`assets/`): Contains robot URDF models and other static assets

### Runtime & Hardware Abstraction Architecture

The project implements a Hardware Abstraction Layer (HAL) using ZMQ for communication, enabling the same parkour policy to run on both simulation and real hardware:

1. **Game Loop (Inference Logic)**: 
   - The core inference logic that runs independently from simulation/hardware
   - Implements a control loop at 100+ Hz that:
     - Polls HAL for latest data (navigation commands, observations)
     - Builds observation tensors matching the training format
     - Runs policy inference to generate joint commands
     - Sends joint commands back via ZMQ
   - Production: `locomotion/jetson/inference_runner.py::InferenceRunner.run()`
   - Testing: `compute/testing/inference_test_runner.py::InferenceTestRunner` (simulates game loop)

2. **HAL Client**:
   - ZMQ-based client that subscribes to telemetry (camera, state) and sends commands
   - Maintains latest-only buffers to avoid stale data
   - Supports both `inproc://` (local) and `tcp://` (network) transports

3. **HAL Server Interface**:
   - Generic base class defining the ZMQ contract (topics, message formats, endpoints)
   - Ensures consistent communication semantics across all backends

4. **IsaacSim HAL Server**:
   - Runs inside IsaacLab process
   - Publishes camera frames and robot state from simulation
   - Receives joint commands and applies them to the simulated robot
   - Uses the same observation/action semantics as training

5. **Jetson HAL Server**:
   - Runs on Jetson Orin hardware (JetPack/L4T - see Embedded Hardware section for details)
   - Publishes real camera frames (ZED Mini) and robot state
   - Receives joint commands for real actuators
   - Mirrors the IsaacSim server interface for seamless switching

This architecture allows the parkour policy to be completely decoupled from the execution backend, enabling:
- Development and testing in simulation
- Deployment to real hardware with minimal changes
- Easy switching between simulation and hardware via configuration
- Future extensibility to additional backends or communication patterns

## Key Features

- **Parallel Simulation**: Runs thousands of environments simultaneously for efficient data collection
- **Privileged Learning**: Teacher policy uses full state information for faster learning
- **Visual Learning**: Student policy learns from depth images, making it more transferable to real robots
- **Dynamic Terrain**: Programmatically generated parkour obstacles with varying difficulty
- **Knowledge Distillation**: Efficient transfer of knowledge from teacher to student policy
- **Hardware Abstraction**: ZMQ-based HAL enables seamless switching between simulation and real hardware
- **Deployable Runtime**: Parkour policy runs as standalone container, decoupled from simulation

## System Requirements

As documented in `DEVELOPER.md`:
- **OS**: Ubuntu 24.04
- **Kernel**: 6.14.0
- **GPU**: NVIDIA RTX 5080 (or compatible CUDA-capable GPU)
- **NVIDIA Driver**: `nvidia-driver-580-open`
- **CUDA**: 13.0
- **Python**: 3.11
- **PyTorch**: 2.7.0 with CUDA 13.0 support

## References

- Base research: Extreme-Parkour (https://extreme-parkour.github.io/)
- Isaac Lab: Isaac Lab Documentation (https://isaac-sim.github.io/IsaacLab/)
- Original paper: Cheng et al., "Extreme Parkour with Legged Robots" (arXiv:2309.14341)

