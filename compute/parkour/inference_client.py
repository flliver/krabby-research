"""Parkour inference client using HAL for observations and commands.

This module provides a client that:
1. Polls observations from HAL server
2. Runs parkour policy inference
3. Sends joint commands back to HAL server

Designed to run in a separate thread from the simulation/HAL server.
"""

import logging
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch
import zmq

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.client.observation.types import NavigationCommand
from hal.client.data_structures.hardware import JointCommand
from compute.parkour.model_definition import ObservationDimensions
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
from compute.parkour.parkour_types import ParkourModelIO, TeacherObservation
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)


class ParkourInferenceClient(HalClient):
    """Parkour inference client that extends HAL client.

    This is a HAL client that also runs parkour policy inference.
    It extends HalClient to inherit all HAL communication functionality,
    and adds inference capabilities on top.

    Attributes:
        model: Parkour policy model
        control_rate: Control loop rate in Hz
    """

    def __init__(
        self,
        hal_client_config: HalClientConfig,
        model_weights: ModelWeights,
        observation_dimensions: ObservationDimensions,
        robot_definition: RobotDefinition,
        control_rate: float = 100.0,
        device: str = "cuda",
        transport_context: Optional[zmq.Context] = None,
        use_env_observations: bool = False,
        stm: bool = False,
    ):
        """Initialize Parkour inference client.

        Args:
            hal_client_config: HAL client configuration
            model_weights: Model weights configuration
            observation_dimensions: Layout from model_definition.get_observation_dimensions(robot_definition)
            robot_definition: Robot definition; command joint count = get_total_joint_count()
            control_rate: Control loop rate in Hz
            device: Device for inference ("cuda" or "cpu")
            transport_context: ZMQ context for inproc connections (required for inproc)
            use_env_observations: If True, use environment observations directly instead of hardware observations
            stm: When True, map model actions to duty [-1, 1] for STM CAN HAL.
        """
        super().__init__(hal_client_config, context=transport_context)
        self.model_weights = model_weights
        self.observation_dimensions = observation_dimensions
        self.robot_definition = robot_definition
        self.control_rate = control_rate
        self.device = device
        self.use_env_observations = use_env_observations
        self._stm = stm
        self.model: Optional[ParkourPolicyModel] = None
        self.nav_cmd: Optional[NavigationCommand] = None
        self._inference_initialized = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mapper = (
            HWObservationsToParkourMapper(observation_dimensions) if not use_env_observations else None
        )
        self._env = None
        self._non_finite_drop_count = 0
        self._missing_action_drop_count = 0

    def initialize(self) -> None:
        """Initialize HAL client and policy model."""
        # Initialize base HalClient first
        super().initialize()
        
        if self._inference_initialized:
            logger.warning("ParkourInferenceClient inference already initialized")
            return

        logger.info(f"Loading parkour policy model from {self.model_weights.checkpoint_path}")
        self.model = ParkourPolicyModel(self.model_weights, device=self.device)

        # Initialize navigation command with default values
        self.nav_cmd = NavigationCommand.create_now(vx=0.0, vy=0.0, yaw_rate=0.0)

        self._inference_initialized = True
        logger.info("ParkourInferenceClient initialized")
    
    def process_env_observation(self, obs_tensor: torch.Tensor, extras: Optional[dict] = None) -> Optional["JointCommand"]:
        """Process environment observation directly (for comparison testing).
        
        This method bypasses HAL polling and hardware observation mapping,
        using environment observations directly. This ensures identical
        observation processing as the environment's native path.
        
        Args:
            obs_tensor: Observation tensor from environment (shape: [num_envs, obs_dim])
            extras: Optional extras dict from environment
        
        Returns:
            JointCommand if inference succeeded
            
        Raises:
            RuntimeError: If inference fails
        """
        if not self._inference_initialized:
            raise RuntimeError("Inference client not initialized. Call initialize() first.")
        if self.model is None:
            raise RuntimeError("Policy model not loaded. Call initialize() first.")
        
        # Ensure tensor is on correct device
        obs_tensor = obs_tensor.to(self.device)
        
        # Apply estimator to fill privileged explicit features
        # NOTE: Do NOT normalize before estimator - normalization happens inside policy function
        # The estimator is applied to raw observations, then policy normalizes internally
        # Get estimator from policy model
        estimator = self.model.runner.get_estimator_inference_policy(device=str(self.device))
        
        d = self.observation_dimensions
        # Use torch.inference_mode() to disable autograd and improve performance
        # Call estimator.inference() which has its own torch.no_grad() inside
        with torch.inference_mode():
            estimator_output = estimator.inference(obs_tensor[:, : d.num_prop])
        
        priv_explicit_start = d.num_prop + d.num_scan + d.num_vision
        priv_explicit_end = priv_explicit_start + d.num_priv_explicit
        obs_tensor[:, priv_explicit_start:priv_explicit_end] = estimator_output
        
        # Use direct tensor inference to avoid numpy conversions
        # inference_tensor() works directly with tensors, avoiding the tensor → numpy → tensor
        # conversion that would occur with inference() (which takes ParkourModelIO)
        inference_result = self.model.inference_tensor(obs_tensor)
        
        if not inference_result.success:
            raise RuntimeError(f"Inference failed: {inference_result.error_message}")
        
        # Map inference response to hardware joint positions
        hw_mapper = ParkourLocomotionToHWMapper(self.robot_definition, stm=self._stm)
        joint_cmd = hw_mapper.map(inference_result, observation_timestamp_ns=time.time_ns())
        
        # Update timestamp to current time
        joint_cmd.timestamp_ns = time.time_ns()
        
        # Send command via HAL (using inherited method)
        self.send_joint_command(joint_cmd)
        
        return joint_cmd

    def _inference_step(self) -> bool:
        """Execute one inference step.

        Polls observation, runs inference, and sends command.

        Returns:
            True if step succeeded, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Base HAL client not initialized. Call initialize() first.")
        if not self._inference_initialized:
            raise RuntimeError("Inference client not initialized. Call initialize() first.")
        if self.model is None:
            raise RuntimeError("Policy model not loaded. Call initialize() first.")

        # Poll HAL for hardware observation (same path for simulation and hardware)
        hw_obs = self.poll(timeout_ms=1)

        if hw_obs is None:
            # No new observation available - this is normal, just continue
            return None  # Return None to indicate no observation received

        # Capture current timestamp for synchronization
        current_timestamp_ns = time.time_ns()
        
        # Map hardware observation to model observation format
        # Create a new nav_cmd with current timestamp
        nav_cmd = NavigationCommand.create_now()
        nav_cmd.timestamp_ns = current_timestamp_ns
        
        # Use persistent mapper (maintains history buffer across steps)
        # Mapper is initialized in initialize(), so it should always be available here
        if self._mapper is None:
            raise RuntimeError("Mapper not initialized. Cannot process hardware observations.")
        model_obs = self._mapper.map(hw_obs, nav_cmd=nav_cmd)
        
        # Apply estimator to fill privileged explicit features
        # The estimator always overwrites privileged explicit features from physics with estimated values
        # Formula: obs[num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator(obs[:num_prop])
        
        # Get estimator from policy model
        estimator = self.model.runner.get_estimator_inference_policy(device=str(self.device))
        
        # Convert observation to array only at model input, then to tensor
        obs_tensor = torch.from_numpy(model_obs.to_array()).unsqueeze(0).to(self.device)
        
        d = self.observation_dimensions
        with torch.inference_mode():
            estimator_output = estimator(obs_tensor[:, : d.num_prop])
        priv_explicit_start = d.num_prop + d.num_scan + d.num_vision
        priv_explicit_end = priv_explicit_start + d.num_priv_explicit
        obs_tensor[:, priv_explicit_start:priv_explicit_end] = estimator_output

        # Update both observation and nav_cmd timestamps to use the captured timestamp
        # This ensures synchronization between observation and nav_cmd
        model_obs.timestamp_ns = current_timestamp_ns
        nav_cmd.timestamp_ns = current_timestamp_ns

        # Use direct tensor inference to avoid numpy conversions (tensor → numpy → tensor)
        # This matches the optimized path in process_env_observation()
        inference_result = self.model.inference_tensor(obs_tensor)

        if not inference_result.success:
            logger.error(f"Inference failed: {inference_result.error_message}")
            return False

        # Guard against invalid model output (NaN/Inf) to keep inference thread alive.
        action_tensor = inference_result.get_action()
        if action_tensor is None:
            self._missing_action_drop_count += 1
            if self._missing_action_drop_count == 1 or self._missing_action_drop_count % 100 == 0:
                logger.warning(
                    "Inference produced no action tensor; dropping step (count=%d)",
                    self._missing_action_drop_count,
                )
            return False
        self._missing_action_drop_count = 0
        if isinstance(action_tensor, torch.Tensor):
            finite_ok = bool(torch.isfinite(action_tensor).all().item())
        else:
            finite_ok = bool(np.isfinite(np.asarray(action_tensor)).all())
        if not finite_ok:
            self._non_finite_drop_count += 1
            if self._non_finite_drop_count == 1 or self._non_finite_drop_count % 100 == 0:
                logger.warning(
                    "Inference produced non-finite action values (NaN/Inf); dropping step (count=%d)",
                    self._non_finite_drop_count,
                )
            return False
        self._non_finite_drop_count = 0

        # Map inference response to hardware joint positions
        hw_mapper = ParkourLocomotionToHWMapper(self.robot_definition, stm=self._stm)
        try:
            joint_cmd = hw_mapper.map(
                inference_result,
                observation_timestamp_ns=hw_obs.timestamp_ns,
            )
        except ValueError as e:
            logger.warning("Failed to map inference output to joint command: %s", e)
            return False
        
        # Update timestamp to current time
        joint_cmd.timestamp_ns = time.time_ns()

        # Send command back to HAL server (using inherited method)
        self.put_joint_command(joint_cmd)
        
        # Update mapper with previous action for next step (model uses observation_joint_count)
        self._mapper.set_previous_action(
            joint_cmd.to_positions_dict(), joint_cmd.joint_names
        )
        
        return True

    def _run_loop(self, running_flag) -> None:
        """Run inference loop at control rate.

        Args:
            running_flag: Callable that returns True while loop should continue
        """
        import threading
        logger.info(f"Starting inference loop at {self.control_rate} Hz")
        period_s = 1.0 / self.control_rate

        iteration = 0
        observations_received = 0
        commands_sent = 0
        no_observation_count = 0
        step_failure_count = 0
        while self._running and running_flag():
            iteration += 1
            loop_start_ns = time.time_ns()

            # Execute inference step
            try:
                step_result = self._inference_step()
            except Exception as e:
                step_result = False
                logger.warning("Inference step raised unexpected exception; continuing: %s", e, exc_info=True)
            if step_result is True:
                # Track successful steps (when observation was received and processed)
                observations_received += 1
                commands_sent += 1
                no_observation_count = 0  # Reset counter on success
                step_failure_count = 0
            elif step_result is False:
                step_failure_count += 1
                if step_failure_count == 1 or step_failure_count % 100 == 0:
                    logger.warning(
                        "Inference step failed (count=%d), continuing...",
                        step_failure_count,
                    )
            else:
                # step_result is None means no observation available
                no_observation_count += 1
                # Log warning if we haven't received observations for a while
                if no_observation_count == 100:
                    logger.warning(f"Inference loop: No observations received for {no_observation_count} iterations. Is HAL server publishing?")

            # Timing control
            loop_end_ns = time.time_ns()
            loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
            sleep_time = max(0.0, period_s - loop_duration_s)

            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Inference loop stopped")

    def start_thread(self, running_flag=lambda: True) -> None:
        """Start inference loop in a separate thread.

        Args:
            running_flag: Callable that returns True while loop should continue
        """
        # Detailed verification before starting thread
        if not self._initialized:
            raise RuntimeError("Base HAL client not initialized. Call initialize() first.")
        if not self._inference_initialized:
            raise RuntimeError("Inference client not initialized. Call initialize() first.")
        if self.model is None:
            raise RuntimeError("Policy model not loaded. Call initialize() first.")

        if self._running:
            logger.warning("Inference thread already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(running_flag,),
            daemon=True,
            name="parkour-inference",
        )
        self._thread.start()
        logger.info("Inference thread started")

    def stop_thread(self, timeout: float = 5.0) -> None:
        """Stop inference thread.

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)
        """
        if not self._running or self._thread is None:
            logger.warning("Inference thread not running")
            return

        logger.info("Stopping inference thread...")
        self._running = False

        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Inference thread did not stop within timeout")

        self._thread = None
        logger.info("Inference thread stopped")

    def set_navigation_command(self, nav_cmd: NavigationCommand) -> None:
        """Set navigation command for inference.

        Args:
            nav_cmd: Navigation command (vx, vy, yaw_rate)
        """
        self.nav_cmd = nav_cmd

    def close(self) -> None:
        """Close HAL client and clean up resources."""
        if self._running:
            self.stop_thread()

        # Close base HalClient
        super().close()

        self._inference_initialized = False
        logger.info("ParkourInferenceClient closed")
