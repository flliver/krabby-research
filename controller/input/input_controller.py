"""InputController singleton for gamepad input using Pygame SDL2 Game Controller API.

Uses pygame's SDL2 Game Controller API (pygame._sdl2.controller) for logical axis/button
mapping, so behavior is consistent across macOS, Linux (e.g. Jetson Orin), and Windows.
SDL's controller mapping database normalizes different physical controllers to the same
logical layout (left stick, right stick, triggers, shoulder buttons, stick clicks).

After idle Bluetooth power-off (or USB unplug), the pad may reappear as a new SDL device.
This class detects disconnect and reopens the controller without restarting the process.

Testing:
--------
1. Install pygame: pip install pygame
2. Use the monitor command: python -m controller.input --monitor
3. List controller-capable devices: python -m controller.input --list
"""
import logging
import threading
import time
from typing import Callable, Optional

try:
    import pygame
    import pygame._sdl2.controller as sdl2_controller
except ImportError as e:
    if "_sdl2" in str(e) or "controller" in str(e):
        raise ImportError(
            "pygame SDL2 controller module not available. Install pygame 2.6+ with SDL2."
        ) from e
    raise ImportError(
        "pygame library not installed. Install with: pip install pygame"
    ) from e

from controller.input.state import ControllerState

logger = logging.getLogger(__name__)

# SDL2 controller get_axis() returns int: sticks -32768..32767, triggers 0..32768
_AXIS_SCALE = 32768.0

# While disconnected, rescan the joystick subsystem this often (seconds).
_RECONNECT_SCAN_INTERVAL_S = 1.0
# Throttle "still waiting for controller" logs (seconds).
_RECONNECT_LOG_INTERVAL_S = 5.0


class InputController:
    """Singleton controller for reading and storing gamepad input state.

    This class provides a thread-safe interface for reading gamepad events
    and storing them in a normalized ControllerState dataclass.
    Uses pygame SDL2 Game Controller API for cross-platform logical mapping.

    Usage:
        controller = InputController.get_instance()
        controller.start(device_id=0, update_rate_hz=50)
        # ... use controller.get_state() or register callbacks
        controller.stop()
    """

    _instance: Optional["InputController"] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize InputController (private, use get_instance())."""
        # Check if an instance already exists and it's not this instance
        if InputController._instance is not None and InputController._instance is not self:
            raise RuntimeError(
                "InputController is a singleton. Use get_instance() instead."
            )

        self._state = ControllerState()
        self._state_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._device_id: Optional[int] = None
        self._update_rate_hz = 50.0   # once every 20ms
        self._callbacks: list[Callable[[ControllerState], None]] = []
        self._callback_lock = threading.Lock()
        self._controller: Optional[sdl2_controller.Controller] = None
        self._controller_instance_id: Optional[int] = None

    @classmethod
    def get_instance(cls) -> "InputController":
        """Get the singleton instance of InputController."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__()
        return cls._instance

    def start(
        self,
        device_id: Optional[int] = None,
        update_rate_hz: float = 50.0,
    ) -> None:
        """Start the input controller event loop.

        Args:
            device_id: Optional device ID to use. If None, uses first available gamepad.
            update_rate_hz: Target update rate for processing controls (default: 50.0).
        """
        if self._running:
            logger.warning("InputController is already running")
            return

        if update_rate_hz <= 0:
            raise ValueError(
                f"update_rate_hz must be greater than 0, got {update_rate_hz}"
            )

        self._update_rate_hz = update_rate_hz
        self._device_id = device_id if device_id is not None else 0

        # Reset state
        with self._state_lock:
            self._state = ControllerState()

        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name="InputController"
        )
        self._thread.start()
        logger.info(
            f"InputController started (device_id={self._device_id}, "
            f"update_rate={update_rate_hz} Hz)"
        )

    def stop(self) -> None:
        """Stop the input controller event loop."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("InputController thread did not stop cleanly")

        self._close_controller()

        logger.info("InputController stopped")

    def get_state(self) -> ControllerState:
        """Get the current normalized controller state (thread-safe).

        Returns:
            A copy of the current ControllerState.
        """
        with self._state_lock:
            return ControllerState(
                LT=self._state.LT,
                LB=self._state.LB,
                LS=self._state.LS,
                RS=self._state.RS,
                RT=self._state.RT,
                RB=self._state.RB,
                LX=self._state.LX,
                LY=self._state.LY,
                RX=self._state.RX,
                RY=self._state.RY,
            )

    def register_callback(
        self, callback: Callable[[ControllerState], None]
    ) -> None:
        """Register a callback to be called when controller state is updated.

        Args:
            callback: Function that takes ControllerState as argument.
        """
        with self._callback_lock:
            self._callbacks.append(callback)

    def unregister_callback(
        self, callback: Callable[[ControllerState], None]
    ) -> None:
        """Unregister a callback.

        Args:
            callback: Callback function to remove.
        """
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _close_controller(self) -> None:
        """Quit and drop the open SDL2 controller handle."""
        if self._controller is not None:
            try:
                self._controller.quit()
            except Exception:
                pass
            self._controller = None
        self._controller_instance_id = None

    def _clear_state(self) -> None:
        """Reset normalized state to zeros (safe after disconnect)."""
        with self._state_lock:
            self._state = ControllerState()

    def _mark_disconnected(self, reason: str) -> None:
        """Close controller, zero state, and log disconnect once per event."""
        if self._controller is None:
            return
        logger.warning(
            "Gamepad disconnected (%s); waiting to reopen (Home-wake / replug)",
            reason,
        )
        self._close_controller()
        self._clear_state()

    def _read_instance_id(self, controller: sdl2_controller.Controller) -> Optional[int]:
        """Return SDL instance id for an open controller, or None."""
        try:
            return controller.as_joystick().get_instance_id()
        except Exception:
            return None

    def _refresh_joystick_subsystem(self) -> None:
        """Rescan joysticks so newly appeared BT/USB nodes are visible to SDL."""
        try:
            if pygame.joystick.get_init():
                pygame.joystick.quit()
            pygame.joystick.init()
            if not sdl2_controller.get_init():
                sdl2_controller.init()
        except Exception as e:
            logger.debug("Joystick subsystem refresh failed: %s", e)

    def _try_open_controller(self, device_index: Optional[int] = None) -> bool:
        """Open a controller-capable device. Returns True on success.

        Prefers ``device_index`` when given and valid; otherwise the configured
        ``_device_id`` if still a controller; otherwise the first controller-capable
        index (hotplug often renumbers devices).
        """
        try:
            count = sdl2_controller.get_count()
        except Exception as e:
            logger.debug("get_count failed while reopening: %s", e)
            return False

        if count <= 0:
            return False

        candidates: list[int] = []
        if device_index is not None:
            candidates.append(device_index)
        if self._device_id is not None:
            candidates.append(self._device_id)
        candidates.extend(range(count))

        seen: set[int] = set()
        for idx in candidates:
            if idx in seen or idx < 0 or idx >= count:
                continue
            seen.add(idx)
            try:
                if not sdl2_controller.is_controller(idx):
                    continue
                controller = sdl2_controller.Controller(idx)
            except Exception as e:
                logger.debug("Failed to open controller index %s: %s", idx, e)
                continue

            self._controller = controller
            self._device_id = idx
            self._controller_instance_id = self._read_instance_id(controller)
            name = sdl2_controller.name_forindex(idx) or "Unknown"
            logger.info("Using SDL2 controller: %s (device_id=%s)", name, idx)
            return True

        return False

    def _process_hotplug_events(self) -> None:
        """Handle CONTROLLERDEVICEADDED / REMOVED from the pygame event queue."""
        try:
            events = pygame.event.get()
        except Exception:
            return

        for event in events:
            etype = getattr(event, "type", None)
            if etype == pygame.CONTROLLERDEVICEREMOVED:
                removed_id = getattr(event, "instance_id", None)
                if (
                    self._controller is not None
                    and removed_id is not None
                    and self._controller_instance_id is not None
                    and removed_id == self._controller_instance_id
                ):
                    self._mark_disconnected("CONTROLLERDEVICEREMOVED")
                elif self._controller is not None and removed_id is not None:
                    # Unknown instance — still check attached() next tick.
                    logger.debug(
                        "CONTROLLERDEVICEREMOVED instance_id=%s (ours=%s)",
                        removed_id,
                        self._controller_instance_id,
                    )
            elif etype == pygame.CONTROLLERDEVICEADDED:
                if self._controller is None:
                    device_index = getattr(event, "device_index", None)
                    if device_index is not None and self._try_open_controller(device_index):
                        logger.info("Gamepad reconnected (CONTROLLERDEVICEADDED)")

    def _controller_still_attached(self) -> bool:
        """Return True if the open controller reports attached."""
        if self._controller is None:
            return False
        try:
            return bool(self._controller.attached())
        except Exception:
            return False

    def _event_loop(self) -> None:
        """Main event loop running in background thread.

        Uses pygame SDL2 Game Controller API. Initializes pygame and controller
        subsystem, opens the selected controller-capable device, and polls state
        at fixed rate. On disconnect (idle BT sleep, unplug), zeros state and
        reopens when the pad returns without requiring a process restart.
        """
        sleep_time = 1.0 / self._update_rate_hz

        pygame_was_initialized = pygame.get_init()
        joystick_was_initialized = pygame.joystick.get_init()

        try:
            if not pygame_was_initialized:
                pygame.init()
            if not joystick_was_initialized:
                pygame.joystick.init()
            if not sdl2_controller.get_init():
                sdl2_controller.init()
            # Ensure controller hotplug events are delivered (pygame issue #4620).
            try:
                sdl2_controller.set_eventstate(True)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}", exc_info=True)
            self._running = False
            return

        device_count = sdl2_controller.get_count()
        logger.debug(f"Found {device_count} joystick(s)")

        if device_count == 0:
            logger.error(
                "No controller-capable device found. Make sure your controller is connected."
            )
            logger.error("Try running with --list to verify the controller is detected.")
            self._running = False
            if not pygame_was_initialized:
                pygame.quit()
            return

        if not self._try_open_controller(self._device_id):
            logger.error(
                "No supported game controller at device_id=%s. "
                "Use --list to see controller-capable devices.",
                self._device_id,
            )
            self._running = False
            if not pygame_was_initialized:
                pygame.quit()
            return

        last_scan_time = 0.0
        last_wait_log_time = 0.0

        try:
            while self._running:
                start_time = time.time()

                self._process_hotplug_events()

                if self._controller is not None and not self._controller_still_attached():
                    self._mark_disconnected("attached()=False")

                if self._controller is not None:
                    self._update_state_controller(self._controller)
                else:
                    now = time.time()
                    if now - last_scan_time >= _RECONNECT_SCAN_INTERVAL_S:
                        last_scan_time = now
                        self._refresh_joystick_subsystem()
                        if self._try_open_controller():
                            logger.info("Gamepad reconnected after rescan")
                        elif now - last_wait_log_time >= _RECONNECT_LOG_INTERVAL_S:
                            last_wait_log_time = now
                            logger.info(
                                "Waiting for gamepad reconnect "
                                "(press Home after idle sleep, or replug USB)…"
                            )
                    # Keep publishing zeros so mappers do not hold last stick values.
                    self._clear_state()

                state = self.get_state()
                self._notify_callbacks(state)

                elapsed = time.time() - start_time
                sleep_duration = max(0, sleep_time - elapsed)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

        except Exception as e:
            logger.error(f"InputController event loop error: {e}", exc_info=True)
            raise
        finally:
            self._close_controller()
            if not pygame_was_initialized:
                pygame.quit()
            self._running = False

    def _update_state_controller(
        self, controller: sdl2_controller.Controller
    ) -> None:
        """Update controller state from SDL2 Game Controller.

        Uses logical axis/button constants; values are normalized to [-1, 1]
        for sticks and thresholded for triggers.
        """
        with self._state_lock:
            self._state.LT = False
            self._state.LB = False
            self._state.LS = False
            self._state.RT = False
            self._state.RB = False
            self._state.RS = False

            # Buttons (logical names)
            self._state.LB = bool(controller.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER))
            self._state.RB = bool(controller.get_button(pygame.CONTROLLER_BUTTON_RIGHTSHOULDER))
            self._state.LS = bool(controller.get_button(pygame.CONTROLLER_BUTTON_LEFTSTICK))
            self._state.RS = bool(controller.get_button(pygame.CONTROLLER_BUTTON_RIGHTSTICK))

            # Triggers (axes 0..32768, normalize then threshold)
            trigger_left = controller.get_axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT)
            trigger_right = controller.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)
            self._state.LT = (trigger_left / _AXIS_SCALE) > 0.1
            self._state.RT = (trigger_right / _AXIS_SCALE) > 0.1

            # Sticks (axes -32768..32767, normalize to [-1, 1])
            self._state.LX = max(-1.0, min(1.0, controller.get_axis(pygame.CONTROLLER_AXIS_LEFTX) / _AXIS_SCALE))
            self._state.LY = max(-1.0, min(1.0, controller.get_axis(pygame.CONTROLLER_AXIS_LEFTY) / _AXIS_SCALE))
            self._state.RX = max(-1.0, min(1.0, controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTX) / _AXIS_SCALE))
            self._state.RY = max(-1.0, min(1.0, controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTY) / _AXIS_SCALE))

            logger.debug(
                f"Left stick X: {self._state.LX}, Left stick Y: {self._state.LY}, "
                f"Right stick X: {self._state.RX}, Right stick Y: {self._state.RY}"
            )
            logger.debug(
                f"Left trigger: {self._state.LT}, Right trigger: {self._state.RT}"
            )
            logger.debug(
                f"Left bumper: {self._state.LB}, Right bumper: {self._state.RB}"
            )
            logger.debug(
                f"Left stick press: {self._state.LS}, Right stick press: {self._state.RS}"
            )

    def _notify_callbacks(self, state: ControllerState) -> None:
        """Notify all registered callbacks with new controller state."""
        with self._callback_lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error in InputController callback: {e}", exc_info=True)

    @staticmethod
    def list_devices() -> list[dict]:
        """List controller-capable gamepad devices (SDL2 Game Controller API).

        Returns:
            List of device dicts with 'name' and 'path' for devices that
            is_controller(i) is True.
        """
        try:
            pygame.init()
            pygame.joystick.init()
            if not sdl2_controller.get_init():
                sdl2_controller.init()
            gamepads = []
            for i in range(sdl2_controller.get_count()):
                if sdl2_controller.is_controller(i):
                    name = sdl2_controller.name_forindex(i)
                    gamepads.append({
                        "name": name if name is not None else f"Controller {i}",
                        "path": f"pygame_controller_{i}",
                        "device_id": i,  # joystick index for Controller(index)
                    })
            pygame.quit()
            return gamepads
        except Exception as e:
            logger.error(f"Error listing devices: {e}", exc_info=True)
            return []
