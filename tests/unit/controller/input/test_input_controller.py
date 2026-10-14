"""Unit tests for InputController."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from controller.input.input_controller import InputController
from controller.input.state import ControllerState


def _reset_singleton():
    if InputController._instance is not None:
        try:
            InputController._instance.stop()
        except Exception:
            pass
    InputController._instance = None


class TestInputControllerSingleton:
    """Test singleton pattern."""

    def setup_method(self):
        _reset_singleton()

    def teardown_method(self):
        _reset_singleton()

    def test_singleton_returns_same_instance(self):
        """Test that get_instance returns the same instance."""
        controller1 = InputController.get_instance()
        controller2 = InputController.get_instance()
        assert controller1 is controller2


class TestInputControllerState:
    """Test state management."""

    def setup_method(self):
        _reset_singleton()

    def teardown_method(self):
        _reset_singleton()

    def test_default_state(self):
        """Test default controller state."""
        controller = InputController.get_instance()
        state = controller.get_state()
        assert state.LT is False
        assert state.LB is False
        assert state.LS is False
        assert state.RT is False
        assert state.RB is False
        assert state.RS is False
        assert state.LX == 0.0
        assert state.LY == 0.0
        assert state.RX == 0.0
        assert state.RY == 0.0


class TestInputControllerThreadSafety:
    """Test thread safety of state access."""

    def setup_method(self):
        _reset_singleton()

    def teardown_method(self):
        _reset_singleton()

    def test_get_state_returns_copy(self):
        """Test that get_state returns a copy, not the internal state."""
        controller = InputController.get_instance()

        state1 = controller.get_state()
        state2 = controller.get_state()

        # Should be different objects
        assert state1 is not state2

        # But should have same values
        assert state1.LT == state2.LT
        assert state1.LY == state2.LY


class TestInputControllerReconnect:
    """Disconnect / reopen helpers."""

    def setup_method(self):
        _reset_singleton()
        self.ic = InputController.get_instance()
        self.ic._device_id = 0

    def teardown_method(self):
        _reset_singleton()

    def test_mark_disconnected_closes_and_zeros_state(self):
        mock_ctl = MagicMock()
        self.ic._controller = mock_ctl
        self.ic._controller_instance_id = 42
        with self.ic._state_lock:
            self.ic._state = ControllerState(LX=0.5, LY=-0.25, LB=True)

        self.ic._mark_disconnected("attached()=False")

        mock_ctl.quit.assert_called_once()
        assert self.ic._controller is None
        assert self.ic._controller_instance_id is None
        state = self.ic.get_state()
        assert state.LX == 0.0
        assert state.LY == 0.0
        assert state.LB is False

    def test_mark_disconnected_noop_when_already_closed(self):
        self.ic._controller = None
        self.ic._mark_disconnected("noop")
        assert self.ic._controller is None

    @patch("controller.input.input_controller.sdl2_controller")
    def test_try_open_controller_prefers_device_index(self, mock_sdl):
        mock_ctl = MagicMock()
        mock_joy = MagicMock()
        mock_joy.get_instance_id.return_value = 7
        mock_ctl.as_joystick.return_value = mock_joy

        mock_sdl.get_count.return_value = 2
        mock_sdl.is_controller.side_effect = lambda i: i == 1
        mock_sdl.Controller.return_value = mock_ctl
        mock_sdl.name_forindex.return_value = "Pad"

        assert self.ic._try_open_controller(device_index=1) is True
        mock_sdl.Controller.assert_called_with(1)
        assert self.ic._controller is mock_ctl
        assert self.ic._device_id == 1
        assert self.ic._controller_instance_id == 7

    @patch("controller.input.input_controller.sdl2_controller")
    def test_try_open_controller_falls_back_to_first_available(self, mock_sdl):
        """After hotplug renumber, preferred index may be gone — open first capable."""
        mock_ctl = MagicMock()
        mock_joy = MagicMock()
        mock_joy.get_instance_id.return_value = 99
        mock_ctl.as_joystick.return_value = mock_joy

        self.ic._device_id = 5  # stale preferred index
        mock_sdl.get_count.return_value = 1
        mock_sdl.is_controller.side_effect = lambda i: i == 0
        mock_sdl.Controller.return_value = mock_ctl
        mock_sdl.name_forindex.return_value = "Clone Pad"

        assert self.ic._try_open_controller() is True
        mock_sdl.Controller.assert_called_with(0)
        assert self.ic._device_id == 0
        assert self.ic._controller_instance_id == 99

    @patch("controller.input.input_controller.sdl2_controller")
    def test_try_open_controller_returns_false_when_none(self, mock_sdl):
        mock_sdl.get_count.return_value = 0
        assert self.ic._try_open_controller() is False
        assert self.ic._controller is None

    @patch("controller.input.input_controller.pygame")
    def test_process_hotplug_removed_closes_matching_instance(self, mock_pygame):
        mock_ctl = MagicMock()
        self.ic._controller = mock_ctl
        self.ic._controller_instance_id = 11
        with self.ic._state_lock:
            self.ic._state = ControllerState(RX=1.0)

        mock_pygame.CONTROLLERDEVICEREMOVED = 1
        mock_pygame.CONTROLLERDEVICEADDED = 2
        mock_pygame.event.get.return_value = [
            SimpleNamespace(type=1, instance_id=11),
        ]

        self.ic._process_hotplug_events()

        mock_ctl.quit.assert_called_once()
        assert self.ic._controller is None
        assert self.ic.get_state().RX == 0.0

    @patch("controller.input.input_controller.sdl2_controller")
    @patch("controller.input.input_controller.pygame")
    def test_process_hotplug_added_reopens_when_closed(self, mock_pygame, mock_sdl):
        self.ic._controller = None
        mock_ctl = MagicMock()
        mock_joy = MagicMock()
        mock_joy.get_instance_id.return_value = 3
        mock_ctl.as_joystick.return_value = mock_joy

        mock_sdl.get_count.return_value = 1
        mock_sdl.is_controller.return_value = True
        mock_sdl.Controller.return_value = mock_ctl
        mock_sdl.name_forindex.return_value = "Reconnected"

        mock_pygame.CONTROLLERDEVICEREMOVED = 1
        mock_pygame.CONTROLLERDEVICEADDED = 2
        mock_pygame.event.get.return_value = [
            SimpleNamespace(type=2, device_index=0),
        ]

        self.ic._process_hotplug_events()

        assert self.ic._controller is mock_ctl
        assert self.ic._controller_instance_id == 3

    def test_controller_still_attached_false_when_detached(self):
        mock_ctl = MagicMock()
        mock_ctl.attached.return_value = False
        self.ic._controller = mock_ctl
        assert self.ic._controller_still_attached() is False

    def test_controller_still_attached_true_when_connected(self):
        mock_ctl = MagicMock()
        mock_ctl.attached.return_value = True
        self.ic._controller = mock_ctl
        assert self.ic._controller_still_attached() is True
