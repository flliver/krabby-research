"""Drive-only loop ordering: firmware step, pin readback, Isaac joint targets, env.step."""

from types import SimpleNamespace

import pytest

from firmware.sim.robot_core import FIRMWARE_STEP_MS, RobotCore
from firmware.sim.simulated_actuator import RecordedArticulation
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


class FakeHostBridge:
    """Stands in for libkrabby_host: records calls, drives pins like manualDrive."""

    def __init__(self) -> None:
        self.now_ms = 0
        self.update_all_calls = 0
        self.calls: list[str] = []
        self._pwm: dict[int, int] = {}
        self._digital: dict[int, int] = {}

    def reset(self) -> None:
        self.now_ms = 0
        self._pwm.clear()
        self._digital.clear()
        self.calls.append("reset")

    def handle_jog(self, actuator_name: str, pwm: int) -> None:
        slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id(actuator_name)
        pwm_r, pwm_l, pin_en = slot.drive_pins
        self._pwm[pwm_r] = max(pwm, 0)
        self._pwm[pwm_l] = max(-pwm, 0)
        self._digital[pin_en] = 1 if pwm else 0
        self.calls.append(f"handle_jog:{actuator_name}:{pwm}")

    def update_all(self) -> None:
        self.update_all_calls += 1
        self.calls.append("update_all")

    def pin_pwm(self, pin: int) -> int:
        self.calls.append(f"pin_pwm:{pin}")
        return self._pwm.get(pin, 0)

    def pin_digital(self, pin: int) -> int:
        self.calls.append(f"pin_digital:{pin}")
        return self._digital.get(pin, 0)

    def advance_millis(self, delta_ms: int) -> None:
        self.now_ms += delta_ms
        self.calls.append(f"advance_millis:{delta_ms}")

    def millis(self) -> int:
        return self.now_ms


def _fake_env(physics_dt_s: float = FIRMWARE_STEP_MS / 1000.0):
    """Minimal Isaac env stub so ``physics_step_ms`` matches production."""
    sim = SimpleNamespace(get_physics_dt=lambda: physics_dt_s)
    return SimpleNamespace(unwrapped=SimpleNamespace(sim=sim))


@pytest.fixture
def core():
    config = SimulatedRobotConfig.front_board_rev3()
    joint_index_for_name = {name: i for i, name in enumerate(config.isaac_joint_names())}
    steps: list[object] = []
    articulation = RecordedArticulation()
    loop = RobotCore.front_board(
        env=_fake_env(),
        articulation=articulation,
        joint_index_for_name=joint_index_for_name,
        bridge=FakeHostBridge(),
        config=config,
        max_joint_speed=2.0,
        advance=lambda env: steps.append(env) or "stepped",
    )
    return loop, articulation, steps


def test_front_board_wires_six_actuators(core):
    loop, _, _ = core
    assert len(loop.actuators) == 6
    assert [a.joint_index for a in loop.actuators] == list(range(6))


def test_step_advances_millis_and_advances_sim(core):
    loop, _, steps = core

    assert loop.millis() == 0
    assert loop.step() == "stepped"

    assert loop.millis() == FIRMWARE_STEP_MS
    assert loop.bridge.update_all_calls == 1
    assert len(steps) == 1


def test_jog_drives_only_the_jogged_joint(core):
    loop, articulation, _ = core
    flhy = loop.actuators[0]

    loop.jog("FLHY", 255)
    loop.step()

    assert articulation.velocity_for(flhy.joint_index) == pytest.approx(2.0)
    for actuator in loop.actuators[1:]:
        assert articulation.velocity_for(actuator.joint_index) == pytest.approx(0.0)


def test_jog_direction_follows_signed_pwm(core):
    loop, articulation, _ = core
    frkl = loop.actuators[-1]

    loop.jog("FRKL", -255)
    loop.step()

    assert articulation.velocity_for(frkl.joint_index) == pytest.approx(-2.0)


def test_firmware_step_runs_before_pins_are_read(core):
    loop, _, _ = core

    loop.step()

    first_pin_read = next(i for i, c in enumerate(loop.bridge.calls) if c.startswith("pin_"))
    assert loop.bridge.calls.index("advance_millis:10") < loop.bridge.calls.index("update_all")
    assert loop.bridge.calls.index("update_all") < first_pin_read


def test_reset_clears_clock_and_pin_state(core):
    loop, _, _ = core

    loop.jog("FLHL", 200)
    loop.step()
    loop.reset()

    assert loop.millis() == 0
    assert loop.pin_state.pwm_duty(loop.actuators[1].slot.pin_pwm_r) == 0
