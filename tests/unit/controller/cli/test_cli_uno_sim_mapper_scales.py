"""krabby-uno-sim mapper scale defaults (Isaac rad vs STM duty)."""

from controller.cli.cli_uno_sim import resolve_mapper_scales


def test_non_stm_uses_isaac_rad_defaults():
    hip, knee, yaw = resolve_mapper_scales(
        stm=False,
        mapper_hip_scale=None,
        mapper_knee_scale=None,
        mapper_hip_yaw_scale=None,
    )
    assert (hip, knee, yaw) == (0.3, 0.3, 0.2)


def test_stm_uses_unit_duty_defaults():
    hip, knee, yaw = resolve_mapper_scales(
        stm=True,
        mapper_hip_scale=None,
        mapper_knee_scale=None,
        mapper_hip_yaw_scale=None,
    )
    assert (hip, knee, yaw) == (1.0, 1.0, 1.0)


def test_stm_overrides_still_apply():
    hip, knee, yaw = resolve_mapper_scales(
        stm=True,
        mapper_hip_scale=0.5,
        mapper_knee_scale=None,
        mapper_hip_yaw_scale=0.2,
    )
    assert hip == 0.5
    assert knee == 1.0
    assert yaw == 0.2