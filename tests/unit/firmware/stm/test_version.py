from firmware.stm import version as leg_version


def test_leg_firmware_version_matches_version_file():
    assert leg_version.__version__ == "0.1.0-dev"
