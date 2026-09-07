"""Actuator connection-state telemetry tests."""
import math

from firmware.interfaces.joint_telemetry import (
    ActuatorConnection,
    JointTelemetry,
)
from firmware.interfaces.telemetry_frame import TelemetryFrame

CONNECTED_SEG = "FLHY 0.123 512 12 1 0 0 128 3"
DISCONNECTED_SEG = "FLHY nan 517 0 0 0 0 0 3"


class TestJointConnectedFlag:
    def test_finite_pos_segment_is_connected(self):
        jt = JointTelemetry.from_tokens(CONNECTED_SEG.split())

        assert jt is not None
        assert jt.connected is True
        assert jt.pos == 0.123

    def test_nan_pos_segment_is_disconnected(self):
        jt = JointTelemetry.from_tokens(DISCONNECTED_SEG.split())

        assert jt is not None
        assert jt.connected is False
        assert math.isnan(jt.pos)

    def test_inf_pos_segment_is_disconnected(self):
        inf_seg = "FLHY inf 517 0 0 0 0 0 3"
        jt = JointTelemetry.from_tokens(inf_seg.split())

        assert jt is not None
        assert jt.connected is False
        assert not math.isfinite(jt.pos)
        assert jt.format_compact() == "FLHY:DISC,517,0"

    def test_direct_non_finite_measurement_is_disconnected(self):
        jt = JointTelemetry(
            name="FLHY",
            pos=math.nan,
            pot=517,
            current=0,
            en=(0, 0),
            pwm=(0, 0),
            saf=3,
        )

        assert jt.connected is False

    def test_disconnected_segment_keeps_its_other_fields(self):
        jt = JointTelemetry.from_tokens(DISCONNECTED_SEG.split())

        assert jt.name == "FLHY"
        assert jt.pot == 517
        assert jt.current == 0
        assert jt.saf == 3


class TestDisconnectedJointInLine:
    def test_disconnected_joint_is_kept_not_dropped(self):
        parsed = TelemetryFrame.parse_line(f"FRONT; {DISCONNECTED_SEG}")

        assert [j.name for j in parsed.joints] == ["FLHY"]
        assert parsed.joints[0].connected is False

    def test_line_mixes_connected_and_disconnected_joints(self):
        second = CONNECTED_SEG.replace("FLHY", "FLHL")
        parsed = TelemetryFrame.parse_line(f"FRONT; {DISCONNECTED_SEG};{second}")

        by_name = {j.name: j for j in parsed.joints}
        assert by_name["FLHY"].connected is False
        assert by_name["FLHL"].connected is True


class TestFormatCompact:
    def test_connected_joint_shows_position(self):
        jt = JointTelemetry.from_tokens(CONNECTED_SEG.split())

        assert jt.format_compact() == "FLHY:0.123,512,12,(1,0),(0,128),3"

    def test_disconnected_joint_shows_disc_marker_not_a_position(self):
        jt = JointTelemetry.from_tokens(DISCONNECTED_SEG.split())

        assert jt.format_compact() == "FLHY:DISC,517,0"


# Rail-short with stall current.
RAIL_SHORTED_SEG = "FLHY nan 1023 45 1 1 0 200 3"
MOVING_SEG = "FLHY 0.742 760 30 1 1 0 200 5"
JOG_TICK_A = "FLHY 0.300 300 210 1 1 0 255 7"
JOG_TICK_B = "FLHY 0.550 555 205 1 1 0 255 9"
IDLE_ATTACHED_SEG = "FLHY 0.500 512 0 0 0 0 0 3"


class TestRailShortedPot:
    def test_rail_shorted_pot_is_disconnected(self):
        jt = JointTelemetry.from_tokens(RAIL_SHORTED_SEG.split())

        assert jt is not None
        assert jt.connected is False
        assert math.isnan(jt.pos)

    def test_rail_shorted_pot_shows_disc_despite_current(self):
        jt = JointTelemetry.from_tokens(RAIL_SHORTED_SEG.split())

        assert jt.format_compact() == "FLHY:DISC,1023,45"


class TestMovingJointNotFalselyDisconnected:
    def test_moving_joint_stays_connected(self):
        jt = JointTelemetry.from_tokens(MOVING_SEG.split())

        assert jt is not None
        assert jt.connected is True
        assert jt.pos == 0.742

    def test_fast_jog_stays_connected_across_ticks(self):
        a = JointTelemetry.from_tokens(JOG_TICK_A.split())
        b = JointTelemetry.from_tokens(JOG_TICK_B.split())

        assert a.connected is True and b.connected is True
        assert abs(b.pot - a.pot) > 200

    def test_idle_attached_joint_stays_connected(self):
        jt = JointTelemetry.from_tokens(IDLE_ATTACHED_SEG.split())

        assert jt is not None
        assert jt.connected is True
        assert jt.pos == 0.500


class TestConnectionStateField:
    NINE = "FLHY 0.300 300 210 1 1 0 255 7"

    def test_nine_field_segment_still_parses(self):
        jt = JointTelemetry.from_tokens(self.NINE.split())
        assert jt is not None
        assert jt.pot == 300 and jt.current == 210

    def test_absent_field_reads_as_unknown(self):
        jt = JointTelemetry.from_tokens(self.NINE.split())
        assert jt.connection_state is ActuatorConnection.UNKNOWN
        assert jt.connected is True

    def test_all_connection_states_are_decoded(self):
        for token, expected in (
            ("0", ActuatorConnection.UNKNOWN),
            ("1", ActuatorConnection.CONNECTED),
            ("2", ActuatorConnection.DISCONNECTED),
        ):
            jt = JointTelemetry.from_tokens((self.NINE + f" {token}").split())
            assert jt.connection_state is expected

    def test_unknown_connection_state_token_is_rejected(self):
        assert JointTelemetry.from_tokens((self.NINE + " 3").split()) is None

    def test_eleven_fields_is_rejected(self):
        assert JointTelemetry.from_tokens((self.NINE + " 1 2").split()) is None

    def test_non_finite_legacy_position_remains_disconnected(self):
        jt = JointTelemetry.from_tokens("FLHY nan 300 210 1 1 0 255 7 1".split())
        assert jt.connected is False
        assert jt.connection_state is ActuatorConnection.DISCONNECTED

    def test_disconnected_state_overrides_finite_position(self):
        jt = JointTelemetry.from_tokens((self.NINE + " 2").split())
        assert jt.connected is False
        assert jt.connection_state is ActuatorConnection.DISCONNECTED
