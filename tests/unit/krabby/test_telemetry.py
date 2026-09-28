"""Unit tests for fleet telemetry collection."""
from __future__ import annotations

import json
from typing import Any

import pytest

from krabby import telemetry as tel


def _patch_collect(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    monkeypatch.setattr(tel, "collect_health", lambda: overrides.get("health", {}))
    monkeypatch.setattr(tel, "_probe_hal_imu_pose", lambda: overrides.get("hal"))
    monkeypatch.setattr(tel, "collect_power", lambda: overrides.get("power"))
    monkeypatch.setattr(tel, "installed_image", lambda: overrides.get("reported_image"))
    monkeypatch.setattr(tel.time, "time", lambda: overrides.get("now", 1710000000.5))


class TestCollectRedFlags:
    def test_agent_down(self):
        health = {"krabby_agent": "inactive", "krabby_locomotion": "active", "locomotion_container": "running"}
        assert "agent_not_running" in tel.collect_red_flags(health, hal={"imu": {}})

    def test_locomotion_container_down(self):
        health = {
            "krabby_agent": "active",
            "krabby_locomotion": "active",
            "locomotion_container": "stopped",
            "mcu_present": True,
        }
        assert "locomotion_container_down" in tel.collect_red_flags(health, hal=None)

    def test_hal_missing_when_container_running(self):
        health = {"locomotion_container": "running", "mcu_present": True}
        assert "hal_no_observation" in tel.collect_red_flags(health, hal=None)

    def test_mcu_missing(self):
        health = {"mcu_present": False}
        assert "mcu_missing" in tel.collect_red_flags(health, hal=None)


class TestCollectTelemetry:
    def test_minimal_snapshot(self, monkeypatch: pytest.MonkeyPatch):
        _patch_collect(
            monkeypatch,
            health={"locomotion_container": "absent", "krabby_agent": "active", "mcu_present": False},
            hal=None,
            power=None,
            reported_image=None,
        )
        payload = tel.collect_telemetry()
        assert payload["timestamp"] == 1710000000
        assert payload["reported_image"] is None
        assert payload["health"]["locomotion_container"] == "absent"
        assert "red_flags" in payload
        assert "mcu_missing" in payload["red_flags"]
        assert "imu" not in payload
        assert "power" not in payload

    def test_includes_hal_and_power(self, monkeypatch: pytest.MonkeyPatch):
        hal = {
            "imu": {"base_quat_w": [0, 0, 0, 1], "base_ang_vel_b": [0, 0, 0], "base_lin_vel_b": [0, 0, 0]},
            "pose": {"joint_positions": [0.1], "joint_velocities": [0.0]},
        }
        _patch_collect(
            monkeypatch,
            health={"locomotion_container": "running", "krabby_agent": "active", "mcu_present": True},
            hal=hal,
            power={"supplies": [{"name": "BAT0", "capacity_percent": 90}]},
            reported_image="public.ecr.aws/t7t7b3i3/krabby-locomotion:release-latest",
        )
        payload = tel.collect_telemetry()
        assert payload["imu"] == hal["imu"]
        assert payload["pose"] == hal["pose"]
        assert payload["power"]["supplies"][0]["capacity_percent"] == 90
        assert payload["reported_image"].endswith("release-latest")
        assert "red_flags" not in payload


class TestCmdGetTelemetry:
    def test_prints_json(self, monkeypatch: pytest.MonkeyPatch, capsys):
        _patch_collect(monkeypatch, health={"mcu_present": True}, hal=None, power=None)
        tel.cmd_get_telemetry()
        payload = json.loads(capsys.readouterr().out)
        assert payload["timestamp"] == 1710000000


class TestAgentShadowPayload:
    def test_publish_uses_collect_telemetry(self, monkeypatch: pytest.MonkeyPatch):
        import sys
        import types

        captured: dict[str, Any] = {}

        class _FakeState:
            def __init__(self, reported):
                captured["reported"] = reported

        class _FakeRequest:
            def __init__(self, thing_name, state):
                captured["thing_name"] = thing_name

        fake_iotshadow = types.SimpleNamespace(
            ShadowState=_FakeState,
            UpdateShadowRequest=_FakeRequest,
        )
        fake_mqtt = types.SimpleNamespace(QoS=types.SimpleNamespace(AT_LEAST_ONCE=1))
        monkeypatch.setitem(sys.modules, "awsiot", types.SimpleNamespace(iotshadow=fake_iotshadow))
        monkeypatch.setitem(sys.modules, "awscrt", types.SimpleNamespace(mqtt=fake_mqtt))

        _patch_collect(
            monkeypatch,
            health={"mcu_present": True},
            hal=None,
            power=None,
            reported_image="img:tag",
        )

        class _FakeShadowClient:
            def publish_update_shadow(self, request, qos):
                captured["qos"] = qos

        from krabby.agent import _publish_shadow_report

        fake_shim = types.SimpleNamespace(robot_edge_connected=lambda: True)
        _publish_shadow_report(_FakeShadowClient(), "bench-krabby-ci", fake_shim)
        assert captured["thing_name"] == "bench-krabby-ci"
        assert captured["reported"]["reported_image"] == "img:tag"
        assert captured["reported"]["timestamp"] == 1710000000
        assert captured["reported"]["teleop_edge_connected"] is True


class TestGetTelemetryCli:
    def test_argv_dispatches(self, monkeypatch: pytest.MonkeyPatch):
        called = {"n": 0}

        def fake_cmd():
            called["n"] += 1

        monkeypatch.setattr("krabby.telemetry.cmd_get_telemetry", fake_cmd)
        monkeypatch.setattr("sys.argv", ["krabby", "get", "telemetry"])
        from krabby.__main__ import main

        main()
        assert called["n"] == 1
