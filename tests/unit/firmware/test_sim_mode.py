"""Host sim mode environment flags."""

import os

from firmware.sim.sim_mode import single_leader_eighteen_enabled


def test_single_leader_flag(monkeypatch):
    monkeypatch.delenv("KRABBY_SIM_SINGLE_LEADER_18", raising=False)
    assert single_leader_eighteen_enabled() is False
    monkeypatch.setenv("KRABBY_SIM_SINGLE_LEADER_18", "1")
    assert single_leader_eighteen_enabled() is True
