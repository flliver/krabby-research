"""GET /devices and GET /devices/{thingName}: mocked IoT clients, no real AWS."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from krabby_fleet_service._auth import require_operator
from krabby_fleet_service._config import Settings, get_settings
from krabby_fleet_service.app import app

_SETTINGS = Settings(
    aws_region="us-east-1",
    cognito_user_pool_id="us-east-1_TESTPOOL",
    cognito_app_client_id="test-client-id",
    iot_ats_endpoint="example-ats.iot.us-east-1.amazonaws.com",
)
_FAKE_CLAIMS = {"sub": "test-operator", "cognito:groups": ["operator"]}


@pytest.fixture
def authed_client():
    app.dependency_overrides[get_settings] = lambda: _SETTINGS
    app.dependency_overrides[require_operator] = lambda: _FAKE_CLAIMS
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_list_devices_uses_search_index(authed_client):
    shadow = json.dumps({"reported": {"timestamp": 1710000000, "reported_image": "img:tag"}})
    fake_iot = MagicMock()
    fake_iot.search_index.return_value = {
        "things": [
            {
                "thingName": "bench-krabby-ci",
                "connectivity": {"connected": True, "timestamp": 1710000123000},
                "shadow": shadow,
            }
        ]
    }

    with patch("krabby_fleet_service._devices._iot_client", return_value=fake_iot):
        resp = authed_client.get("/devices")

    assert resp.status_code == 200
    body = resp.json()
    assert body == [
        {
            "thingName": "bench-krabby-ci",
            "connected": True,
            "connectivityTimestamp": 1710000123000,
            "reported": {"timestamp": 1710000000, "reported_image": "img:tag"},
        }
    ]
    fake_iot.search_index.assert_called_once_with(queryString="thingTypeName:Krab")


def test_get_device_uses_describe_shadow_and_search(authed_client):
    fake_iot = MagicMock()
    fake_iot.describe_thing.return_value = {
        "thingName": "bench-krabby-ci",
        "thingTypeName": "Krab",
        "attributes": {"site": "bench"},
    }
    fake_iot.search_index.return_value = {
        "things": [
            {
                "thingName": "bench-krabby-ci",
                "connectivity": {"connected": False, "timestamp": 1710000456000},
            }
        ]
    }

    fake_iot_data = MagicMock()
    fake_iot_data.get_thing_shadow.return_value = {
        "payload": MagicMock(
            read=lambda: json.dumps(
                {
                    "state": {
                        "reported": {
                            "timestamp": 1710000000,
                            "health": {"krabby_agent": "active"},
                        }
                    }
                }
            ).encode()
        )
    }

    with patch("krabby_fleet_service._devices._iot_client", return_value=fake_iot), patch(
        "krabby_fleet_service._devices._iot_data_client", return_value=fake_iot_data
    ):
        resp = authed_client.get("/devices/bench-krabby-ci")

    assert resp.status_code == 200
    assert resp.json() == {
        "thingName": "bench-krabby-ci",
        "thingTypeName": "Krab",
        "attributes": {"site": "bench"},
        "connected": False,
        "connectivityTimestamp": 1710000456000,
        "reported": {"timestamp": 1710000000, "health": {"krabby_agent": "active"}},
    }
    fake_iot.describe_thing.assert_called_once_with(thingName="bench-krabby-ci")
    fake_iot_data.get_thing_shadow.assert_called_once_with(thingName="bench-krabby-ci")


def test_get_device_missing_thing_is_404(authed_client):
    fake_iot = MagicMock()

    class _NotFound(Exception):
        pass

    fake_iot.exceptions.ResourceNotFoundException = _NotFound
    fake_iot.describe_thing.side_effect = _NotFound()

    with patch("krabby_fleet_service._devices._iot_client", return_value=fake_iot):
        resp = authed_client.get("/devices/missing-krab")

    assert resp.status_code == 404


def test_get_device_wrong_thing_type_is_404(authed_client):
    fake_iot = MagicMock()
    fake_iot.describe_thing.return_value = {
        "thingName": "other-thing",
        "thingTypeName": "NotKrab",
        "attributes": {},
    }

    with patch("krabby_fleet_service._devices._iot_client", return_value=fake_iot):
        resp = authed_client.get("/devices/other-thing")

    assert resp.status_code == 404
