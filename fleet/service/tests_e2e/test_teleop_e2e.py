"""Teleop E2E: Playwright against real fleet portal + bench robot.

Requires deployed fleet host, Cognito, and bench Orin running ``krabby agent``
+ WebRTC edge (``--teleop-ip 127.0.0.1``). Uses the persistent CI operator
(``fleet.toml`` + ``COGNITO_CI_PASSWORD``). Skipped locally unless
``BENCH_E2E=1`` or GitHub Actions.

The bench HAL server needs ``--teleop-control-echo`` for the control-ack
assertion (see ``teleop/edge/robot_settings.py:build_teleop_edge_settings``).
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
from urllib.parse import quote

import pytest
import requests

from tests_e2e._fleet_env import AWS_REGION, BENCH_THING_NAME, FLEET_PORTAL_URL, FLEET_SERVICE_URL

SIGNALING_TIMEOUT_S = float(os.environ.get("TELEOP_E2E_SIGNALING_TIMEOUT_S", "90"))
VIDEO_TIMEOUT_S = float(os.environ.get("TELEOP_E2E_VIDEO_TIMEOUT_S", "120"))
CONTROL_TIMEOUT_S = float(os.environ.get("TELEOP_E2E_CONTROL_TIMEOUT_S", "60"))
MQTT_IDLE_SECS = float(os.environ.get("TELEOP_E2E_MQTT_IDLE_SECS", "8"))
ICE_STUCK_NEW_SECS = float(os.environ.get("TELEOP_E2E_ICE_STUCK_NEW_SECS", "30"))


def _viewer_url(thing: str, token: str) -> str:
    # e2e=1: single stream + catalog_ids aligned with one recvonly m-line (see teleop_session.js).
    return (
        f"{FLEET_PORTAL_URL}/teleop/viewer.html"
        f"?thing={quote(thing)}&e2e=1&token={quote(token)}"
    )


class _MqttSniffer:
    """Subscribe to teleop signaling topics; record inbound frames (SigV4 MQTT)."""

    def __init__(self, thing: str) -> None:
        self.thing = thing
        self.in_topic = f"teleop/{thing}/signaling/in"
        self.out_topic = f"teleop/{thing}/signaling/out"
        self.received_in: list[tuple[float, str]] = []
        self.received_out: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        self._connection: Any = None

    def start(self) -> None:
        from awscrt import auth, mqtt
        from awsiot import mqtt_connection_builder

        iot = __import__("boto3").client("iot", region_name=AWS_REGION)
        endpoint = iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]
        credentials_provider = auth.AwsCredentialsProvider.new_default_chain()
        client_id = f"teleop-e2e-sniffer-{os.getpid()}-{int(time.time())}"

        def _on_in(topic: str, payload: bytes, **kwargs: Any) -> None:
            text = payload.decode("utf-8", errors="replace")
            with self._lock:
                self.received_in.append((time.monotonic(), text))

        def _on_out(topic: str, payload: bytes, **kwargs: Any) -> None:
            text = payload.decode("utf-8", errors="replace")
            with self._lock:
                self.received_out.append((time.monotonic(), text))

        conn = mqtt_connection_builder.websockets_with_default_aws_signing(
            endpoint=endpoint,
            region=AWS_REGION,
            credentials_provider=credentials_provider,
            client_id=client_id,
            clean_session=True,
            keep_alive_secs=30,
        )
        conn.connect().result(timeout=30)
        conn.subscribe(topic=self.in_topic, qos=mqtt.QoS.AT_LEAST_ONCE, callback=_on_in)
        conn.subscribe(topic=self.out_topic, qos=mqtt.QoS.AT_LEAST_ONCE, callback=_on_out)
        self._connection = conn

    def stop(self) -> None:
        conn = self._connection
        self._connection = None
        if conn is None:
            return
        try:
            conn.disconnect().result(timeout=10)
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            self.received_in.clear()
            self.received_out.clear()

    def count_since(self, t0: float) -> tuple[int, int]:
        with self._lock:
            nin = sum(1 for t, _ in self.received_in if t >= t0)
            nout = sum(1 for t, _ in self.received_out if t >= t0)
            return nin, nout

    def out_has_type(self, msg_type: str) -> bool:
        with self._lock:
            for _, text in self.received_out:
                try:
                    if json.loads(text).get("type") == msg_type:
                        return True
                except json.JSONDecodeError:
                    continue
        return False


@pytest.fixture
def mqtt_sniffer() -> Any:
    sniffer = _MqttSniffer(BENCH_THING_NAME)
    sniffer.start()
    try:
        yield sniffer
    finally:
        sniffer.stop()


_TELEOP_DIAG_JS = """() => (
  window.__krabbyTeleop && window.__krabbyTeleop.getDiagnostics
    ? window.__krabbyTeleop.getDiagnostics()
    : null
)"""


def _format_diagnostic_timeline(timeline: list[tuple[float, Any | None]]) -> str:
    lines: list[str] = []
    for elapsed, diag in timeline[-24:]:
        if not diag:
            lines.append(f"  t+{elapsed:.1f}s: (no __krabbyTeleop / getDiagnostics)")
            continue
        lines.append(
            f"  t+{elapsed:.1f}s: status={diag.get('status')!r} "
            f"ice={diag.get('pcIceConnectionState')} conn={diag.get('pcConnectionState')} "
            f"dc={diag.get('controlDcReadyState')} live={diag.get('isSessionLive')} "
            f"playingBeforeLive={diag.get('playingLabelWhileNotLive')}"
        )
    return "diagnostic timeline:\n" + "\n".join(lines)


def _assert_no_bad_webrtc_sequence(diag: dict[str, Any], timeline: list[tuple[float, Any | None]]) -> None:
    if diag.get("webrtcFailed"):
        relay = diag.get("forceRelayIce")
        turn_n = diag.get("iceServerCount")
        raise AssertionError(
            "WebRTC ICE/connection entered failed state (fast-fail). "
            f"forceRelayIce={relay!r} iceServerCount={turn_n!r} — "
            "verify coturn on fleet host and GET /api/teleop/ice-servers returns TURN.\n"
            f"{_format_diagnostic_timeline(timeline)}"
        )
    if diag.get("playingLabelWhileNotLive"):
        raise AssertionError(
            "sequence: UI label 'Playing' while session not live "
            f"(ice={diag.get('pcIceConnectionState')}, dc={diag.get('controlDcReadyState')}) — "
            "deployed portal may still use pre-fix teleop_session.js, or ICE never finished.\n"
            f"{_format_diagnostic_timeline(timeline)}\n"
            f"statusHistory={diag.get('statusHistory')!r}"
        )
    status = str(diag.get("status") or "")
    if status.lower().startswith("webrtc error"):
        raise AssertionError(
            f"viewer reported WebRTC error: {status!r}\n{_format_diagnostic_timeline(timeline)}"
        )
    if (
        status.startswith("WebRTC connecting")
        and diag.get("pcSignalingState") == "stable"
        and diag.get("pcIceConnectionState") == "new"
        and timeline
    ):
        stuck_start: float | None = None
        for elapsed, d in timeline:
            if not d:
                continue
            st = str(d.get("status") or "")
            if (
                st.startswith("WebRTC connecting")
                and d.get("pcSignalingState") == "stable"
                and d.get("pcIceConnectionState") == "new"
            ):
                if stuck_start is None:
                    stuck_start = elapsed
            else:
                stuck_start = None
        if stuck_start is not None and timeline[-1][0] - stuck_start >= ICE_STUCK_NEW_SECS:
            raise AssertionError(
                "WebRTC ICE never left 'new' after SDP stable (fast-fail). "
                f"forceRelayIce={diag.get('forceRelayIce')!r} iceGathering={diag.get('pcIceGatheringState')!r} — "
                "avoid ?ice=relay in CI; ensure coturn + bench locomotion 0.1.1+ and optional "
                "KRABBY_TELEOP_TURN_* on HAL.\n"
                f"{_format_diagnostic_timeline(timeline)}"
            )


def _poll_teleop(
    page: Any,
    *,
    timeout_s: float,
    step: str,
    until_js: str,
) -> list[tuple[float, Any | None]]:
    """Poll getDiagnostics on a fixed interval; fail fast on bad sequence, not on blind timeout alone."""
    timeline: list[tuple[float, Any | None]] = []
    start = time.monotonic()
    deadline = start + timeout_s
    while time.monotonic() < deadline:
        elapsed = time.monotonic() - start
        diag = page.evaluate(_TELEOP_DIAG_JS)
        timeline.append((elapsed, diag))
        if diag:
            _assert_no_bad_webrtc_sequence(diag, timeline)
            if page.evaluate(until_js):
                return timeline
        time.sleep(0.25)
    last = timeline[-1][1] if timeline else None
    raise AssertionError(
        f"teleop E2E: {step} not satisfied within {timeout_s:.0f}s\n"
        f"{_format_diagnostic_timeline(timeline)}\n"
        f"last statusHistory={last.get('statusHistory') if last else None!r}"
    )


def _teleop_failure_context(page: Any, mqtt_sniffer: _MqttSniffer) -> str:
    diag = page.evaluate(_TELEOP_DIAG_JS)
    nin, nout = mqtt_sniffer.count_since(0)
    return (
        f"viewer getDiagnostics={diag!r}\n"
        f"mqtt signaling since test start: in={nin} out={nout}\n"
        "Bench needs krabby-agent teleop shim :9000 and HAL edge with "
        "--teleop-ip 127.0.0.1 (--teleop-control-echo for control-ack assertion). "
        "Deployed fleet portal must serve teleop_session.js with Playing = ICE + control DC open."
    )


def test_teleop_signaling_control_and_video(operator_token: str, mqtt_sniffer: _MqttSniffer):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    mqtt_sniffer.clear()
    url = _viewer_url(BENCH_THING_NAME, operator_token)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
            ],
        )
        page = browser.new_page()
        console_lines: list[str] = []

        def _on_console(msg: Any) -> None:
            console_lines.append(f"{msg.type}: {msg.text}")

        page.on("console", _on_console)
        page.on("pageerror", lambda err: console_lines.append(f"pageerror: {err}"))

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)

            def _fail_with_context(exc: AssertionError) -> None:
                tail = (
                    "\n".join(console_lines[-30:]) if console_lines else "(no browser console output)"
                )
                raise AssertionError(
                    f"{exc}\n{_teleop_failure_context(page, mqtt_sniffer)}\n"
                    f"browser console (last 30 lines):\n{tail}"
                ) from exc

            try:
                _poll_teleop(
                    page,
                    timeout_s=SIGNALING_TIMEOUT_S,
                    step="session live (ICE + control DC)",
                    until_js="""() => {
                      const d = window.__krabbyTeleop.getDiagnostics();
                      return d && d.isSessionLive;
                    }""",
                )
            except AssertionError as exc:
                _fail_with_context(exc)

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if mqtt_sniffer.out_has_type("hello_ack") or mqtt_sniffer.out_has_type("answer"):
                    break
                time.sleep(0.5)
            else:
                nin, nout = mqtt_sniffer.count_since(0)
                assert nin + nout > 0, "expected MQTT traffic on teleop/{thing}/signaling/*"

            sent = page.evaluate("() => window.__krabbyTeleop.sendMotionSafeControl()")
            assert sent is True

            try:
                _poll_teleop(
                    page,
                    timeout_s=CONTROL_TIMEOUT_S,
                    step="last_control.RS echo on telemetry DC",
                    until_js="""() => {
                      const t = window.__krabbyTeleop.getLastTelemetry();
                      return !!(t && t.last_control && t.last_control.RS === true);
                    }""",
                )
            except AssertionError as exc:
                _fail_with_context(exc)

            try:
                _poll_teleop(
                    page,
                    timeout_s=VIDEO_TIMEOUT_S,
                    step="at least one video tile",
                    until_js="() => window.__krabbyTeleop.videoTrackCount() >= 1",
                )
            except AssertionError as exc:
                _fail_with_context(exc)
        finally:
            browser.close()

    time.sleep(2.0)
    mqtt_sniffer.clear()
    t0 = time.monotonic()
    time.sleep(MQTT_IDLE_SECS)
    nin, nout = mqtt_sniffer.count_since(t0)
    assert nin == 0 and nout == 0, (
        f"expected teleop signaling idle after teardown; got in={nin} out={nout}"
    )


def test_teleop_ice_servers_authed(operator_token: str):
    resp = requests.get(
        f"{FLEET_SERVICE_URL}/teleop/ice-servers",
        headers={"Authorization": f"Bearer {operator_token}"},
        timeout=30,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("version") == 1
    assert isinstance(body.get("iceServers"), list) and body["iceServers"]
    urls = []
    for entry in body["iceServers"]:
        u = entry.get("urls")
        if isinstance(u, str):
            urls.append(u)
        elif isinstance(u, list):
            urls.extend(u)
    assert any(u.startswith("stun:") for u in urls)
