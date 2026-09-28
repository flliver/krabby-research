"""Control-plane E2E against the permanently enrolled bench Orin.

Skipped locally unless ``BENCH_E2E=1`` or GitHub Actions; when enabled, missing
config, AWS access, tools, or bench connectivity → **fail**.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import boto3
import pytest

from krabby_fleet_config.e2e_env import AWS_REGION, BENCH_THING_NAME

SHADOW_MAX_AGE_SECS = int(os.environ.get("SHADOW_MAX_AGE_SECS", "180"))

KRAB_THING_TYPE = "Krab"
KRAB_DEVICE_POLICY = "KrabDevicePolicy"
AMAZON_ROOT_CA_URL = "https://www.amazontrust.com/repository/AmazonRootCA1.pem"
_LOCALPROXY_BIN = "localproxy"
_TUNNEL_E2E_DEBUG = os.environ.get("TUNNEL_E2E_DEBUG", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
_POLL_TRACE_MAX = 50


def _tunnel_from_describe(response: dict[str, Any]) -> dict[str, Any]:
    nested = response.get("tunnel")
    if isinstance(nested, dict):
        return nested
    return response


def _format_tunnel_connection_state(
    response: dict[str, Any], *, thing_name: str, tunnel_id: str
) -> str:
    tunnel = _tunnel_from_describe(response)
    dest = tunnel.get("destinationConnectionState") or {}
    src = tunnel.get("sourceConnectionState") or {}
    return (
        f"thing={thing_name} tunnel_id={tunnel_id} "
        f"tunnel_status={tunnel.get('status')!r} "
        f"destination={dest.get('status')!r} source={src.get('status')!r} "
        f"description={tunnel.get('description')!r}"
    )


def _probe_tcp_connect(host: str, port: int, timeout: float = 0.5) -> str:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "tcp_ok"
    except OSError as exc:
        return f"tcp_error:{exc!r}"


def _record_poll_trace(trace: list[str], line: str) -> None:
    trace.append(line)
    if len(trace) > _POLL_TRACE_MAX:
        del trace[0]
    if _TUNNEL_E2E_DEBUG:
        print(f"[tunnel-e2e] {line}", file=sys.stderr, flush=True)


def _format_poll_trace(trace: list[str]) -> str:
    if not trace:
        return "(no poll samples recorded)"
    return "\n".join(trace)


def _fetch_tunnel_connection_state(client: Any, tunnel_id: str, thing_name: str) -> str:
    try:
        return _format_tunnel_connection_state(
            client.describe_tunnel(tunnelId=tunnel_id),
            thing_name=thing_name,
            tunnel_id=tunnel_id,
        )
    except Exception as exc:  # noqa: BLE001 — diagnostics only
        return f"describe_tunnel failed for tunnel_id={tunnel_id}: {exc!r}"


def _start_localproxy_stderr_drain(proc: subprocess.Popen) -> list[bytes]:
    """Read localproxy output in a thread so a full PIPE cannot block the process."""
    chunks: list[bytes] = []
    stream = proc.stderr if proc.stderr is not None else proc.stdout

    def _reader() -> None:
        if stream is None:
            return
        try:
            while True:
                part = stream.read(4096)
                if not part:
                    break
                chunks.append(part)
        except OSError:
            pass

    threading.Thread(target=_reader, daemon=True).start()
    return chunks


def _localproxy_stderr_text(
    proc: subprocess.Popen, captured: list[bytes] | None = None, max_bytes: int = 8192
) -> str:
    if captured is not None:
        raw = b"".join(captured)[:max_bytes]
        text = raw.decode("utf-8", errors="replace").strip()
        return text or "(localproxy output empty)"
    stream = proc.stderr if proc.stderr is not None else proc.stdout
    if stream is None:
        return "(localproxy output not captured)"
    try:
        chunks: list[bytes] = []
        fd = stream.fileno()
        while sum(len(part) for part in chunks) < max_bytes:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            part = os.read(fd, 1024)
            if not part:
                break
            chunks.append(part)
        text = b"".join(chunks).decode("utf-8", errors="replace").strip()
        return text or "(localproxy stderr empty)"
    except OSError as exc:
        return f"(could not read localproxy stderr: {exc})"


def _localproxy_failure(
    proc: subprocess.Popen, detail: str, stderr_chunks: list[bytes] | None = None
) -> AssertionError:
    code = proc.returncode if proc.poll() is not None else "still running"
    return AssertionError(
        f"{detail} (localproxy pid={proc.pid} exit={code})\n"
        f"localproxy stderr:\n{_localproxy_stderr_text(proc, stderr_chunks)}"
    )


def _wait_tunnel_destination_connected(
    client: Any, tunnel_id: str, thing_name: str, timeout: float = 60.0
) -> None:
    """Wait until bench destination localproxy has joined the tunnel."""
    started = time.monotonic()
    deadline = started + timeout
    trace: list[str] = []
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        elapsed = time.monotonic() - started
        last = client.describe_tunnel(tunnelId=tunnel_id)
        tunnel = _tunnel_from_describe(last)
        dest = (tunnel.get("destinationConnectionState") or {}).get("status")
        src = (tunnel.get("sourceConnectionState") or {}).get("status")
        _record_poll_trace(trace, f"t+{elapsed:.2f}s dest={dest!r} source={src!r}")
        if dest == "CONNECTED":
            return
        time.sleep(0.5)
    raise AssertionError(
        "destination localproxy did not connect within "
        f"{timeout:.0f}s — check krabby-agent on the bench (tunnels/notify, destination localproxy).\n"
        f"{_format_tunnel_connection_state(last, thing_name=thing_name, tunnel_id=tunnel_id)}\n"
        f"Poll trace (describe_tunnel every ~500ms):\n{_format_poll_trace(trace)}"
    )


def _wait_tunnel_source_connected(
    client: Any,
    tunnel_id: str,
    thing_name: str,
    local_port: int,
    proc: subprocess.Popen,
    stderr_chunks: list[bytes],
    timeout: float = 30.0,
) -> None:
    """Wait until source localproxy has joined the tunnel (DescribeTunnel gate)."""
    started = time.monotonic()
    deadline = started + timeout
    trace: list[str] = []
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                "source localproxy exited before tunnel source CONNECTED "
                f"(exit={proc.returncode})\n"
                f"Poll trace:\n{_format_poll_trace(trace)}\n"
                f"localproxy stderr:\n{_localproxy_stderr_text(proc, stderr_chunks)}"
            )
        elapsed = time.monotonic() - started
        last = client.describe_tunnel(tunnelId=tunnel_id)
        tunnel = _tunnel_from_describe(last)
        dest = (tunnel.get("destinationConnectionState") or {}).get("status")
        src = (tunnel.get("sourceConnectionState") or {}).get("status")
        tcp = _probe_tcp_connect("127.0.0.1", local_port)
        _record_poll_trace(
            trace,
            f"t+{elapsed:.2f}s dest={dest!r} source={src!r} {tcp} proc=running",
        )
        if src == "CONNECTED" and tcp == "tcp_ok":
            return
        time.sleep(0.2)
    raise AssertionError(
        "source localproxy did not reach CONNECTED with a listening local port within "
        f"{timeout:.0f}s — {_format_tunnel_connection_state(last, thing_name=thing_name, tunnel_id=tunnel_id)}\n"
        f"Poll trace (describe_tunnel + tcp probe to 127.0.0.1:{local_port} every ~200ms):\n"
        f"{_format_poll_trace(trace)}\n"
        f"localproxy stderr:\n{_localproxy_stderr_text(proc, stderr_chunks)}"
    )


def _read_ssh_banner(
    local_port: int,
    proc: subprocess.Popen,
    deadline: float,
    stderr_chunks: list[bytes] | None = None,
) -> bytes:
    banner = b""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise _localproxy_failure(
                proc, "source localproxy exited while reading SSH banner", stderr_chunks
            )
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=2.0) as conn:
                conn.settimeout(2.0)
                while time.monotonic() < deadline:
                    chunk = conn.recv(64)
                    if chunk:
                        banner += chunk
                        if banner.startswith(b"SSH-"):
                            return banner
                    else:
                        break
        except OSError:
            pass
        time.sleep(0.5)
    return banner


def _iot() -> Any:
    return boto3.client("iot", region_name=AWS_REGION)


def _iot_data() -> Any:
    iot = _iot()
    endpoint = iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]
    return boto3.client("iot-data", endpoint_url=f"https://{endpoint}", region_name=AWS_REGION)


def _tunneling() -> Any:
    return boto3.client("iotsecuretunneling", region_name=AWS_REGION)


def _parse_indexed_shadow(shadow: Any) -> dict[str, Any]:
    if shadow is None:
        return {}
    if isinstance(shadow, str):
        try:
            shadow = json.loads(shadow)
        except json.JSONDecodeError:
            return {}
    if not isinstance(shadow, dict):
        return {}
    reported = shadow.get("reported")
    return reported if isinstance(reported, dict) else {}


def _search_bench() -> dict[str, Any]:
    resp = _iot().search_index(
        queryString=f"thingName:{BENCH_THING_NAME} AND thingTypeName:{KRAB_THING_TYPE}",
        maxResults=1,
    )
    things = resp.get("things") or []
    assert things, f"SearchIndex returned no hit for {BENCH_THING_NAME}"
    return things[0]


def test_control_plane_prereqs():
    """Persistent ControlPlaneStack resources exist (reuse, don't redeploy)."""
    iot = _iot()
    iot.get_policy(policyName=KRAB_DEVICE_POLICY)
    iot.describe_thing_type(thingTypeName=KRAB_THING_TYPE)
    iot.describe_thing(thingName=BENCH_THING_NAME)


def test_bench_connectivity_and_shadow_index():
    """Connected + recent shadow reported.timestamp via Fleet Indexing."""
    thing = _search_bench()
    connectivity = thing.get("connectivity") or {}
    assert connectivity.get("connected") is True, (
        f"{BENCH_THING_NAME} is offline (connectivity={connectivity!r}) — bench offline = CI red"
    )
    assert connectivity.get("timestamp"), "missing connectivity.timestamp"

    reported = _parse_indexed_shadow(thing.get("shadow"))
    assert "timestamp" in reported, f"indexed shadow missing reported.timestamp: {reported!r}"
    age = time.time() - int(reported["timestamp"])
    assert age < SHADOW_MAX_AGE_SECS, (
        f"shadow reported.timestamp is {age:.0f}s old (limit {SHADOW_MAX_AGE_SECS}s) — "
        "is krabby-agent running on the bench?"
    )


def test_bench_get_thing_shadow_schema():
    """GetThingShadow returns expected open-ended reported fields."""
    resp = _iot_data().get_thing_shadow(thingName=BENCH_THING_NAME)
    payload = json.loads(resp["payload"].read())
    reported = payload.get("state", {}).get("reported")
    assert isinstance(reported, dict), f"expected state.reported object, got {reported!r}"
    assert isinstance(reported.get("timestamp"), int), reported
    # Agent may publish reported_image=null when no image is installed; AWS IoT
    # Device Shadow treats null as delete, so the key is often absent until
    # krabby install/update has written a ref.
    image = reported.get("reported_image")
    assert image is None or isinstance(image, str), image

    age = time.time() - int(reported["timestamp"])
    assert age < SHADOW_MAX_AGE_SECS, (
        f"GetThingShadow timestamp is {age:.0f}s old (limit {SHADOW_MAX_AGE_SECS}s)"
    )


@pytest.fixture
def scratch_device(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """Provision a throwaway thing+cert with KrabDevicePolicy; tear down after."""
    iot = _iot()
    scratch_name = f"e2e-scratch-{os.urandom(4).hex()}"
    cert_path = tmp_path / "device.pem.crt"
    key_path = tmp_path / "private.pem.key"
    ca_path = tmp_path / "AmazonRootCA1.pem"

    iot.create_thing(thingName=scratch_name, thingTypeName=KRAB_THING_TYPE)
    created = iot.create_keys_and_certificate(setAsActive=True)
    cert_arn = created["certificateArn"]
    cert_id = created["certificateId"]
    cert_path.write_text(created["certificatePem"])
    key_path.write_text(created["keyPair"]["PrivateKey"])
    with urllib.request.urlopen(AMAZON_ROOT_CA_URL, timeout=15) as resp:
        ca_path.write_bytes(resp.read())

    iot.attach_policy(policyName=KRAB_DEVICE_POLICY, target=cert_arn)
    iot.attach_thing_principal(thingName=scratch_name, principal=cert_arn)

    endpoint = iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]
    info = {
        "thing_name": scratch_name,
        "cert_arn": cert_arn,
        "cert_id": cert_id,
        "cert_path": cert_path,
        "key_path": key_path,
        "ca_path": ca_path,
        "endpoint": endpoint,
    }
    try:
        yield info
    finally:
        try:
            iot.detach_policy(policyName=KRAB_DEVICE_POLICY, target=cert_arn)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        try:
            iot.detach_thing_principal(thingName=scratch_name, principal=cert_arn)
        except Exception:  # noqa: BLE001
            pass
        try:
            iot.update_certificate(certificateId=cert_id, newStatus="INACTIVE")
            iot.delete_certificate(certificateId=cert_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            iot.delete_thing(thingName=scratch_name)
        except Exception:  # noqa: BLE001
            pass


def test_scratch_cert_cannot_update_bench_shadow(scratch_device: dict[str, Any]):
    """Per-thing isolation — scratch cert cannot write the bench shadow."""
    from awscrt import mqtt
    from awsiot import iotshadow, mqtt_connection_builder

    probe_key = f"e2e_isolation_{os.urandom(3).hex()}"
    before = json.loads(_iot_data().get_thing_shadow(thingName=BENCH_THING_NAME)["payload"].read())
    before_reported = (before.get("state") or {}).get("reported") or {}

    connection = mqtt_connection_builder.mtls_from_path(
        endpoint=scratch_device["endpoint"],
        cert_filepath=str(scratch_device["cert_path"]),
        pri_key_filepath=str(scratch_device["key_path"]),
        ca_filepath=str(scratch_device["ca_path"]),
        client_id=scratch_device["thing_name"],
        clean_session=True,
        keep_alive_secs=30,
    )
    connection.connect().result(timeout=20)
    shadow_client = iotshadow.IotShadowClient(connection)

    rejected: list[Any] = []
    accepted: list[Any] = []

    # Subscriptions on the *bench* topics should also be denied; we still try
    # the publish and verify the bench shadow never gains our probe key.
    try:
        shadow_client.subscribe_to_update_shadow_rejected(
            request=iotshadow.UpdateShadowSubscriptionRequest(thing_name=BENCH_THING_NAME),
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=lambda err: rejected.append(err),
        )
    except Exception:  # noqa: BLE001 - deny is success for isolation
        pass
    try:
        shadow_client.subscribe_to_update_shadow_accepted(
            request=iotshadow.UpdateShadowSubscriptionRequest(thing_name=BENCH_THING_NAME),
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=lambda resp: accepted.append(resp),
        )
    except Exception:  # noqa: BLE001
        pass

    request = iotshadow.UpdateShadowRequest(
        thing_name=BENCH_THING_NAME,
        state=iotshadow.ShadowState(reported={probe_key: True, "timestamp": int(time.time())}),
    )
    publish_error: Exception | None = None
    try:
        shadow_client.publish_update_shadow(request, mqtt.QoS.AT_LEAST_ONCE).result(timeout=15)
    except Exception as exc:  # noqa: BLE001 - unauthorized publish is expected
        publish_error = exc

    # Give any illicit accepted delivery a moment, then disconnect.
    time.sleep(2)
    connection.disconnect().result(timeout=15)

    after = json.loads(_iot_data().get_thing_shadow(thingName=BENCH_THING_NAME)["payload"].read())
    after_reported = (after.get("state") or {}).get("reported") or {}
    assert probe_key not in after_reported, (
        f"scratch cert wrote {probe_key} into bench shadow — isolation broken "
        f"(publish_error={publish_error!r}, rejected={rejected!r})"
    )
    assert not accepted, f"unexpected update/accepted for bench shadow: {accepted!r}"
    if "timestamp" in before_reported:
        assert "timestamp" in after_reported


def test_secure_tunnel_source_proxy_reaches_ssh():
    """OpenTunnel → destination localproxy on bench → source TCP sees SSH."""
    assert shutil.which(_LOCALPROXY_BIN), f"{_LOCALPROXY_BIN} not on PATH (required for bench E2E)"
    client = _tunneling()
    tunnel = client.open_tunnel(
        description=f"task1-e2e:{BENCH_THING_NAME}",
        destinationConfig={"thingName": BENCH_THING_NAME, "services": ["SSH"]},
        timeoutConfig={"maxLifetimeTimeoutMinutes": 30},
    )
    tunnel_id = tunnel["tunnelId"]
    source_token = tunnel["sourceAccessToken"]
    # Secure Tunneling: destination must connect before source (DescribeTunnel gate).
    _wait_tunnel_destination_connected(client, tunnel_id, BENCH_THING_NAME)
    proc: subprocess.Popen | None = None
    stderr_chunks: list[bytes] = []
    local_port = 0
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            local_port = s.getsockname()[1]

        proc = subprocess.Popen(
            [
                _LOCALPROXY_BIN,
                "-s",
                str(local_port),
                "-b",
                "127.0.0.1",
                "-t",
                source_token,
                "-r",
                AWS_REGION,
                "-c",
                "/etc/ssl/certs",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stderr_chunks = _start_localproxy_stderr_drain(proc)
        _wait_tunnel_source_connected(
            client, tunnel_id, BENCH_THING_NAME, local_port, proc, stderr_chunks
        )
        banner = _read_ssh_banner(
            local_port, proc, time.monotonic() + 30, stderr_chunks
        )
        if not banner.startswith(b"SSH-"):
            diag = _fetch_tunnel_connection_state(client, tunnel_id, BENCH_THING_NAME)
            tcp = _probe_tcp_connect("127.0.0.1", local_port)
            lp = _localproxy_stderr_text(proc, stderr_chunks) if proc is not None else ""
            raise AssertionError(
                "expected SSH banner through Secure Tunnel, "
                f"got {banner!r} on 127.0.0.1:{local_port}\n"
                f"{diag}\n"
                f"tcp_probe 127.0.0.1:{local_port}: {tcp}\n"
                "If destination=CONNECTED but banner is empty, check sshd on the bench (:22) "
                "and source localproxy on the runner.\n"
                f"localproxy stderr:\n{lp}"
            )
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            client.close_tunnel(tunnelId=tunnel_id, delete=True)
        except Exception:  # noqa: BLE001
            pass
