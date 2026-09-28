"""krabby agent — the always-on MQTT client, run by krabby-agent.service.

Single outbound MQTT connection (mTLS, cert/key written by `krabby enroll`)
carries three flows for the life of the process:

1. Classic Shadow: publishes `{"state": {"reported": {...}}}` once a minute
   via the SDK's shadow client, so `update/accepted`/`update/rejected` are
   handled by the SDK rather than hand-rolled here.
2. Secure Tunneling: subscribes to the reserved `.../tunnels/notify` topic
   and spawns the destination `localproxy` against localhost:22 on
   notification. This is intentionally a minimal spawn/track/reap loop.
3. Teleop signaling: `teleop/{thing}/signaling/in|out` bridged by
   ``krabby.teleop_shim`` to a localhost WebSocket that the existing WebRTC
   edge agent dials (``--teleop-ip 127.0.0.1``). SDP/ICE JSON shape is unchanged.

No flags — everything needed (thing name, endpoint, cert paths) comes from
what `krabby enroll` already wrote to `_iot.IOT_DIR`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

from krabby import _iot
from krabby.teleop_shim import TeleopSignalingShim

SHADOW_REPORT_INTERVAL_SECS = 60

# Secure Tunneling destination proxy: spawn on notify, reap on exit, don't
# leave zombies.
_LOCALPROXY_BIN = "localproxy"
_SSH_DEST = "localhost:22"
_tunnel_procs: list[subprocess.Popen] = []


def _reap_tunnel_procs() -> None:
    """Drop finished child processes so they don't linger as zombies.

    localproxy exits on its own when AWS closes the tunnel (there is no
    separate "tunnel closed" message on the notify topic) — so "kill on
    close" is really just "notice it already exited and stop tracking it".
    """
    global _tunnel_procs
    still_running = []
    for proc in _tunnel_procs:
        ret = proc.poll()
        if ret is None:
            still_running.append(proc)
        else:
            print(f"[ok]  localproxy (pid {proc.pid}) exited (code {ret})")
    _tunnel_procs = still_running


def _on_tunnel_notify(topic: str, payload: bytes, **kwargs: Any) -> None:
    import shutil

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        print(f"[err] malformed tunnel notify payload on {topic}", file=sys.stderr)
        return

    token = data.get("clientAccessToken")
    region = data.get("region")
    services = data.get("services") or ["SSH"]
    if not token or not region:
        print(f"[err] tunnel notify missing clientAccessToken/region: {data}", file=sys.stderr)
        return
    if not shutil.which(_LOCALPROXY_BIN):
        print(f"[err] {_LOCALPROXY_BIN} not installed — run `krabby enroll` again", file=sys.stderr)
        return

    print(f"[+]   tunnel notify received (services={services}) — spawning destination localproxy")
    # -c /etc/ssl/certs: arm64/Jetson OpenSSL often fails TLS to the tunneling
    # endpoint with "unregistered scheme (STORE routines)" without an explicit CA path
    # (https://github.com/aws-samples/aws-iot-securetunneling-localproxy/issues/151).
    proc = subprocess.Popen([
        _LOCALPROXY_BIN,
        "-d", _SSH_DEST,
        "-t", token,
        "-r", region,
        "-c", "/etc/ssl/certs",
    ])
    _tunnel_procs.append(proc)


def _on_shadow_update_accepted(response: Any) -> None:
    pass  # SDK already confirmed the write; nothing else to do for this task.


def _on_shadow_update_rejected(error: Any) -> None:
    print(f"[err] shadow update rejected: code={error.code} message={error.message}", file=sys.stderr)


def _publish_shadow_report(shadow_client: Any, thing_name: str, teleop_shim: TeleopSignalingShim) -> None:
    from awscrt import mqtt
    from awsiot import iotshadow

    from krabby.telemetry import collect_telemetry

    reported = collect_telemetry(teleop_edge_connected=teleop_shim.robot_edge_connected())
    request = iotshadow.UpdateShadowRequest(
        thing_name=thing_name,
        state=iotshadow.ShadowState(reported=reported),
    )
    shadow_client.publish_update_shadow(request, mqtt.QoS.AT_LEAST_ONCE)


def cmd_agent() -> None:
    from awscrt import mqtt
    from awsiot import iotshadow

    config = _iot.load_config()
    thing_name = config["thing_name"]
    endpoint = config["endpoint"]

    print(f"Connecting to {endpoint} as {thing_name} ...")
    connection = _iot.build_mqtt_connection(thing_name, endpoint)
    connection.connect().result()
    print(f"[ok]  connected as {thing_name}")

    shadow_client = iotshadow.IotShadowClient(connection)
    shadow_client.subscribe_to_update_shadow_accepted(
        request=iotshadow.UpdateShadowSubscriptionRequest(thing_name=thing_name),
        qos=mqtt.QoS.AT_LEAST_ONCE,
        callback=_on_shadow_update_accepted,
    )
    shadow_client.subscribe_to_update_shadow_rejected(
        request=iotshadow.UpdateShadowSubscriptionRequest(thing_name=thing_name),
        qos=mqtt.QoS.AT_LEAST_ONCE,
        callback=_on_shadow_update_rejected,
    )

    tunnels_notify_topic = f"$aws/things/{thing_name}/tunnels/notify"
    connection.subscribe(
        topic=tunnels_notify_topic,
        qos=mqtt.QoS.AT_LEAST_ONCE,
        callback=_on_tunnel_notify,
    )
    print(f"[ok]  subscribed to {tunnels_notify_topic}")

    teleop_shim = TeleopSignalingShim()
    teleop_shim.start(connection, thing_name)

    print(f"      reporting shadow state every {SHADOW_REPORT_INTERVAL_SECS}s. Ctrl-C to stop.")
    try:
        while True:
            _reap_tunnel_procs()
            _publish_shadow_report(shadow_client, thing_name, teleop_shim)
            time.sleep(SHADOW_REPORT_INTERVAL_SECS)
    except KeyboardInterrupt:
        print("\n[ok]  shutting down")
    finally:
        teleop_shim.stop()
        for proc in _tunnel_procs:
            if proc.poll() is None:
                proc.terminate()
        connection.disconnect().result()
