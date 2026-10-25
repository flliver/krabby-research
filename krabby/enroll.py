"""krabby enroll — one-time, operator-run fleet onboarding.

Uses AWS creds available at enroll time (only once — nothing AWS-shaped is
persisted) to create/find the IoT thing, generate a keypair + CSR on-device
(the private key never leaves the Orin — see `_generate_keypair_and_csr`),
get it signed by IoT Core, attach the fleet's per-thing policy, and write the
resulting identity to disk. Then installs the Secure Tunneling local proxy,
enables `krabby-agent.service`, and does a short-lived MQTT connect to prove
the new identity actually works before handing off to the always-on agent.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from krabby import _iot

# Must match ControlPlaneStack (fleet/infra/control_plane_stack.py) exactly —
# duplicated as plain constants here rather than imported, since krabby-launcher
# ships standalone (pip install krabby-launcher) with no dependency on the
# fleet/infra CDK package.
KRAB_THING_TYPE = "Krab"
KRAB_DEVICE_POLICY = "KrabDevicePolicy"

_NET_CLASS_DIR = Path("/sys/class/net")


def _wired_mac_address() -> Optional[str]:
    """MAC address of the Orin's wired (Ethernet) interface, colon-formatted.

    A `wireless` subdirectory under sysfs is the standard signal an interface
    is WiFi rather than Ethernet; type "1" is ARPHRD_ETHER. Skips loopback and
    any interface without a real address (unplugged NICs still report one).
    """
    if not _NET_CLASS_DIR.is_dir():
        return None
    for iface_path in sorted(_NET_CLASS_DIR.iterdir()):
        if iface_path.name == "lo" or (iface_path / "wireless").exists():
            continue
        type_path, addr_path = iface_path / "type", iface_path / "address"
        if not (type_path.exists() and addr_path.exists()):
            continue
        try:
            if type_path.read_text().strip() != "1":
                continue
            mac = addr_path.read_text().strip()
        except OSError:
            continue
        if mac and mac != "00:00:00:00:00:00":
            return mac
    return None


def _default_thing_name() -> str:
    """The wired NIC's MAC address, used when `--thing-name` isn't given.

    No hostname fallback: hostnames aren't unique across a fleet (every Orin
    devkit ships with the same default hostname unless manually renamed), so
    using one here would silently collide thing names at scale. The wired
    MAC is hardware-unique and present on every Orin devkit.
    """
    mac = _wired_mac_address()
    if mac:
        return f"krab-{mac.replace(':', '-')}"

    print(
        "[err] could not determine a default thing name (no wired MAC address) "
        "— pass --thing-name explicitly",
        file=sys.stderr,
    )
    sys.exit(1)


def _ensure_control_plane_deployed(iot_client) -> None:
    """Fail fast with a clean message if `ControlPlaneStack` hasn't been deployed yet.

    Without this, the first AWS call that actually needs `KrabDevicePolicy`
    (attach_policy, well after the thing's already been created and a cert
    already issued) raises a raw boto3 ClientError traceback — confusing for
    an operator whose real mistake was just deploying order, not a permissions
    or config issue. Checking the policy up front catches the missing-stack
    case before any AWS resource is created, so a bad run leaves nothing to
    clean up.
    """
    try:
        iot_client.get_policy(policyName=KRAB_DEVICE_POLICY)
    except iot_client.exceptions.ResourceNotFoundException:
        print(
            f"[err] IoT policy '{KRAB_DEVICE_POLICY}' not found — has ControlPlaneStack "
            "been deployed? Run `fleet/infra/scripts/deploy-control-plane.sh` first.",
            file=sys.stderr,
        )
        sys.exit(1)


def _ensure_thing(iot_client, thing_name: str) -> None:
    try:
        iot_client.describe_thing(thingName=thing_name)
        print(f"[ok]  thing already exists: {thing_name}")
    except iot_client.exceptions.ResourceNotFoundException:
        iot_client.create_thing(thingName=thing_name, thingTypeName=KRAB_THING_TYPE)
        print(f"[+]   created thing: {thing_name} (type={KRAB_THING_TYPE})")


def _resolve_ats_endpoint(iot_client) -> str:
    resp = iot_client.describe_endpoint(endpointType="iot:Data-ATS")
    return resp["endpointAddress"]


def _generate_keypair_and_csr(thing_name: str) -> tuple[bytes, bytes]:
    """RSA-2048 keypair + CSR, generated locally. The private key is returned
    only to be written straight to disk by the caller — it is never sent
    anywhere, including to AWS (only the CSR, a public-key artifact, is)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, thing_name)]))
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    csr_pem = csr.public_bytes(serialization.Encoding.PEM)
    return key_pem, csr_pem


def _create_cert_from_csr(iot_client, csr_pem: bytes) -> tuple[str, bytes]:
    resp = iot_client.create_certificate_from_csr(
        certificateSigningRequest=csr_pem.decode("ascii"),
        setAsActive=True,
    )
    cert_arn = resp["certificateArn"]
    cert_pem = resp["certificatePem"].encode("ascii")
    print(f"[+]   certificate created and activated: {resp['certificateId']}")
    return cert_arn, cert_pem


def _attach_policy_and_thing(iot_client, cert_arn: str, thing_name: str) -> None:
    iot_client.attach_policy(policyName=KRAB_DEVICE_POLICY, target=cert_arn)
    print(f"[+]   attached policy {KRAB_DEVICE_POLICY} to certificate")
    iot_client.attach_thing_principal(thingName=thing_name, principal=cert_arn)
    print(f"[+]   attached certificate to thing {thing_name}")


def _ensure_localproxy_installed() -> bool:
    """Install aws-iot-securetunneling-localproxy (apt package), skipping if present.

    Assumes the package is resolvable via the host's configured apt sources
    (adding AWS's apt repo, if the distro doesn't carry it directly, is a
    one-time host-image concern, not something `krabby enroll` provisions
    on every run).
    """
    if shutil.which("localproxy"):
        print("[ok]  aws-iot-securetunneling-localproxy already installed")
        return True
    if not shutil.which("apt-get"):
        print("[skip] apt-get not found — skipping localproxy install (not a Debian/Ubuntu host?)")
        return True
    print("      installing aws-iot-securetunneling-localproxy ...")
    if subprocess.run(["apt-get", "install", "-y", "aws-iot-securetunneling-localproxy"]).returncode != 0:
        print("[err] apt-get install aws-iot-securetunneling-localproxy failed", file=sys.stderr)
        return False
    print("[+]   aws-iot-securetunneling-localproxy installed")
    return True


def _verify_connect(thing_name: str, endpoint: str) -> bool:
    print(f"      verifying MQTT connect to {endpoint} as {thing_name} ...")
    connection = _iot.build_mqtt_connection(thing_name, endpoint)
    try:
        connection.connect().result(timeout=15)
    except Exception as exc:  # noqa: BLE001 - report any connect failure, SDK raises varied types
        print(f"[err] MQTT connect failed: {exc}", file=sys.stderr)
        return False
    print("[ok]  MQTT connect verified")
    connection.disconnect().result(timeout=15)
    return True


def cmd_enroll(
    thing_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    *,
    locomotion_control_source: Optional[str] = None,
    locomotion_robot: Optional[str] = None,
    locomotion_checkpoint: Optional[str] = None,
    locomotion_checkpoint_host_dir: Optional[str] = None,
    locomotion_teleop_control_echo: bool = False,
) -> None:
    import boto3

    thing = thing_name or _default_thing_name()
    print(f"Enrolling {thing} ...")

    iot_client = boto3.client("iot")
    _ensure_control_plane_deployed(iot_client)
    _ensure_thing(iot_client, thing)

    resolved_endpoint = endpoint or _resolve_ats_endpoint(iot_client)
    print(f"[ok]  ATS endpoint: {resolved_endpoint}")

    key_pem, csr_pem = _generate_keypair_and_csr(thing)
    cert_arn, cert_pem = _create_cert_from_csr(iot_client, csr_pem)
    _attach_policy_and_thing(iot_client, cert_arn, thing)

    print("      fetching Amazon Root CA ...")
    root_ca_pem = _iot.fetch_amazon_root_ca()

    _iot.write_identity(thing, resolved_endpoint, cert_pem, key_pem, root_ca_pem)
    print(f"[+]   wrote device identity to {_iot.IOT_DIR}")

    from krabby._host import _ensure_boot_service
    from krabby._locomotion_config import LOCOMOTION_SERVICE_NAME, write_default_config

    write_default_config(
        control_source=locomotion_control_source,
        robot=locomotion_robot,
        checkpoint=locomotion_checkpoint,
        checkpoint_host_dir=locomotion_checkpoint_host_dir,
        teleop_control_echo=locomotion_teleop_control_echo or None,
    )

    ok = _ensure_localproxy_installed()
    ok &= _iot.ensure_agent_service()
    ok &= _ensure_boot_service(launch_on_startup=True)
    ok &= _verify_connect(thing, resolved_endpoint)

    if ok:
        subprocess.run(["systemctl", "start", _iot.AGENT_SERVICE_NAME], check=False)
        subprocess.run(["systemctl", "start", LOCOMOTION_SERVICE_NAME], check=False)
        print(
            f"\n[ok]  Enrolled {thing}. "
            f"`{_iot.AGENT_SERVICE_NAME}` and `{LOCOMOTION_SERVICE_NAME}` enabled "
            f"(fleet teleop HAL via `krabby run` on boot)."
        )
    else:
        print("\n[err] Enroll finished with errors — see above.", file=sys.stderr)
        sys.exit(1)
