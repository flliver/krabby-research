"""CDK app entry for Krabby fleet infra."""

from __future__ import annotations

import os
import sys

import aws_cdk as cdk

from control_plane_stack import ControlPlaneStack
from fleet_service_stack import FleetServiceStack


app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
)

control_plane_stack = ControlPlaneStack(
    app,
    "ControlPlaneStack",
    description="Krabby fleet IoT control plane (thing type, device policy, Fleet Indexing, presence)",
    env=env,
)

# FleetServiceStack's domain/DNS values aren't known project constants yet
# (no domain is registered) and aren't per-shell environment state either --
# they're required CDK context, passed as `-c key=value` on the command line
# (deploy-fleet-service.sh forwards its own args straight through). CDK
# constructs the whole app tree regardless of which stack a `cdk` command
# targets, so requiring these unconditionally would break
# `cdk deploy/synth/diff ControlPlaneStack` for anyone who hasn't decided on
# a domain yet. Only build FleetServiceStack when both required context
# values are present; a partial set (one supplied, the other missing) still
# fails loudly rather than silently deploying with a gap.
_FLEET_CONTEXT_KEYS = ("domainName", "hostedZoneName")
_fleet_context = {key: app.node.try_get_context(key) for key in _FLEET_CONTEXT_KEYS}

if all(_fleet_context.values()):
    fleet_service_stack = FleetServiceStack(
        app,
        "FleetServiceStack",
        description="Krabby fleet service host (EC2 + EIP + Route 53 + Cognito + instance IAM + GitHub Actions OIDC)",
        domain_name=_fleet_context["domainName"],
        hosted_zone_name=_fleet_context["hostedZoneName"],
        cognito_domain_prefix=app.node.try_get_context("cognitoDomainPrefix"),
        env=env,
    )
    # Explicit ordering: FleetServiceStack imports ControlPlaneStack's
    # IotAtsEndpoint export, so it must deploy after (and destroy before) it.
    fleet_service_stack.add_dependency(control_plane_stack)
elif any(_fleet_context.values()):
    missing = [f"-c {key}=..." for key, value in _fleet_context.items() if not value]
    print(
        f"error: {', '.join(missing)} must be set to deploy FleetServiceStack "
        "(see fleet/infra/fleet-service.md) -- some but not all context "
        "values were set, which looks like a mistake rather than an "
        "intentional ControlPlaneStack-only run",
        file=sys.stderr,
    )
    raise SystemExit(1)
else:
    print(
        "-c domainName / -c hostedZoneName not set "
        "-- skipping FleetServiceStack (only ControlPlaneStack will synth)",
        file=sys.stderr,
    )

app.synth()
