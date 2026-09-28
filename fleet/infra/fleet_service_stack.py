"""FleetServiceStack -- EC2 host, networking, and Cognito for the fleet service.

Provisions the EC2 instance, its Elastic IP, a Route 53 A record, a security
group, the instance's IAM role, the Cognito user pool operators authenticate
against, and the GitHub Actions OIDC role used by fleet-ci / fleet-deploy.
Depends on `ControlPlaneStack` via the `IotAtsEndpoint` CFN export.

The instance IAM role grants Secure Tunneling open/close, fleet listing
(SearchIndex/GetThingShadow/DescribeThing), and the teleop signaling
bridge's MQTT connect/pub/sub -- a fully specified permission shape, granted
in full here.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from aws_cdk import CfnOutput, Duration, Fn, RemovalPolicy, Stack
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_route53 as route53
from aws_cdk import aws_s3_assets as s3_assets
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_ssm as ssm
from constructs import Construct

# SSM Parameter Store paths the fleet service reads at runtime -- decouples
# this stack's "hand off config" duty from guessing the service's exact
# config-loading mechanism (env file, secrets manager, etc). krabby_fleet_service
# reads these same paths; keep the two in sync if either changes.
IOT_ATS_ENDPOINT_PARAM_NAME = "/krabby/fleet/iot-ats-endpoint"
COGNITO_USER_POOL_ID_PARAM_NAME = "/krabby/fleet/cognito-user-pool-id"
COGNITO_APP_CLIENT_ID_PARAM_NAME = "/krabby/fleet/cognito-app-client-id"

# Auth.js cookie-signing secret for the portal; written into
# /etc/krabby-fleet/portal.env by deploy-fleet-service.sh.
PORTAL_AUTH_SECRET_NAME = "/krabby/fleet/portal-auth-secret"

# Shared HMAC secret for coturn TURN REST API credentials; written into
# /etc/krabby-fleet/turnserver.conf + service.env by deploy-fleet-service.sh.
TURN_AUTH_SECRET_NAME = "/krabby/fleet/turn-auth-secret"

# Node binary installed by UserData (must match ExecStart in
# krabby-fleet-portal.service).
_NODE_VERSION = "22.14.0"

# GitHub Actions OIDC issuer host (fixed by GitHub; used in trust
# conditions). Repo owner/name/branch, IdP ARN, and CDK bootstrap
# qualifier are required CDK context -- no fallbacks (see fleet-service.md).
_GITHUB_OIDC_ISSUER = "token.actions.githubusercontent.com"


class FleetServiceStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        domain_name: str,
        hosted_zone_name: str,
        cognito_domain_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        domain_name: fully-qualified DNS name for the fleet host.
        hosted_zone_name: the Route 53 hosted zone that owns `domain_name`;
            its ID is resolved via a live AWS lookup at synth time (see
            `HostedZone.from_lookup` below), which needs real AWS credentials.
        cognito_domain_prefix: Hosted UI domain prefix; defaults to a
            per-account-unique value if omitted (Cognito domain prefixes must
            be globally unique across all AWS accounts).
        """
        super().__init__(scope, construct_id, **kwargs)

        # Single-host deployment: the account's default VPC is enough, no
        # custom networking needed for one EC2 instance in a public subnet.
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        security_group = ec2.SecurityGroup(
            self,
            "FleetServiceSecurityGroup",
            vpc=vpc,
            description=(
                "Krabby fleet service host: inbound HTTP/HTTPS (80/443) and "
                "STUN/TURN (3478 UDP, 5349 TCP/UDP). No inbound SSH -- admin "
                "access is via SSM Session Manager (instance role below), "
                "not an open port 22."
            ),
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(80),
            "HTTP -- ACME HTTP-01 challenge and HTTP-to-HTTPS redirect",
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "HTTPS",
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.udp(3478), "STUN/TURN (UDP)",
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(3478), "TURN (TCP)",
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(5349), "TURNS (TLS)",
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.udp(5349), "TURNS (DTLS)",
        )
        # coturn relay allocations (min-port/max-port in turnserver.conf).
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.udp_range(49152, 65535),
            "TURN relay media (UDP)",
        )

        # AmazonSSMManagedInstanceCore makes the box reachable via SSM Session
        # Manager / Run Command -- how both interactive admin access and the
        # automated deploy restart reach the instance, with no SSH key or
        # open port 22 anywhere in this stack.
        instance_role = iam.Role(
            self,
            "FleetServiceInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
            ],
        )
        instance_role.add_to_policy(
            iam.PolicyStatement(
                sid="SecureTunnelingOpenClose",
                # IAM action prefix is `iot:` (not `iotsecuretunneling:`) —
                # see https://docs.aws.amazon.com/iot/latest/developerguide/tunnel-access.html
                actions=[
                    "iot:OpenTunnel",
                    "iot:CloseTunnel",
                    "iot:DescribeTunnel",
                ],
                # OpenTunnel authorizes against tunnel/* (and optionally thing/*);
                # Resource: "*" matches the AWS-documented broad form.
                resources=["*"],
            )
        )
        instance_role.add_to_policy(
            iam.PolicyStatement(
                sid="FleetListing",
                actions=["iot:SearchIndex"],
                # Fleet Indexing queries the whole index, not one thing --
                # SearchIndex doesn't support resource-level permissions.
                resources=["*"],
            )
        )
        instance_role.add_to_policy(
            iam.PolicyStatement(
                sid="FleetDeviceRead",
                actions=["iot:GetThingShadow", "iot:DescribeThing"],
                resources=[f"arn:aws:iot:{self.region}:{self.account}:thing/*"],
            )
        )
        instance_role.add_to_policy(
            iam.PolicyStatement(
                sid="TeleopSignalingConnect",
                actions=["iot:Connect"],
                # The signaling bridge's own persistent MQTT connection; its
                # client ID isn't fixed by this stack, so scope to the
                # account/region rather than one specific client ARN.
                resources=[f"arn:aws:iot:{self.region}:{self.account}:client/*"],
            )
        )
        instance_role.add_to_policy(
            iam.PolicyStatement(
                sid="TeleopSignalingPublish",
                actions=["iot:Publish"],
                resources=[f"arn:aws:iot:{self.region}:{self.account}:topic/teleop/*/signaling/*"],
            )
        )
        instance_role.add_to_policy(
            iam.PolicyStatement(
                sid="TeleopSignalingSubscribe",
                actions=["iot:Subscribe"],
                resources=[f"arn:aws:iot:{self.region}:{self.account}:topicfilter/teleop/*/signaling/*"],
            )
        )
        instance_role.add_to_policy(
            iam.PolicyStatement(
                sid="TeleopSignalingReceive",
                actions=["iot:Receive"],
                resources=[f"arn:aws:iot:{self.region}:{self.account}:topic/teleop/*/signaling/*"],
            )
        )

        # One-time OS-level bootstrap: installs the packages, system users,
        # and directories that app deploys (pushed separately via SSM --
        # see deploy-fleet-service.sh) assume already exist. This only runs
        # on first boot, so it deliberately doesn't touch app code or
        # config content (Caddyfile, systemd units) -- those are pushed by
        # every deploy, not baked into the instance at creation.
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "set -euo pipefail",
            "dnf install -y python3.11 python3.11-pip unzip xz tar",
            # coturn is in AL2023 SPAL (not the base repos).
            "dnf install -y spal-release",
            "dnf install -y coturn",
            # No official Amazon Linux package for Caddy -- pull the static
            # binary directly from Caddy's own documented download API
            # rather than depending on a third-party yum repo's uptime.
            "curl -fsSL 'https://caddyserver.com/api/download?os=linux&arch=amd64' -o /usr/bin/caddy",
            "chmod 0755 /usr/bin/caddy",
            # Disable the distro coturn unit; deploy-fleet-service.sh installs
            # krabby-coturn.service with our conf (shared REST-API secret).
            "systemctl disable --now coturn 2>/dev/null || true",
            "systemctl disable --now turnserver 2>/dev/null || true",
            # Node for krabby-fleet-portal (Next.js standalone). Official
            # tarball under /usr/local so ExecStart can use /usr/local/bin/node.
            f"curl -fsSL 'https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-x64.tar.xz' "
            "-o /tmp/node.tar.xz",
            "tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1",
            "rm -f /tmp/node.tar.xz",
            "getent group caddy >/dev/null || groupadd --system caddy",
            "id -u caddy >/dev/null 2>&1 || "
            "useradd --system --no-create-home --shell /usr/sbin/nologin -g caddy caddy",
            "id -u krabby-fleet >/dev/null 2>&1 || "
            "useradd --system --no-create-home --shell /usr/sbin/nologin krabby-fleet",
            "mkdir -p /etc/caddy /etc/krabby-fleet /opt/krabby-fleet-service "
            "/opt/krabby-fleet-portal /opt/krabby-fleet-portal-src /var/lib/caddy",
            "chown caddy:caddy /var/lib/caddy",
            "chown root:krabby-fleet /etc/krabby-fleet && chmod 0750 /etc/krabby-fleet",
            # Provisioning-time config (this stack's domainName), not app
            # code -- baked in here rather than shipped as a repo artifact,
            # since it's specific to this deployment.
            f"echo 'FLEET_DOMAIN={domain_name}' > /etc/caddy/caddy.env",
        )

        instance = ec2.Instance(
            self,
            "FleetServiceInstance",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            # Non-burstable: this host also runs coturn under TURN relay,
            # which can hold sustained CPU/network load that would throttle a
            # burstable instance's credit balance mid-session. c7i.large is
            # 2 vCPU / 4 GiB, non-burstable Intel, up to 12.5 Gbps network,
            # x86_64 only.
            instance_type=ec2.InstanceType("c7i.large"),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            security_group=security_group,
            role=instance_role,
            require_imdsv2=True,
            user_data=user_data,
            # We assign a static Elastic IP explicitly below; skip the
            # subnet's auto-assigned ephemeral public IP so the instance
            # doesn't end up with two public addresses.
            associate_public_ip_address=False,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        30, volume_type=ec2.EbsDeviceVolumeType.GP3, encrypted=True,
                    ),
                )
            ],
        )

        # Packages fleet/service (app code + deploy/ + systemd/) as a zip,
        # uploaded to the CDK bootstrap bucket on every `cdk deploy`.
        # deploy-fleet-service.sh reads the resulting bucket/key outputs
        # below and pushes them onto the instance via SSM after this stack
        # deploys, so an app-only change doesn't require replacing the
        # instance.
        service_asset = s3_assets.Asset(
            self,
            "FleetServiceSourceAsset",
            path=os.path.join(os.path.dirname(__file__), "..", "service"),
            exclude=["__pycache__", "*.egg-info", ".venv", ".pytest_cache"],
        )
        service_asset.grant_read(instance_role)

        # Portal source (built on the instance during SSM deploy). Excludes
        # local node_modules/.next so the zip stays small; `npm ci` runs remotely.
        portal_asset = s3_assets.Asset(
            self,
            "FleetPortalSourceAsset",
            path=os.path.join(os.path.dirname(__file__), "..", "portal"),
            exclude=[
                "node_modules",
                ".next",
                ".venv",
                ".env",
                ".env.local",
                "*.tsbuildinfo",
            ],
        )
        portal_asset.grant_read(instance_role)

        # Stable Auth.js signing secret across portal redeploys.
        portal_auth_secret = secretsmanager.Secret(
            self,
            "PortalAuthSecret",
            secret_name=PORTAL_AUTH_SECRET_NAME,
            description="AUTH_SECRET for krabby-fleet-portal (Auth.js)",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=48,
                exclude_punctuation=True,
            ),
        )
        portal_auth_secret.grant_read(instance_role)

        # coturn TURN REST API shared secret (HMAC key). Fleet service mints
        # short-lived username/credential pairs; coturn verifies with the
        # same secret (use-auth-secret / static-auth-secret).
        turn_auth_secret = secretsmanager.Secret(
            self,
            "TurnAuthSecret",
            secret_name=TURN_AUTH_SECRET_NAME,
            description="static-auth-secret for coturn TURN REST API (krabby teleop ICE)",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=48,
                exclude_punctuation=True,
            ),
        )
        turn_auth_secret.grant_read(instance_role)

        # Static IP so the Route 53 record survives instance replacement
        # (redeploys, AZ failure recovery) without a DNS update.
        eip = ec2.CfnEIP(self, "FleetServiceEip", domain="vpc")
        ec2.CfnEIPAssociation(
            self, "FleetServiceEipAssociation",
            # `allocation_id`, not the deprecated `eip` (ref) prop --
            # AllocationId is the VPC-EIP association mechanism; `eip` is a
            # holdover from EC2-Classic.
            allocation_id=eip.attr_allocation_id, instance_id=instance.instance_id,
        )

        hosted_zone = route53.HostedZone.from_lookup(
            self, "FleetHostedZone", domain_name=hosted_zone_name,
        )
        route53.ARecord(
            self, "FleetServiceDnsRecord",
            zone=hosted_zone,
            record_name=domain_name,
            target=route53.RecordTarget.from_ip_addresses(eip.attr_public_ip),
            ttl=Duration.minutes(5),
        )

        # Cross-stack import from ControlPlaneStack. Handed to the not-yet-built
        # fleet service via an SSM Parameter rather than assumed env-file
        # plumbing, so this stack doesn't need to guess how that service will
        # eventually load its config.
        iot_ats_endpoint = Fn.import_value("IotAtsEndpoint")
        iot_ats_endpoint_param = ssm.StringParameter(
            self, "IotAtsEndpointParam",
            parameter_name=IOT_ATS_ENDPOINT_PARAM_NAME,
            string_value=iot_ats_endpoint,
            description="AWS IoT Core ATS endpoint (from ControlPlaneStack), read by krabby-fleet-service at runtime",
        )
        iot_ats_endpoint_param.grant_read(instance_role)

        # Operators authenticate here; the fleet service verifies the
        # resulting JWT rather than this stack minting per-user AWS
        # credentials. RemovalPolicy.RETAIN: destroying the stack shouldn't
        # silently delete every operator's account and lock the fleet out.
        # Stable user_pool_name so the Cognito console shows a readable name
        # (otherwise CFN generates FleetOperatorPool…-<hash>).
        user_pool = cognito.UserPool(
            self, "FleetOperatorPool",
            user_pool_name="krabby-fleet-operators",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True),
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Membership is granted by an admin after account creation, not by
        # this stack -- being in this group is what the fleet service checks
        # before opening a tunnel on a user's behalf.
        cognito.CfnUserPoolGroup(
            self, "OperatorGroup",
            user_pool_id=user_pool.user_pool_id,
            group_name="operator",
            description="Fleet operators allowed to open SSH tunnels and view telemetry",
        )

        # Cognito Hosted UI domain prefixes are unique across all of AWS, not
        # just this account -- default to an account-scoped value so this
        # doesn't collide with another AWS customer's stack.
        domain_prefix = cognito_domain_prefix or f"krabby-fleet-{self.account}"
        user_pool_domain = user_pool.add_domain(
            "FleetPoolDomain",
            cognito_domain=cognito.CognitoDomainOptions(domain_prefix=domain_prefix),
        )

        # Supports USER_SRP_AUTH (no browser) and OAuth authorization-code +
        # PKCE, no client secret needed since PKCE is the confidentiality
        # mechanism for a public client. The callback path follows the
        # NextAuth.js Cognito provider convention.
        user_pool_client = user_pool.add_client(
            "FleetOperatorClient",
            user_pool_client_name="krabby-fleet",
            generate_secret=False,
            auth_flows=cognito.AuthFlow(user_srp=True),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.PROFILE,
                ],
                callback_urls=[
                    f"https://{domain_name}/api/auth/callback/cognito",
                    "http://localhost:3000/api/auth/callback/cognito",
                ],
                logout_urls=[
                    f"https://{domain_name}",
                    "http://localhost:3000",
                ],
            ),
        )

        # Published to SSM Parameter Store so the JWT verification middleware
        # can read them at runtime and validate tokens against the right
        # user pool/client, without this stack guessing its config format.
        cognito_user_pool_id_param = ssm.StringParameter(
            self, "CognitoUserPoolIdParam",
            parameter_name=COGNITO_USER_POOL_ID_PARAM_NAME,
            string_value=user_pool.user_pool_id,
            description="Cognito user pool ID, read by krabby-fleet-service at runtime",
        )
        cognito_user_pool_id_param.grant_read(instance_role)
        cognito_app_client_id_param = ssm.StringParameter(
            self, "CognitoAppClientIdParam",
            parameter_name=COGNITO_APP_CLIENT_ID_PARAM_NAME,
            string_value=user_pool_client.user_pool_client_id,
            description="Cognito app client ID, read by krabby-fleet-service at runtime",
        )
        cognito_app_client_id_param.grant_read(instance_role)

        # --- GitHub Actions OIDC role (fleet-ci / fleet-deploy) ---
        # One IdP per issuer URL per account: import an existing GitHub
        # provider (e.g. firmware CI) rather than creating a second one that
        # would fail deploy with EntityAlreadyExists.
        _github_ctx_keys = (
            "githubOwner",
            "githubRepo",
            "githubBranch",
            "githubOidcProviderArn",
            "cdkBootstrapQualifier",
        )
        _github_ctx = {
            key: self.node.try_get_context(key) for key in _github_ctx_keys
        }
        _missing_github = [k for k, v in _github_ctx.items() if not v]
        if _missing_github:
            raise ValueError(
                "FleetServiceStack GitHub Actions OIDC requires CDK context "
                + ", ".join(f"-c {k}=..." for k in _missing_github)
                + " (see fleet/infra/fleet-service.md)"
            )
        github_owner = _github_ctx["githubOwner"]
        github_repo = _github_ctx["githubRepo"]
        github_branch = _github_ctx["githubBranch"]
        github_oidc_provider_arn = _github_ctx["githubOidcProviderArn"]
        cdk_bootstrap_qualifier = _github_ctx["cdkBootstrapQualifier"]
        github_repo_slug = f"{github_owner}/{github_repo}"

        github_oidc_provider = iam.OpenIdConnectProvider.from_open_id_connect_provider_arn(
            self, "GitHubOidcProvider", github_oidc_provider_arn,
        )
        # Trust: default-branch pushes + same-repo pull_request subjects.
        # Fork PRs do not match these `sub` values (fork sub uses the fork
        # owner/repo). repository claim is a second belt against confused deputy.
        github_actions_principal = iam.OpenIdConnectPrincipal(
            github_oidc_provider,
            conditions={
                "StringEquals": {
                    f"{_GITHUB_OIDC_ISSUER}:aud": "sts.amazonaws.com",
                    f"{_GITHUB_OIDC_ISSUER}:repository": github_repo_slug,
                },
                "StringLike": {
                    f"{_GITHUB_OIDC_ISSUER}:sub": [
                        f"repo:{github_repo_slug}:ref:refs/heads/{github_branch}",
                        f"repo:{github_repo_slug}:pull_request",
                    ],
                },
            },
        )
        github_actions_role = iam.Role(
            self,
            "FleetGitHubActionsRole",
            role_name="krabby-fleet-ci",
            description=(
                "GitHub Actions OIDC role for fleet-ci / fleet-deploy "
                f"({github_repo_slug})"
            ),
            assumed_by=github_actions_principal,
            max_session_duration=Duration.hours(1),
        )
        # CDK deploy of ControlPlaneStack + FleetServiceStack: assume the
        # account's bootstrap roles (file publish + deploy + lookup).
        github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="CdkBootstrapAssumeRoles",
                actions=["sts:AssumeRole"],
                resources=[
                    f"arn:aws:iam::{self.account}:role/cdk-{cdk_bootstrap_qualifier}-*",
                ],
            )
        )
        # Post-deploy: read stack outputs, wait for SSM Online, Run Command
        # restart (mirrors deploy-fleet-service.sh after `cdk deploy`).
        github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="FleetDeployStackOutputs",
                actions=["cloudformation:DescribeStacks"],
                resources=[
                    f"arn:aws:cloudformation:{self.region}:{self.account}:stack/ControlPlaneStack/*",
                    f"arn:aws:cloudformation:{self.region}:{self.account}:stack/FleetServiceStack/*",
                ],
            )
        )
        github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="FleetDeploySsmRunCommand",
                actions=[
                    "ssm:DescribeInstanceInformation",
                    "ssm:SendCommand",
                    "ssm:GetCommandInvocation",
                    "ssm:ListCommandInvocations",
                ],
                resources=["*"],
            )
        )
        # Bench E2E + CI Cognito user lifecycle (scratch users / rotation).
        github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="FleetCiCognitoAdmin",
                actions=[
                    "cognito-idp:AdminCreateUser",
                    "cognito-idp:AdminDeleteUser",
                    "cognito-idp:AdminSetUserPassword",
                    "cognito-idp:AdminAddUserToGroup",
                    "cognito-idp:AdminRemoveUserFromGroup",
                    "cognito-idp:AdminGetUser",
                    "cognito-idp:AdminResetUserPassword",
                    "cognito-idp:ListUsers",
                    "cognito-idp:DescribeUserPool",
                ],
                resources=[user_pool.user_pool_arn],
            )
        )
        # Bench E2E against IoT / tunneling / teleop MQTT (scratch thing+cert
        # isolation tests need create/delete).
        github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="FleetCiEcrPublicPull",
                actions=[
                    "ecr-public:GetAuthorizationToken",
                    "ecr-public:BatchCheckLayerAvailability",
                    "ecr-public:GetDownloadUrlForLayer",
                    "ecr-public:BatchGetImage",
                ],
                resources=["*"],
            )
        )
        # ECR Public clients (CLI + SDK) call sts:GetServiceBearerToken without passing
        # sts:AWSServiceName in a form that matches a StringEquals condition on the role
        # policy; unscoped Allow is required for bench CI docker login to succeed.
        github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="FleetCiStsEcrPublicBearerToken",
                actions=["sts:GetServiceBearerToken"],
                resources=["*"],
            )
        )
        github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="FleetCiIotBench",
                actions=[
                    "iot:GetPolicy",
                    "iot:DescribeThingType",
                    "iot:DescribeThing",
                    "iot:DescribeEndpoint",
                    "iot:SearchIndex",
                    "iot:GetThingShadow",
                    "iot:CreateThing",
                    "iot:DeleteThing",
                    "iot:CreateKeysAndCertificate",
                    "iot:UpdateCertificate",
                    "iot:DeleteCertificate",
                    "iot:AttachPolicy",
                    "iot:DetachPolicy",
                    "iot:AttachThingPrincipal",
                    "iot:DetachThingPrincipal",
                    "iot:OpenTunnel",
                    "iot:CloseTunnel",
                    "iot:DescribeTunnel",
                    "iot:Connect",
                    "iot:Publish",
                    "iot:Subscribe",
                    "iot:Receive",
                ],
                resources=["*"],
            )
        )

        CfnOutput(self, "FleetServiceInstanceId", value=instance.instance_id)
        CfnOutput(self, "FleetServicePublicIp", value=eip.attr_public_ip)
        CfnOutput(self, "FleetServiceDomainName", value=domain_name)
        # Read by deploy-fleet-service.sh to push this deploy's app code
        # onto the instance via SSM after `cdk deploy` finishes.
        CfnOutput(self, "FleetServiceAssetS3BucketName", value=service_asset.s3_bucket_name)
        CfnOutput(self, "FleetServiceAssetS3ObjectKey", value=service_asset.s3_object_key)
        CfnOutput(self, "FleetPortalAssetS3BucketName", value=portal_asset.s3_bucket_name)
        CfnOutput(self, "FleetPortalAssetS3ObjectKey", value=portal_asset.s3_object_key)
        CfnOutput(self, "FleetPortalAuthSecretArn", value=portal_auth_secret.secret_arn)
        CfnOutput(self, "FleetTurnAuthSecretArn", value=turn_auth_secret.secret_arn)
        CfnOutput(
            self, "FleetCognitoUserPoolId",
            value=user_pool.user_pool_id, export_name="FleetCognitoUserPoolId",
        )
        CfnOutput(
            self, "FleetCognitoUserPoolClientId",
            value=user_pool_client.user_pool_client_id, export_name="FleetCognitoUserPoolClientId",
        )
        CfnOutput(self, "FleetCognitoDomain", value=user_pool_domain.domain_name)
        CfnOutput(
            self, "FleetGitHubActionsRoleArn",
            value=github_actions_role.role_arn,
            export_name="FleetGitHubActionsRoleArn",
            description="IAM role ARN for fleet-ci / fleet-deploy (GitHub OIDC)",
        )
