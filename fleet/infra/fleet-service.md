# `FleetServiceStack`

EC2 host, networking, and Cognito user pool for the fleet service. Defined
in `fleet_service_stack.py`. Depends on `ControlPlaneStack` (imports its
`IotAtsEndpoint` export).

## Required context values

These are required CDK context, passed as `-c key=value` on the command
line rather than environment variables or hardcoded constants.

| Context key | Placeholder | Purpose |
|---|---|---|
| `domainName` | `{domain-name}` | Fully-qualified DNS name for the fleet host |
| `hostedZoneName` | `{hosted-zone-name}` | The Route 53 hosted zone that owns the domain above -- its ID is resolved via a live AWS lookup at synth time, not supplied directly |
| `cognitoDomainPrefix` (optional) | `{cognito-domain-prefix}` | Cognito Hosted UI domain prefix — must be globally unique across AWS; defaults to `krabby-fleet-<account-id>` if unset |
| `githubOwner` | `{github-owner}` | GitHub org/user for Actions OIDC trust (`krabby-fleet-ci`) |
| `githubRepo` | `{github-repo}` | GitHub repo name for Actions OIDC trust |
| `githubBranch` | `{github-branch}` | Branch whose pushes may assume the role (also trusts same-repo `pull_request`) |
| `githubOidcProviderArn` | `{github-oidc-provider-arn}` | Existing account GitHub OIDC IdP ARN to import (this stack does **not** create a second IdP — one per issuer URL per account) |
| `cdkBootstrapQualifier` | `{cdk-bootstrap-qualifier}` | Qualifier used when granting `sts:AssumeRole` on CDK bootstrap roles |

## Deploy

```
cd fleet/infra
source .venv/bin/activate
./scripts/deploy-fleet-service.sh \
  -c domainName={domain-name} \
  -c hostedZoneName={hosted-zone-name} \
  -c githubOwner={github-owner} \
  -c githubRepo={github-repo} \
  -c githubBranch={github-branch} \
  -c githubOidcProviderArn={github-oidc-provider-arn} \
  -c cdkBootstrapQualifier={cdk-bootstrap-qualifier}
```

Requires `ControlPlaneStack` to already be deployed (its `IotAtsEndpoint`
export must exist). See [README.md](README.md) for the shared deploy-script
behavior (credential checks, identity confirmation prompt).

After `cdk deploy` finishes, the script calls
[`scripts/push-fleet-apps.sh`](scripts/push-fleet-apps.sh), which pushes **both**
`fleet/service` and `fleet/portal` onto the instance via SSM
`AWS-RunShellScript` and restarts `krabby-fleet-service`,
`krabby-fleet-portal`, `krabby-coturn`, and `caddy` — there's no SSH access to
this box (see `FleetServiceSecurityGroup` below), so this replaces what would
otherwise be an `scp` + remote install step. `fleet-deploy.yml` uses the same
push script after a combined `cdk deploy ControlPlaneStack FleetServiceStack`.

On the instance the portal zip is `npm ci` + `npm run build`'d into a Next.js
standalone tree under `/opt/krabby-fleet-portal`, and
`/etc/krabby-fleet/portal.env` is written from Cognito stack outputs + the
`/krabby/fleet/portal-auth-secret` Secrets Manager value. coturn gets
`/etc/krabby-fleet/turnserver.conf` + `/etc/krabby-fleet/service.env` from
`/krabby/fleet/turn-auth-secret` and the stack's domain/EIP.

The instance itself only gets OS-level bootstrap (Python, Node, coturn, the
`caddy` binary, system users, directories) from CDK UserData at first boot; an
app-only change pushes onto the existing instance rather than replacing it.

This stack looks up the account's default VPC (`ec2.Vpc.from_lookup`) and the
hosted zone's ID (`route53.HostedZone.from_lookup`), both of which need real
AWS credentials to resolve at synth time.

## Resources

| Resource | Type | Purpose |
|---|---|---|
| `FleetServiceSecurityGroup` | `AWS::EC2::SecurityGroup` | Inbound HTTP/HTTPS (80/443), STUN/TURN (3478 UDP+TCP, 5349 TCP/UDP), TURN relay (49152–65535/udp). No inbound SSH. |
| `FleetServiceInstanceRole` | `AWS::IAM::Role` | EC2 instance role: `AmazonSSMManagedInstanceCore` (Session Manager / Run Command, no SSH key needed) + Secure Tunneling `OpenTunnel`/`CloseTunnel`/`DescribeTunnel` + fleet listing (`iot:SearchIndex`/`GetThingShadow`/`DescribeThing`) + teleop signaling bridge (`iot:Connect`/`Publish`/`Subscribe`/`Receive` on `teleop/*/signaling/*`). |
| `FleetServiceInstance` | `AWS::EC2::Instance` | `c7i.large` (see rationale in `fleet_service_stack.py`), Amazon Linux 2023, IMDSv2 required, 30 GiB encrypted gp3 root volume, no auto-assigned public IP (uses the EIP below instead). UserData installs Python, Node, `spal-release` + `coturn` (SPAL), the `caddy` binary, and the `caddy`/`krabby-fleet` system users on first boot only -- app code and config are pushed separately (see Deploy above). |
| `FleetServiceAssetS3BucketName` / `FleetServiceAssetS3ObjectKey` (outputs) | `CfnOutput` | Location of the `fleet/service` zip CDK uploads to the bootstrap bucket on every deploy; read by `deploy-fleet-service.sh` to push it onto the instance via SSM. |
| `FleetPortalAssetS3BucketName` / `FleetPortalAssetS3ObjectKey` (outputs) | `CfnOutput` | Location of the `fleet/portal` source zip; built to Next.js standalone on the instance during SSM deploy. |
| `PortalAuthSecret` | `AWS::SecretsManager::Secret` (`/krabby/fleet/portal-auth-secret`) | Stable `AUTH_SECRET` for Auth.js; written into `/etc/krabby-fleet/portal.env` on each deploy. |
| `FleetPortalAuthSecretArn` (output) | `CfnOutput` | ARN of the portal auth secret above. |
| `TurnAuthSecret` | `AWS::SecretsManager::Secret` (`/krabby/fleet/turn-auth-secret`) | coturn TURN REST API `static-auth-secret`; written into `/etc/krabby-fleet/turnserver.conf` + `service.env` on each deploy. |
| `FleetTurnAuthSecretArn` (output) | `CfnOutput` | ARN of the TURN auth secret above. |
| `FleetServiceEip` / `FleetServiceEipAssociation` | `AWS::EC2::EIP` / `AWS::EC2::EIPAssociation` | Static public IP so the DNS record survives instance replacement. |
| `FleetServiceDnsRecord` | `AWS::Route53::RecordSet` | A record for `domainName` in the given hosted zone, pointed at the EIP. |
| `IotAtsEndpointParam` | `AWS::SSM::Parameter` (`/krabby/fleet/iot-ats-endpoint`) | `ControlPlaneStack`'s `IotAtsEndpoint` export, handed off via SSM Parameter Store for the fleet service to read at runtime. Instance role has read access. |
| `FleetOperatorPool` | `AWS::Cognito::UserPool` | Operator accounts (`user_pool_name=krabby-fleet-operators`). No self-sign-up (admin-created only). `RemovalPolicy.RETAIN`. |
| `OperatorGroup` | `AWS::Cognito::UserPoolGroup` (`operator`) | Membership grants tunnel/telemetry access; assigned manually by an admin, not by this stack. See [`OPERATORS.md`](../OPERATORS.md) (CLI + Console). |
| `FleetPoolDomain` | Cognito Hosted UI domain | `cognitoDomainPrefix` (or the account-scoped default). |
| `FleetOperatorClient` | `AWS::Cognito::UserPoolClient` | App client name `krabby-fleet`. Supports `USER_SRP_AUTH` and OAuth authorization-code + PKCE (no client secret). |
| `CognitoUserPoolIdParam` / `CognitoAppClientIdParam` | `AWS::SSM::Parameter` (`/krabby/fleet/cognito-user-pool-id`, `/krabby/fleet/cognito-app-client-id`) | The user pool ID and app client ID above, published to SSM Parameter Store so `krabby-fleet-service`'s JWT middleware can read them at runtime and know which pool/client to validate tokens against. Instance role has read access. |
| `FleetGitHubActionsRole` | `AWS::IAM::Role` (`krabby-fleet-ci`) | GitHub Actions OIDC role for `fleet-ci.yml` / `fleet-deploy.yml` / `fleet-ci-rotate-cognito.yml`. Trusts the configured `githubOwner`/`githubRepo` branch pushes and same-repo `pull_request` subjects (not forks); imports the account's existing GitHub OIDC provider ARN from context. Permissions: assume CDK bootstrap roles, `cloudformation:DescribeStacks` + SSM Run Command for post-deploy, Cognito admin on this pool (incl. CI password rotation), authenticated pull from ECR Public (localproxy image), IoT/Secure Tunneling/teleop MQTT for bench E2E. |
| `FleetServiceInstanceId`, `FleetServicePublicIp`, `FleetServiceDomainName`, `FleetCognitoUserPoolId`, `FleetCognitoUserPoolClientId`, `FleetCognitoDomain`, `FleetGitHubActionsRoleArn` (outputs) | `CfnOutput` | Console visibility. Cognito IDs and `FleetGitHubActionsRoleArn` are also exported for cross-stack / workflow use. |

## Remove the stack

```
cd fleet/infra
source .venv/bin/activate
./scripts/destroy-fleet-service.sh \
  -c domainName={domain-name} \
  -c hostedZoneName={hosted-zone-name} \
  -c githubOwner={github-owner} \
  -c githubRepo={github-repo} \
  -c githubBranch={github-branch} \
  -c githubOidcProviderArn={github-oidc-provider-arn} \
  -c cdkBootstrapQualifier={cdk-bootstrap-qualifier}
```

(CDK has to construct the stack to destroy it, so the same context values
used to deploy are required here too.)

### Manual steps that may block destroy or need cleanup after

- **`FleetOperatorPool` isn't deleted.** `RemovalPolicy.RETAIN` means destroy
  doesn't fail here, but the user pool (and every operator account) is left
  behind, orphaned from the stack. Delete manually via the Cognito console or
  `aws cognito-idp delete-user-pool` if you want a full cleanup.
- **`ControlPlaneStack` destroy ordering.** This stack imports
  `ControlPlaneStack`'s `IotAtsEndpoint` export, so `ControlPlaneStack` can't
  be destroyed while this stack still exists — destroy `FleetServiceStack`
  first (the explicit `add_dependency` in `app.py` also makes
  `cdk deploy ControlPlaneStack FleetServiceStack` order correctly on
  deploy).

If `cdk destroy` fails partway through, resolve the blocker and re-run —
CloudFormation resumes the rollback from where it stopped.
