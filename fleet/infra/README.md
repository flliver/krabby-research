# fleet/infra

CDK app for Krabby's AWS fleet infrastructure. Everything goes through
`cdk deploy` / `cdk destroy` via the scripts below -- no console click-ops.

## Setup (before deploying)

Run from `krabby-research/fleet/infra` -- `setup-venv.sh` creates `.venv/`
relative to the current directory, so running it from elsewhere activates
the wrong venv (or none).

```
cd fleet/infra
./scripts/setup-venv.sh
source .venv/bin/activate
```

Creates `.venv/` (Python CDK deps) and downloads a project-local Node +
`aws-cdk` CLI into `.tools/`. No system Node/npm required.

## AWS credentials (deploy / destroy)

The deploy/destroy scripts check `aws sts get-caller-identity` but never
create credentials. Authenticate as an IAM user or role that can run CDK in
this account. That is separate from the `krabby-enroll` user (Orin enroll
only — see [Enroll user access key](#enroll-user-access-key) below).

Use a short-lived access key **exported in this shell only** so closing the
terminal drops the creds (do not use `aws login` / `aws configure` here;
those cache on disk and survive a new shell):

1. IAM Console → Users → **your deploy user** → Security credentials → Create
   access key.
2. Export it for this shell session only:
   `export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=...`
   Optional: `export AWS_PAGER=""` so CLI tables do not open `less`.

Nothing was written to disk; closing the terminal clears these creds.

## Enroll user access key

`ControlPlaneStack` creates IAM user [`krabby-enroll`](control-plane.md) with
least-privilege enroll permissions. CDK does **not** create an access key. After
control plane deploy, someone with deploy IAM permissions creates an access key
**for IAM user `krabby-enroll`** (not for their own deploy user). That is often
a different person from the operator who enrolls devices on the Orin — store the
key in your team's usual secret store and distribute it only to people who need
to run enroll.

On the Orin, the operator exports that key in the enroll shell only, runs
[`krabby enroll`](../ENROLL.md), then discards the shell — enroll never persists
AWS secrets on the device.

## Deploy / destroy scripts

Every `deploy-*.sh` and `destroy-*.sh` script under `scripts/` follows the
same pattern: refuse to run unless the fleet infra venv is active and
Node/cdk exist under `.tools/`; fail fast if `aws sts get-caller-identity`
doesn't succeed; print the logged-in user, account, and region; then
prompt for a `y` confirmation before touching AWS -- there's no env var
that can silently redirect a deploy to the wrong account.

Destroy scripts additionally don't pass `--force` to `cdk destroy` itself,
so CDK prompts a *second* time for its own confirmation. Pass `--force`
yourself to skip that second prompt for non-interactive/CI use.

If deploy fails with `SSM parameter /cdk-bootstrap/.../version not found`,
bootstrap once (project-local `cdk` is under `.tools/`, not on the system
PATH):

```bash
export PATH="$PWD/.tools/node/bin:$PWD/.tools/npm-global/node_modules/.bin:$PATH"
cdk bootstrap aws://<account-id>/<region>
```

## Stacks

Each stack has its own doc with its resource table and destroy blockers.

Operator setup: [SETUP-FLEET.md](../SETUP-FLEET.md). Enroll:
[ENROLL.md](../ENROLL.md). One-source SSH: [SSH-TUNNEL.md](../SSH-TUNNEL.md).

| Stack | Docs | Deploy | Destroy |
|---|---|---|---|
| `ControlPlaneStack` | [control-plane.md](control-plane.md) | `./scripts/deploy-control-plane.sh` | `./scripts/destroy-control-plane.sh` |
| `FleetServiceStack` | [fleet-service.md](fleet-service.md) | `./scripts/deploy-fleet-service.sh` | `./scripts/destroy-fleet-service.sh` |

`FleetServiceStack` depends on `ControlPlaneStack` (imports its
`IotAtsEndpoint` export) — deploy `ControlPlaneStack` first, or run
`cdk deploy ControlPlaneStack FleetServiceStack` and let CDK resolve the
order.

If `cdk destroy` fails partway through, resolve the blocker (see the
stack's own doc) and re-run -- CloudFormation resumes the rollback from
where it stopped.
