#!/usr/bin/env bash
# Destroy FleetServiceStack via CDK. Requires an activated fleet/infra venv
# (see ./scripts/setup-venv.sh). Puts project-local Node + cdk CLI on PATH.
#
# Requires the same -c context as deploy-fleet-service.sh (domainName,
# hostedZoneName, githubOwner, githubRepo, githubBranch,
# githubOidcProviderArn, cdkBootstrapQualifier) -- CDK has to construct the
# stack to destroy it (see fleet/infra/fleet-service.md).
#
# Does NOT pass --force -- cdk will prompt for confirmation. Pass --force
# yourself for non-interactive/CI use: ./scripts/destroy-fleet-service.sh --force
#
# The Cognito user pool has RemovalPolicy.RETAIN, so destroy doesn't fail
# here but leaves the pool (and every operator account) behind, orphaned
# from the stack. See the "Remove the stack" section in
# fleet/infra/fleet-service.md for manual cleanup steps, then re-run.

set -euo pipefail

INFRA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$INFRA_DIR"

# Mirrors the Makefile pattern in krabby-research (VIRTUAL_ENV required).
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  cat >&2 <<'EOF'
No Python virtual environment found (VIRTUAL_ENV is unset).

From fleet/infra:
  ./scripts/setup-venv.sh
  source .venv/bin/activate

Then re-run this script.
EOF
  exit 1
fi

NODE_HOME="$INFRA_DIR/.tools/node"
NPM_GLOBAL="$INFRA_DIR/.tools/npm-global"
export PATH="$NODE_HOME/bin:$NPM_GLOBAL/node_modules/.bin:$PATH"

if [[ ! -x "$NODE_HOME/bin/node" ]] || ! command -v cdk >/dev/null 2>&1; then
  echo "Node/CDK toolkit missing under .tools/. Run: ./scripts/setup-venv.sh" >&2
  exit 1
fi

# Prove the active interpreter is the fleet infra venv (not a random host python).
EXPECTED_VENV="$INFRA_DIR/.venv"
if [[ "$(cd "$VIRTUAL_ENV" && pwd)" != "$(cd "$EXPECTED_VENV" && pwd)" ]]; then
  echo "VIRTUAL_ENV is '$VIRTUAL_ENV' but this script expects '$EXPECTED_VENV'." >&2
  echo "source $EXPECTED_VENV/bin/activate" >&2
  exit 1
fi

echo "Using python: $(command -v python) ($(python -V 2>&1))"
echo "Using node:   $(command -v node) ($(node -v))"
echo "Using cdk:    $(command -v cdk) ($(cdk --version))"

if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI ('aws') not found on PATH. Install/configure AWS CLI v2 first." >&2
  exit 1
fi

echo "Using aws:    $(command -v aws) ($(aws --version 2>&1))"

# Fail fast if credentials/session are missing or expired.
export AWS_IDENTITY_JSON
if ! AWS_IDENTITY_JSON="$(aws sts get-caller-identity --output json 2>/dev/null)"; then
  cat >&2 <<'EOF'
AWS credentials are not working (sts:GetCallerIdentity failed).

Fix by ensuring you are logged in (e.g. AWS SSO) or have valid env/credentials:
  - env vars: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
  - shared credentials: ~/.aws/credentials
  - SSO: aws sso login --profile <name>
EOF
  exit 1
fi

AWS_ACCOUNT_ID="$(python - <<'PY'
import json, os
print(json.loads(os.environ["AWS_IDENTITY_JSON"])["Account"])
PY
)"
AWS_ARN="$(python - <<'PY'
import json, os
print(json.loads(os.environ["AWS_IDENTITY_JSON"])["Arn"])
PY
)"
# Last path segment of the ARN: the IAM username, or the SSO session name
# (often an email) for an assumed-role session -- the human-readable
# "who is this" part of the identity, for the confirmation prompt below.
AWS_USER="${AWS_ARN##*/}"

AWS_REGION_EFFECTIVE="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
if [[ -z "$AWS_REGION_EFFECTIVE" ]]; then
  AWS_REGION_EFFECTIVE="$(aws configure get region 2>/dev/null || true)"
fi
if [[ -z "$AWS_REGION_EFFECTIVE" ]]; then
  echo "AWS region is not set. Set AWS_REGION/AWS_DEFAULT_REGION or configure a default region." >&2
  exit 1
fi

echo "Logged in as: $AWS_USER"
echo "AWS account:  $AWS_ACCOUNT_ID"
echo "AWS region:   $AWS_REGION_EFFECTIVE"
read -r -p "Destroy FleetServiceStack in this account? [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "Aborted." >&2
  exit 1
fi

echo "Destroying FleetServiceStack ..."

cdk destroy FleetServiceStack "$@"
