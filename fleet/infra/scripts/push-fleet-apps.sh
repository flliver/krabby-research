#!/usr/bin/env bash
# Push fleet/service + fleet/portal CDK assets from S3 onto the fleet EC2 via
# SSM Run Command, then restart krabby-fleet-service / krabby-fleet-portal /
# krabby-coturn / caddy. Used by deploy-fleet-service.sh and fleet-deploy.yml
# after `cdk deploy` uploads the Asset zips.

set -euo pipefail

INFRA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$INFRA_DIR"

if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI ('aws') not found on PATH." >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  echo "python/python3 not found on PATH." >&2
  exit 1
fi
PYTHON="$(command -v python 2>/dev/null || command -v python3)"

AWS_REGION_EFFECTIVE="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
if [[ -z "$AWS_REGION_EFFECTIVE" ]]; then
  AWS_REGION_EFFECTIVE="$(aws configure get region 2>/dev/null || true)"
fi
if [[ -z "$AWS_REGION_EFFECTIVE" ]]; then
  echo "AWS region is not set. Set AWS_REGION/AWS_DEFAULT_REGION or configure a default region." >&2
  exit 1
fi

echo "Pushing fleet service + portal onto the instance ..."

export STACK_OUTPUTS_JSON
STACK_OUTPUTS_JSON="$(aws cloudformation describe-stacks \
  --stack-name FleetServiceStack \
  --region "$AWS_REGION_EFFECTIVE" \
  --query "Stacks[0].Outputs" --output json)"

_stack_output() {
  STACK_OUTPUT_KEY="$1" "$PYTHON" - <<'PY'
import json, os, sys

key = os.environ["STACK_OUTPUT_KEY"]
outputs = json.loads(os.environ["STACK_OUTPUTS_JSON"])
for o in outputs:
    if o["OutputKey"] == key:
        print(o["OutputValue"])
        break
else:
    print(f"error: output {key!r} not found in FleetServiceStack", file=sys.stderr)
    sys.exit(1)
PY
}

INSTANCE_ID="$(_stack_output FleetServiceInstanceId)"
SERVICE_ASSET_BUCKET="$(_stack_output FleetServiceAssetS3BucketName)"
SERVICE_ASSET_KEY="$(_stack_output FleetServiceAssetS3ObjectKey)"
PORTAL_ASSET_BUCKET="$(_stack_output FleetPortalAssetS3BucketName)"
PORTAL_ASSET_KEY="$(_stack_output FleetPortalAssetS3ObjectKey)"
PORTAL_AUTH_SECRET_ARN="$(_stack_output FleetPortalAuthSecretArn)"
TURN_AUTH_SECRET_ARN="$(_stack_output FleetTurnAuthSecretArn)"
DOMAIN_NAME="$(_stack_output FleetServiceDomainName)"
PUBLIC_IP="$(_stack_output FleetServicePublicIp)"
COGNITO_POOL_ID="$(_stack_output FleetCognitoUserPoolId)"
COGNITO_CLIENT_ID="$(_stack_output FleetCognitoUserPoolClientId)"

# First boot: UserData installs packages + SSM agent registers. send-command
# fails with InvalidInstanceId until the instance is Online in SSM.
echo "Waiting for $INSTANCE_ID to appear Online in SSM ..."
SSM_WAIT_SECONDS=0
SSM_WAIT_LIMIT=600
while true; do
  PING_STATUS="$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
    --region "$AWS_REGION_EFFECTIVE" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || true)"
  if [[ "$PING_STATUS" == "Online" ]]; then
    echo "SSM Online after ${SSM_WAIT_SECONDS}s."
    break
  fi
  if (( SSM_WAIT_SECONDS >= SSM_WAIT_LIMIT )); then
    echo "Timed out after ${SSM_WAIT_LIMIT}s waiting for SSM Online (last status: ${PING_STATUS:-none})." >&2
    echo "Re-run this script once the instance is Online in Systems Manager." >&2
    exit 1
  fi
  sleep 15
  SSM_WAIT_SECONDS=$((SSM_WAIT_SECONDS + 15))
  echo "  ... not Online yet (${PING_STATUS:-none}), ${SSM_WAIT_SECONDS}s elapsed"
done

export REMOTE_SCRIPT
REMOTE_SCRIPT="$(cat <<REMOTE
set -euo pipefail

# coturn may already be present from UserData; install on older instances too.
# Package lives in AL2023 SPAL (Supplementary Packages), not the base repos.
dnf install -y spal-release
dnf install -y coturn
getent group turnserver >/dev/null || groupadd --system turnserver
id -u turnserver >/dev/null 2>&1 || \
  useradd --system --no-create-home --shell /usr/sbin/nologin -g turnserver turnserver
systemctl disable --now coturn 2>/dev/null || true
systemctl disable --now turnserver 2>/dev/null || true

# --- fleet service ---
aws s3 cp "s3://${SERVICE_ASSET_BUCKET}/${SERVICE_ASSET_KEY}" /tmp/fleet-service.zip
rm -rf /opt/krabby-fleet-service/src
mkdir -p /opt/krabby-fleet-service/src
unzip -o -q /tmp/fleet-service.zip -d /opt/krabby-fleet-service/src
install -m 0644 /opt/krabby-fleet-service/src/deploy/Caddyfile /etc/caddy/Caddyfile
install -m 0644 /opt/krabby-fleet-service/src/deploy/caddy.service /etc/systemd/system/caddy.service
mkdir -p /var/lib/caddy
chown caddy:caddy /var/lib/caddy
install -m 0644 /opt/krabby-fleet-service/src/systemd/krabby-fleet-service.service /etc/systemd/system/krabby-fleet-service.service
install -m 0644 /opt/krabby-fleet-service/src/deploy/coturn.service /etc/systemd/system/krabby-coturn.service
/usr/bin/pip3.11 install --quiet --upgrade /opt/krabby-fleet-service/src

TURN_AUTH_SECRET_VALUE="\$(aws secretsmanager get-secret-value \\
  --secret-id '${TURN_AUTH_SECRET_ARN}' \\
  --region '${AWS_REGION_EFFECTIVE}' \\
  --query SecretString --output text)"

# Render coturn conf (shared HMAC secret + public EIP for relay candidates).
umask 027
sed -e "s|__TURN_AUTH_SECRET__|\${TURN_AUTH_SECRET_VALUE}|g" \\
    -e "s|__TURN_REALM__|${DOMAIN_NAME}|g" \\
    -e "s|__TURN_EXTERNAL_IP__|${PUBLIC_IP}|g" \\
    /opt/krabby-fleet-service/src/deploy/turnserver.conf.in \\
    > /etc/krabby-fleet/turnserver.conf
chmod 0640 /etc/krabby-fleet/turnserver.conf
# turnserver package user needs read access to the conf.
if id turnserver >/dev/null 2>&1; then
  chown root:turnserver /etc/krabby-fleet/turnserver.conf
else
  chown root:root /etc/krabby-fleet/turnserver.conf
fi

# Fleet service env: region (SSM/Cognito lookups), Cognito IDs, TURN.
# Without AWS_REGION the app defaults to us-east-1 and GetParameter misses
# /krabby/fleet/* params that live in the stack region.
cat > /etc/krabby-fleet/service.env <<ENV
AWS_REGION=${AWS_REGION_EFFECTIVE}
AWS_DEFAULT_REGION=${AWS_REGION_EFFECTIVE}
KRABBY_FLEET_COGNITO_USER_POOL_ID=${COGNITO_POOL_ID}
KRABBY_FLEET_COGNITO_APP_CLIENT_ID=${COGNITO_CLIENT_ID}
KRABBY_FLEET_TURN_AUTH_SECRET=\${TURN_AUTH_SECRET_VALUE}
KRABBY_FLEET_TURN_HOST=${DOMAIN_NAME}
ENV
chown root:krabby-fleet /etc/krabby-fleet/service.env
chmod 0640 /etc/krabby-fleet/service.env

# --- portal (Next.js standalone) ---
aws s3 cp "s3://${PORTAL_ASSET_BUCKET}/${PORTAL_ASSET_KEY}" /tmp/fleet-portal.zip
rm -rf /opt/krabby-fleet-portal-src
mkdir -p /opt/krabby-fleet-portal-src
unzip -o -q /tmp/fleet-portal.zip -d /opt/krabby-fleet-portal-src
cd /opt/krabby-fleet-portal-src
# Build-time placeholders; runtime values come from /etc/krabby-fleet/portal.env.
# DevDependencies (typescript) are required for \`next build\`.
export AUTH_SECRET=build-placeholder
export AUTH_URL="https://${DOMAIN_NAME}"
export AUTH_COGNITO_ID="${COGNITO_CLIENT_ID}"
export AUTH_COGNITO_ISSUER="https://cognito-idp.${AWS_REGION_EFFECTIVE}.amazonaws.com/${COGNITO_POOL_ID}"
export FLEET_SERVICE_URL="http://127.0.0.1:8080"
/usr/local/bin/npm ci
/usr/local/bin/npm run build
bash scripts/assemble-standalone.sh /opt/krabby-fleet-portal-src /opt/krabby-fleet-portal
install -m 0644 /opt/krabby-fleet-portal-src/systemd/krabby-fleet-portal.service \\
  /etc/systemd/system/krabby-fleet-portal.service
chown -R krabby-fleet:krabby-fleet /opt/krabby-fleet-portal

AUTH_SECRET_VALUE="\$(aws secretsmanager get-secret-value \\
  --secret-id '${PORTAL_AUTH_SECRET_ARN}' \\
  --region '${AWS_REGION_EFFECTIVE}' \\
  --query SecretString --output text)"
umask 027
cat > /etc/krabby-fleet/portal.env <<ENV
AUTH_SECRET=\${AUTH_SECRET_VALUE}
AUTH_URL=https://${DOMAIN_NAME}
AUTH_COGNITO_ID=${COGNITO_CLIENT_ID}
AUTH_COGNITO_ISSUER=https://cognito-idp.${AWS_REGION_EFFECTIVE}.amazonaws.com/${COGNITO_POOL_ID}
FLEET_SERVICE_URL=http://127.0.0.1:8080
ENV
chown root:krabby-fleet /etc/krabby-fleet/portal.env
chmod 0640 /etc/krabby-fleet/portal.env

systemctl daemon-reload
systemctl enable caddy krabby-fleet-service krabby-fleet-portal krabby-coturn
systemctl restart krabby-coturn krabby-fleet-service krabby-fleet-portal caddy
REMOTE
)"

# JSON built with python (not CLI shorthand) so embedded newlines/quotes in
# the multi-line script above can't be misparsed by --parameters shorthand.
# executionTimeout covers npm ci + next build on a cold instance.
export SSM_PARAMS_JSON
SSM_PARAMS_JSON="$("$PYTHON" - <<'PY'
import json, os
print(json.dumps({
    "commands": [os.environ["REMOTE_SCRIPT"]],
    "executionTimeout": ["3600"],
}))
PY
)"

COMMAND_ID="$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "krabby-fleet service+portal+coturn app deploy" \
  --region "$AWS_REGION_EFFECTIVE" \
  --parameters "$SSM_PARAMS_JSON" \
  --query "Command.CommandId" --output text)"

echo "Waiting for SSM command $COMMAND_ID to finish on $INSTANCE_ID ..."
aws ssm wait command-executed \
  --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" --region "$AWS_REGION_EFFECTIVE" || true

SSM_STATUS="$(aws ssm get-command-invocation \
  --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" --region "$AWS_REGION_EFFECTIVE" \
  --query "Status" --output text)"

if [[ "$SSM_STATUS" != "Success" ]]; then
  echo "App deploy failed on the instance (status: $SSM_STATUS):" >&2
  aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" --region "$AWS_REGION_EFFECTIVE" \
    --query "StandardErrorContent" --output text >&2
  exit 1
fi

echo "[ok] krabby-fleet-service + krabby-fleet-portal + krabby-coturn deployed and restarted."
echo "     Caddy: /api/auth* + UI -> portal:3000; other /api/* -> service:8080"
echo "     TURN:  UDP/TCP 3478 + relay 49152-65535/udp (ICE via GET /api/teleop/ice-servers)"
