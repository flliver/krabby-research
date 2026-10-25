# shellcheck shell=bash
# Load CDK -c flags from fleet/config/fleet.toml (same values CI exports as FLEET_CDK_*).
# Source from deploy/destroy scripts; sets CDK_CONTEXT_FROM_TOML array when domainName
# is not already present in the caller's extra args.

_fleet_cdk_context_wants_toml() {
  local arg
  for arg in "$@"; do
    if [[ "$arg" == domainName=* ]]; then
      return 1
    fi
  done
  return 0
}

_fleet_load_cdk_context_from_toml() {
  local infra_dir="$1"
  shift
  CDK_CONTEXT_FROM_TOML=()
  if ! _fleet_cdk_context_wants_toml "$@"; then
    return 0
  fi
  python -m pip install -q -e "$infra_dir/../config"
  local line
  while IFS= read -r line; do
    CDK_CONTEXT_FROM_TOML+=("$line")
  done < <(
    python - <<'PY'
from krabby_fleet_config.loader import load_fleet_config

for arg in load_fleet_config().cdk_context_args():
    print(arg)
PY
  )
  if ((${#CDK_CONTEXT_FROM_TOML[@]} == 0)); then
    echo "error: fleet.toml produced no CDK context (check fleet/config/fleet.toml [deploy])" >&2
    exit 1
  fi
  echo "Using CDK context from fleet/config/fleet.toml (${#CDK_CONTEXT_FROM_TOML[@]} -c flags)"
}
