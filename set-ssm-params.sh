#!/usr/bin/env sh
# Populate the /krabby/bench SSM parameters that the krabby-bench watchdog
# reads (SMTP + GitHub credentials). Operator-side, fleet-wide: run once from a
# workstation whose AWS identity can `ssm:PutParameter`, and again to rotate.
# This is NOT a Jetson bring-up step — devices only hold a read-only IAM key.
#
# Safe to re-run: existing values are shown (secrets masked) and kept when you
# press Enter.
#
# Region: the devices read us-east-1 (hardcoded in bench/krabby_bench/_ssm.py),
# so leave REGION at us-east-1 unless you also change the device side.
#
# POSIX sh (no bashisms) for maximum portability across operator machines.

set -eu

PREFIX="/krabby/bench"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# Parameter names whose current value must be masked in hints.
is_secure() {
  case "$1" in
    smtp-password|github-token) return 0 ;;
    *) return 1 ;;
  esac
}

# Fetch existing params once into a tab-separated temp file ("<name>\t<value>").
CURRENT_FILE="$(mktemp)"
trap 'rm -f "$CURRENT_FILE"' EXIT INT TERM

echo "Fetching existing parameters..."
aws ssm get-parameters-by-path \
  --path "$PREFIX" --with-decryption --region "$REGION" --no-cli-pager 2>/dev/null \
  | jq -r --arg p "${PREFIX}/" '.Parameters[] | [(.Name | ltrimstr($p)), .Value] | @tsv' \
  > "$CURRENT_FILE" || true
echo

# Current value of a parameter name, or empty if unset.
current() {
  awk -F '\t' -v k="$1" '$1==k { sub(/^[^\t]*\t/, ""); print; exit }' "$CURRENT_FILE"
}

# " [current: ...]" hint for the prompt (masked for secrets), or empty.
hint() {
  cur="$(current "$1")"
  [ -z "$cur" ] && return
  if is_secure "$1"; then
    printf ' [current: ****]'
  else
    printf ' [current: %s]' "$cur"
  fi
}

# Prompt on stderr (so command substitution captures only the value on stdout);
# echo the typed value, or the existing value when the reply is empty.
prompt() {
  cur="$(current "$1")"
  printf '%s%s: ' "$1" "$(hint "$1")" >&2
  IFS= read -r reply || reply=''
  if [ -z "$reply" ] && [ -n "$cur" ]; then
    printf '%s' "$cur"
  else
    printf '%s' "$reply"
  fi
}

# Like prompt(), but with terminal echo disabled (POSIX has no `read -s`).
prompt_secret() {
  cur="$(current "$1")"
  printf '%s%s: ' "$1" "$(hint "$1")" >&2
  if [ -t 0 ]; then
    stty_orig="$(stty -g)"
    stty -echo
    IFS= read -r reply || reply=''
    stty "$stty_orig"
    printf '\n' >&2
  else
    IFS= read -r reply || reply=''
  fi
  if [ -z "$reply" ] && [ -n "$cur" ]; then
    printf '%s' "$cur"
  else
    printf '%s' "$reply"
  fi
}

put() {
  name="$1"; value="$2"; type="${3:-String}"
  if [ -z "$value" ]; then
    printf '  skipped %s/%s (blank)\n' "$PREFIX" "$name"
    return
  fi
  aws ssm put-parameter \
    --region "$REGION" \
    --name "${PREFIX}/${name}" \
    --value "$value" \
    --type "$type" \
    --overwrite \
    --no-cli-pager > /dev/null
  printf '  set %s/%s\n' "$PREFIX" "$name"
}

echo "Setting SSM parameters under ${PREFIX} (region: ${REGION})"
echo "Press Enter to keep an existing value."
echo

smtp_host="$(prompt smtp-host)"
smtp_port="$(prompt smtp-port)"
smtp_port="${smtp_port:-587}"
smtp_user="$(prompt smtp-user)"
smtp_pass="$(prompt_secret smtp-password)"
smtp_from="$(prompt smtp-from)"
smtp_to="$(prompt smtp-to)"
gh_repo="$(prompt github-repo)"
gh_token="$(prompt_secret github-token)"

echo
echo "Writing parameters..."
put smtp-host     "$smtp_host"
put smtp-port     "$smtp_port"
put smtp-user     "$smtp_user"
put smtp-password "$smtp_pass" SecureString
put smtp-from     "$smtp_from"
put smtp-to       "$smtp_to"
put github-repo   "$gh_repo"
put github-token  "$gh_token" SecureString

echo
echo "Done. Verify with:"
echo "  aws ssm get-parameters-by-path --path $PREFIX --with-decryption --region $REGION"
