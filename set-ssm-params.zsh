#!/usr/bin/env zsh
# One-shot script to populate /krabby/bench SSM parameters.
# Delete after use.

set -e

PREFIX="/krabby/bench"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

typeset -A CURR
typeset -A IS_SECURE
IS_SECURE=(smtp-password 1 github-token 1)

echo "Fetching existing parameters..."
while IFS=$'\t' read -r k v; do
  CURR[$k]="$v"
done < <(aws ssm get-parameters-by-path \
  --path "$PREFIX" --with-decryption --region "$REGION" --no-cli-pager 2>/dev/null \
  | jq -r --arg p "${PREFIX}/" '.Parameters[] | [(.Name | ltrimstr($p)), .Value] | @tsv')
echo ""

_hint() {
  local name="$1"
  [[ -z "${CURR[$name]}" ]] && return
  if [[ -n "${IS_SECURE[$name]}" ]]; then
    echo " [current: ****]"
  else
    echo " [current: ${CURR[$name]}]"
  fi
}

prompt() {
  local name="$1" REPLY=""
  read "REPLY?${name}$(_hint $name): "
  if [[ -z "$REPLY" && -n "${CURR[$name]}" ]]; then
    printf '%s' "${CURR[$name]}"
  else
    printf '%s' "$REPLY"
  fi
}

prompt_secret() {
  local name="$1" REPLY=""
  read -rs "REPLY?${name}$(_hint $name): "
  echo
  if [[ -z "$REPLY" && -n "${CURR[$name]}" ]]; then
    printf '%s' "${CURR[$name]}"
  else
    printf '%s' "$REPLY"
  fi
}

put() {
  local name="$1" value="$2" type="${3:-String}"
  if [[ -z "$value" ]]; then
    echo "  skipped ${PREFIX}/${name} (blank)"
    return
  fi
  aws ssm put-parameter \
    --region "$REGION" \
    --name "${PREFIX}/${name}" \
    --value "$value" \
    --type "$type" \
    --overwrite \
    --no-cli-pager > /dev/null
  echo "  set ${PREFIX}/${name}"
}

echo "Setting SSM parameters under ${PREFIX} (region: ${REGION})"
echo "Press Enter to keep an existing value.\n"

smtp_host=$(prompt smtp-host)
smtp_port=$(prompt smtp-port)
smtp_port="${smtp_port:-587}"
smtp_user=$(prompt smtp-user)
smtp_pass=$(prompt_secret smtp-password)
smtp_from=$(prompt smtp-from)
smtp_to=$(prompt smtp-to)
gh_repo=$(prompt github-repo)
gh_token=$(prompt_secret github-token)

echo "\nWriting parameters..."
put smtp-host     "$smtp_host"
put smtp-port     "$smtp_port"
put smtp-user     "$smtp_user"
put smtp-password "$smtp_pass" SecureString
put smtp-from     "$smtp_from"
put smtp-to       "$smtp_to"
put github-repo   "$gh_repo"
put github-token  "$gh_token" SecureString

echo "\nDone. Verify with:"
echo "  aws ssm get-parameters-by-path --path $PREFIX --with-decryption --region $REGION"
