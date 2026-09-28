# Cognito fleet operators

`FleetServiceStack` creates the Cognito **user pool** and empty **`operator`**
group. It does **not** create user accounts (no passwords in CDK). Admins add
operators after deploy. Membership in `operator` is required for portal,
`krabby-fleet`, and authenticated `/api/*` (otherwise 403).

Placeholders: `<region>`, `<user-pool-id>`, `<username>`, `<email>`,
`<password>`. Pool/client IDs are stack outputs `FleetCognitoUserPoolId` /
`FleetCognitoUserPoolClientId`, or SSM `/krabby/fleet/cognito-user-pool-id`
and `/krabby/fleet/cognito-app-client-id`.

```bash
export AWS_DEFAULT_REGION=<region>
export AWS_PAGER=""
export COGNITO_USER_POOL_ID=$(aws cloudformation describe-stacks --stack-name FleetServiceStack --query "Stacks[0].Outputs[?OutputKey=='FleetCognitoUserPoolId'].OutputValue" --output text)
```

## AWS Console

1. Open **Amazon Cognito** in `<region>`.
2. **User pools** → select **`krabby-fleet-operators`** (Id from
   `FleetCognitoUserPoolId` / SSM `/krabby/fleet/cognito-user-pool-id`).
3. **Users** → **Create user**
   - Username: `<username>`
   - Uncheck “Send an invitation” if setting the password yourself
   - Set a permanent password (or temporary, then change on first sign-in)
   - Email: `<email>`; mark email verified if prompted
4. Open the new user → **Group memberships** → **Add user to group** →
   **`operator`** (required).
5. Confirm: user shows group `operator`.

Sign-in: portal at `https://<fleet-domain>/`, or `krabby-fleet` (see
[`cli/README.md`](cli/README.md)).

## CLI

```bash
aws cognito-idp admin-create-user --user-pool-id "$COGNITO_USER_POOL_ID" --username '<username>' --user-attributes Name=email,Value='<email>' Name=email_verified,Value=true --message-action SUPPRESS --temporary-password '<TempPass1!>'

aws cognito-idp admin-set-user-password --user-pool-id "$COGNITO_USER_POOL_ID" --username '<username>' --password '<password>' --permanent

aws cognito-idp admin-add-user-to-group --user-pool-id "$COGNITO_USER_POOL_ID" --username '<username>' --group-name operator
```

Unauthenticated API check (expect `401`):

```bash
curl -sS -o /dev/null -w '%{http_code}\n' "https://<fleet-domain>/api/devices"
```

## CI operator password rotation

The persistent CI operator (`[ci].operator_username` in
[`config/fleet.toml`](config/fleet.toml)) has its password in GitHub secret
`COGNITO_CI_PASSWORD`. Workflow
[`.github/workflows/fleet-ci-rotate-cognito.yml`](../.github/workflows/fleet-ci-rotate-cognito.yml)
rotates that password on a monthly cron (and via **Actions → Fleet CI Cognito
rotate → Run workflow**):

1. Assumes `krabby-fleet-ci` via OIDC and calls `AdminSetUserPassword`.
2. Verifies USER_SRP_AUTH with the new password.
3. Updates `COGNITO_CI_PASSWORD` with `gh secret set` (using
   `FLEET_CI_SECRETS_PAT`).
4. If step 3 fails, restores the previous Cognito password from the still-valid
   workflow env copy of `COGNITO_CI_PASSWORD`.

One-time setup: create a fine-grained PAT with **Secrets: Read and write** on
this repo, store it as GitHub secret **`FLEET_CI_SECRETS_PAT`**. The default
`GITHUB_TOKEN` cannot update repository secrets.

## Notes

- No self-sign-up (`self_sign_up_enabled=False`).
- Users without `operator` can authenticate to Cognito but get **403** on
  fleet APIs.
- On stack destroy the pool is **retained**; delete users/pool in Cognito
  (or CLI) if you want a full cleanup — see [`infra/fleet-service.md`](infra/fleet-service.md).
