# Bench SSH user and CI pubkey

SSH through a Secure Tunnel uses a **Linux login on the Orin**, not Cognito.
The fleet CLI and E2E tests default to user **`operator`**. Cognito
(`OPERATORS.md`, `ci-fleet-operator@krabbyco.com`) is only for the fleet API
and portal.

## 1. Create the `operator` Linux user (bench Orin, once)

On the bench Orin, create a local user named `operator` with primary group
`users`. Jetson/Ubuntu images already have a system group named `operator`, so
use `-g users` on `useradd`.

```bash
id operator
```

If `id` succeeds, skip `useradd` and run only the `~/.ssh` steps below.

Otherwise:

```bash
sudo useradd -m -s /bin/bash -g users operator
id operator   # must succeed before continuing
sudo passwd -l operator   # optional: disable password login; pubkey only
sudo mkdir -p /home/operator/.ssh
sudo chmod 700 /home/operator/.ssh
sudo chown -R operator: /home/operator/.ssh
```

Confirm `sshd` is running:

```bash
systemctl is-active ssh
```

Expect `active`.

## 2. Generate CI key pair (bench Orin)

```bash
KEY="$HOME/bench-ci-ed25519"

ssh-keygen -t ed25519 -f "$KEY" -N "" -C "krabby-fleet-ci"
```

## 3. Install the public key

```bash
KEY="$HOME/bench-ci-ed25519"

sudo bash -c 'cat >> /home/operator/.ssh/authorized_keys' < "${KEY}.pub"
sudo chmod 600 /home/operator/.ssh/authorized_keys
sudo chown operator: /home/operator/.ssh/authorized_keys
```

## 4. GitHub secret

Print the private key on the Orin:

```bash
KEY="$HOME/bench-ci-ed25519"

echo "Paste into GitHub secret BENCH_CI_SSH_PRIVATE_KEY:"
cat "$KEY"
```

On GitHub, open this repository's **Settings > Secrets and variables >
Actions > New repository secret**.

| Secret | Value |
|--------|--------|
| `BENCH_CI_SSH_PRIVATE_KEY` | Full private key (including `BEGIN` / `END` lines) |

## 5. Test before CI

Run from any Linux machine that meets the requirements below (CI uses a GitHub
runner; a laptop is the usual manual run). The test checks: Cognito login,
fleet API tunnel open, `krabby-fleet ssh` + source `localproxy`, pubkey auth
as the Linux `operator` user on the bench, `echo hello` over SSH, then tunnel
closed. Traffic still relayed through AWS Secure Tunneling even when pytest runs
on the bench Orin.

**Requirements**

- Steps 1-3 done on the bench (Linux `operator` user, CI pubkey in
  `authorized_keys`)
- Deployed fleet service; bench enrolled; `krabby-agent` running
- Repo checkout with committed [`config/fleet.toml`](config/fleet.toml)
- On `PATH`: `localproxy`, `ssh` (see install steps below)

**Install source `localproxy` (once per operator machine)**

arm64: [`ENROLL.md`](ENROLL.md#destination-localproxy-if-apt-failed). x86_64:
[`SSH-TUNNEL.md`](SSH-TUNNEL.md#ssh-source-install-localproxy). Confirm:
`command -v localproxy`.

**Environment**

Everything else comes from [`config/fleet.toml`](config/fleet.toml) (pytest loads
it at startup). Export these before running pytest locally:

```bash
export BENCH_E2E=1
export COGNITO_CI_PASSWORD='...'
export AWS_ACCESS_KEY_ID='...'
export AWS_SECRET_ACCESS_KEY='...'
# export BENCH_SSH_USER=operator   # optional; default is operator
```

Install the CI private key on **the machine running pytest** (laptop, CI runner,
etc. - not necessarily the Orin). Same path as [`fleet-ci.yml`](../.github/workflows/fleet-ci.yml):
paste the value of GitHub secret `BENCH_CI_SSH_PRIVATE_KEY` from step 4:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat > ~/.ssh/id_ed25519 <<'EOF'
(paste full private key including BEGIN / END lines)
EOF
chmod 600 ~/.ssh/id_ed25519
```

Opening the tunnel and SSH use Cognito and the fleet API only. The test also
calls `DescribeTunnel` at the end to confirm the CLI closed the tunnel; boto3
needs AWS credentials for that call. Locally, export an IAM user's keys with
`iot:DescribeTunnel` (same permission as the CI role). In GitHub Actions there
are no AWS keys in repository secrets: the workflow assumes
`FLEET_CI_ROLE_ARN` from `fleet.toml` via OIDC and the runner gets short-lived
credentials automatically.

**Run**

Installs `krabby-fleet` into the venv; pytest subprocess-invokes
`krabby-fleet ssh` (you do not run the CLI yourself).

```bash
cd fleet/service
python3 -m venv .venv
source .venv/bin/activate
pip install -e ../config -e ../cli -e ".[e2e]"

pytest tests_e2e/test_ssh_tunnel_e2e.py::test_krabby_fleet_ssh_runs_command_end_to_end -q
```

## 6. Remove private key from Orin

After the GitHub secret is saved and step 5 has the key on the pytest machine:

```bash
KEY="$HOME/bench-ci-ed25519"
rm -f "$KEY" "${KEY}.pub"
```

## 7. Test in GitHub Actions

Push a branch or run `fleet-ci.yml` on `main`. The **Bench E2E** job writes
`BENCH_CI_SSH_PRIVATE_KEY` to `~/.ssh/id_ed25519` before
`pytest fleet/service/tests_e2e`.

Override SSH user in CI with env `BENCH_SSH_USER` in the workflow if not
`operator`.
