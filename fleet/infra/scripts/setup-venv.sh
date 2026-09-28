#!/usr/bin/env bash
# Create fleet/infra/.venv, install Python CDK deps, and fetch a project-local
# Node + aws-cdk CLI into .tools/ (jsii and the CDK toolkit need Node; we do not
# install system packages).

set -euo pipefail

INFRA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$INFRA_DIR"

PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi

echo "Creating venv at $INFRA_DIR/.venv using $PYTHON ..."
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Portable Node for jsii + CDK CLI (stays under fleet/infra/.tools).
NODE_VERSION="${NODE_VERSION:-v20.18.1}"
TOOLS_DIR="$INFRA_DIR/.tools"
NODE_HOME="$TOOLS_DIR/node"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) NODE_DIST="node-${NODE_VERSION}-linux-x64" ;;
  Linux-aarch64|Linux-arm64) NODE_DIST="node-${NODE_VERSION}-linux-arm64" ;;
  Darwin-arm64) NODE_DIST="node-${NODE_VERSION}-darwin-arm64" ;;
  Darwin-x86_64) NODE_DIST="node-${NODE_VERSION}-darwin-x64" ;;
  *)
    echo "Unsupported platform for portable Node: $(uname -s) $(uname -m)" >&2
    exit 1
    ;;
esac

if [[ ! -x "$NODE_HOME/bin/node" ]]; then
  echo "Downloading portable Node ${NODE_VERSION} into .tools/ ..."
  mkdir -p "$TOOLS_DIR"
  TMP_TGZ="$(mktemp)"
  curl -fsSL "https://nodejs.org/dist/${NODE_VERSION}/${NODE_DIST}.tar.xz" -o "$TMP_TGZ"
  rm -rf "$NODE_HOME"
  tar -xJf "$TMP_TGZ" -C "$TOOLS_DIR"
  mv "$TOOLS_DIR/$NODE_DIST" "$NODE_HOME"
  rm -f "$TMP_TGZ"
fi

export PATH="$NODE_HOME/bin:$PATH"
if [[ ! -x "$NODE_HOME/bin/node" ]] || [[ ! -x "$NODE_HOME/bin/npm" ]]; then
  echo "Portable Node install looks broken under $NODE_HOME" >&2
  exit 1
fi

echo "Installing aws-cdk CLI into .tools/npm-global ..."
mkdir -p "$TOOLS_DIR/npm-global"
npm install --prefix "$TOOLS_DIR/npm-global" aws-cdk@2

cat <<EOF

Setup complete. Activate the venv before deploying:

  source $INFRA_DIR/.venv/bin/activate
  ./scripts/deploy-control-plane.sh

EOF
