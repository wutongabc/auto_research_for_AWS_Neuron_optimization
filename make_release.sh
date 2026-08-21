#!/usr/bin/env bash
set -euo pipefail

# Assemble the release folder with clean source code.
# Output: dist/ directory ready for distribution.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="${1:-$SCRIPT_DIR/dist}"

VLLM_SRC="$SCRIPT_DIR/vllm-neuron"
NKILIB_SRC="$SCRIPT_DIR/nkilib-fork"

echo "Assembling release into: $DIST_DIR"

if [[ ! -d "$VLLM_SRC/vllm_neuron" ]]; then
    echo "ERROR: vllm-neuron source not found at $VLLM_SRC" >&2
    exit 1
fi

if [[ ! -d "$NKILIB_SRC/core" ]]; then
    echo "ERROR: nkilib source not found at $NKILIB_SRC" >&2
    exit 1
fi

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Copy release scaffolding
cp -r "$SCRIPT_DIR/release/." "$DIST_DIR/"

# Copy vllm-neuron source (exclude compile caches, pycache, drafts)
mkdir -p "$DIST_DIR/vllm-neuron"
rsync -a \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='global_metric_store.json' \
    --include='vllm_neuron/***' \
    --include='examples/***' \
    --include='setup.py' \
    --include='pyproject.toml' \
    --include='README.md' \
    --include='LICENSE' \
    --include='NOTICE' \
    --exclude='*' \
    "$VLLM_SRC/" "$DIST_DIR/vllm-neuron/"

# Copy nkilib source (exclude pycache)
rsync -a \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "$NKILIB_SRC/" "$DIST_DIR/nkilib/"

# Make scripts executable
chmod +x "$DIST_DIR/build.sh" "$DIST_DIR/run.sh"

echo ""
echo "Release assembled:"
echo "  vllm-neuron: $(find "$DIST_DIR/vllm-neuron" -name '*.py' | wc -l) Python files"
echo "  nkilib:      $(find "$DIST_DIR/nkilib" -name '*.py' | wc -l) Python files"
echo ""
echo "To build: cd $DIST_DIR && bash build.sh"
