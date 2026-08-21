#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-zigeng/vllm-neuron-opt}"
IMAGE_TAG="${IMAGE_TAG:-0.21.0.1.0.0-1.0.0}"

echo "Building image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "$SCRIPT_DIR"
echo "Done: ${IMAGE_NAME}:${IMAGE_TAG}"
