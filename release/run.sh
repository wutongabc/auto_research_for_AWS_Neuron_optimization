#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-zigeng/vllm-neuron-opt}"
IMAGE_TAG="${IMAGE_TAG:-0.21.0.1.0.0-1.0.0}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-neuron-opt}"

# Optional: env file (e.g., configs/tp4.env for NEURON_VISIBLE_DEVICES)
ENV_FILE="${ENV_FILE:-}"
ENV_ARGS=()
if [[ -n "$ENV_FILE" ]]; then
    ENV_ARGS=(--env-file "$ENV_FILE")
fi

# Optional: compile cache persistence
COMPILE_CACHE="${COMPILE_CACHE:-}"
CACHE_ARGS=()
if [[ -n "$COMPILE_CACHE" ]]; then
    mkdir -p "$COMPILE_CACHE"
    CACHE_ARGS=(-v "$COMPILE_CACHE:/var/tmp/neuron-compile-cache")
fi

# Detect Neuron devices
NEURON_DEVICES=()
for device in /dev/neuron*; do
    if [[ -c "$device" ]]; then
        NEURON_DEVICES+=("--device=$device:$device")
    fi
done

if [[ ${#NEURON_DEVICES[@]} -eq 0 ]]; then
    echo "WARNING: No Neuron devices found at /dev/neuron*" >&2
fi

# Stop existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "$CONTAINER_NAME" &>/dev/null || true
    docker rm "$CONTAINER_NAME" &>/dev/null || true
fi

echo "Starting container: $CONTAINER_NAME"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
[[ -n "$ENV_FILE" ]] && echo "  Config: $ENV_FILE"
[[ -n "$COMPILE_CACHE" ]] && echo "  Cache: $COMPILE_CACHE"
echo ""
echo "To mount model weights, use: EXTRA_ARGS=\"-v /path/to/model:/model\" bash run.sh"
echo ""

docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --network host \
    --privileged \
    "${NEURON_DEVICES[@]}" \
    "${CACHE_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    ${EXTRA_ARGS:-} \
    -e "HF_HOME=/root/.cache/huggingface" \
    -e "PYTHONPATH=/opt/vllm-neuron" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    /bin/bash
