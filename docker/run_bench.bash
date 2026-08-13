#!/usr/bin/env bash
set -euo pipefail

# Parameterized Docker launcher for benchmark containers.
# Supports running multiple containers with different NeuronCore assignments.
#
# Usage:
#   bash docker/run_bench.bash <container_name> <cores> <src_dir> [port]
#
# Examples:
#   bash docker/run_bench.bash bench-oss20b "0,1,2,3,4,5,6,7" /dev3/zigeng/bc/opt 8100
#   bash docker/run_bench.bash bench-qwen3vl "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" /dev3/zigeng/bc/opt-baseline 8101

CONTAINER_NAME="${1:?Usage: $0 <container_name> <cores> <src_dir> [port]}"
NEURON_CORES="${2:?Usage: $0 <container_name> <cores> <src_dir> [port]}"
SRC_DIR="${3:?Usage: $0 <container_name> <cores> <src_dir> [port]}"
PORT="${4:-8100}"

ZIGENG_ROOT="${ZIGENG_ROOT:-/dev3/zigeng}"
BC_ROOT="${BC_ROOT:-/dev3/zigeng/bc}"
BROWSECOMP_ROOT="$BC_ROOT/BrowseComp-Plus"
VLLM_FORK="$SRC_DIR/vllm-neuron"
NKILIB_FORK="$SRC_DIR/nkilib-fork"
HF_CACHE_DIR="${HF_CACHE_DIR:-$BROWSECOMP_ROOT/local_models/huggingface}"

IMAGE_NAME="${IMAGE_NAME:-browsecomp-neuron}"
IMAGE_TAG="${IMAGE_TAG:-0.21.0}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

CONTAINER_NKILIB="/opt/conda/lib/python3.13/site-packages/nkilib"
CONTAINER_VLLM="/opt/vllm-neuron"

# Load HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
    if grep -q "HF_TOKEN" ~/.bashrc 2>/dev/null; then
        source ~/.bashrc
    fi
fi

# Stop existing container with same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "$CONTAINER_NAME" &> /dev/null || true
    docker rm "$CONTAINER_NAME" &> /dev/null || true
fi

# Detect Neuron devices
NEURON_DEVICES=()
for device in /dev/neuron*; do
    if [[ -c "$device" ]]; then
        NEURON_DEVICES+=("--device=$device:$device")
    fi
done

echo "Starting container: $CONTAINER_NAME"
echo "  Image: $FULL_IMAGE_NAME"
echo "  Neuron cores: $NEURON_CORES"
echo "  Source dir: $SRC_DIR"
echo "  Port: $PORT"
echo ""

# Use unique cache dirs per container to avoid conflicts
CACHE_DIR="$SRC_DIR/.cache-$CONTAINER_NAME"
mkdir -p "$CACHE_DIR/nki" "$CACHE_DIR/vllm" "$HF_CACHE_DIR"

DOCKER_ARGS=(
    --name "$CONTAINER_NAME"
    --network host
    --privileged
    "${NEURON_DEVICES[@]}"
    -v "$ZIGENG_ROOT:/dev3/zigeng"
    -v "$SRC_DIR:/dev3/zigeng/bc/opt"
    -v "$HF_CACHE_DIR:/root/.cache/huggingface"
    -v "$CACHE_DIR/nki:/var/tmp/nki-intermediate-cache"
    -v "$VLLM_FORK:$CONTAINER_VLLM"
    -v "$NKILIB_FORK:$CONTAINER_NKILIB"
    -w /dev3/zigeng/bc/opt
    -e "HF_HOME=/root/.cache/huggingface"
    -e "TRANSFORMERS_CACHE=/root/.cache/huggingface"
    -e "PYTHONPATH=/opt/vllm-neuron"
    -e "PYTHONDONTWRITEBYTECODE=1"
    -e "VLLM_CACHE_ROOT=$CACHE_DIR/vllm"
    -e "NEURON_VISIBLE_DEVICES=$NEURON_CORES"
    -e "NEURON_SKIP_EFA_AFFINITY=1"
    -e "PORT=$PORT"
)

[[ -n "${HF_TOKEN:-}" ]] && DOCKER_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")

docker run -d "${DOCKER_ARGS[@]}" "$FULL_IMAGE_NAME" /bin/bash -c "tail -f /dev/null"
echo "  Container started in background"
echo "  Exec: docker exec $CONTAINER_NAME bash -c '...'"
echo "  Stop: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
