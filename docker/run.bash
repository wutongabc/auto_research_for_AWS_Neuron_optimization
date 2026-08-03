#!/usr/bin/env bash
set -euo pipefail

# Neuron Prefill Optimization — Docker launcher
# Uses NeuronCores 0-7 (devices 0-7) for prefill optimization work.
#
# Usage:
#   bash docker/run.bash           # Interactive (--rm on exit)
#   bash docker/run.bash -d        # Daemon (persistent background)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ZIGENG_ROOT="${ZIGENG_ROOT:-/dev3/zigeng}"
BC_ROOT="${BC_ROOT:-/dev3/zigeng/bc}"
BROWSECOMP_ROOT="$BC_ROOT/BrowseComp-Plus"
VLLM_FORK="${VLLM_FORK:-$BC_ROOT/vllm-neuron}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$BROWSECOMP_ROOT/local_models/huggingface}"
LOCAL_MODELS_DIR="$REPO_ROOT/local-models"

# Cache directories (shared with BrowseComp-Plus)
CACHE_BUNDLE_DIR="$BROWSECOMP_ROOT/local-models/vllm-neuron-cache-v1"
NKI_CACHE_DIR="$CACHE_BUNDLE_DIR/nki"
VLLM_CACHE_DIR="$CACHE_BUNDLE_DIR/vllm"

IMAGE_NAME="${IMAGE_NAME:-browsecomp-neuron}"
IMAGE_TAG="${IMAGE_TAG:-0.21.0}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="${CONTAINER_NAME:-neuron-prefill}"

# NeuronCores 0-7
NEURON_VISIBLE_DEVICES_RANGE="0,1,2,3,4,5,6,7"

# nkilib patch directories (from this repo — AI-editable)
KERNEL_PATCHES_DIR="$REPO_ROOT/kernel"
# vLLM-neuron patches (from this repo — AI-editable)
VLLM_PATCHES_DIR="$REPO_ROOT/vLLM-neuron"

# Base nkilib patches from BrowseComp-Plus (current production)
BASE_PATCHES_DIR="$BROWSECOMP_ROOT/patches/nkilib_tkg_swdge"
BASE_UTILS_PATCHES_DIR="$BROWSECOMP_ROOT/patches/nkilib_utils"

# Container paths
CONTAINER_NKILIB="/opt/conda/lib/python3.13/site-packages/nkilib/core/moe/moe_tkg"
CONTAINER_NKILIB_UTILS="/opt/conda/lib/python3.13/site-packages/nkilib/core/utils"
CONTAINER_VLLM="/opt/vllm-neuron/vllm_neuron"

# Parse mode
MODE="interactive"
if [[ "${1:-}" == "-d" ]] || [[ "${1:-}" == "--daemon" ]]; then
    MODE="daemon"
fi

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH" >&2
    exit 1
fi

if ! docker image inspect "$FULL_IMAGE_NAME" &> /dev/null; then
    echo "ERROR: Docker image $FULL_IMAGE_NAME not found" >&2
    exit 1
fi

mkdir -p "$HF_CACHE_DIR" "$NKI_CACHE_DIR" "$VLLM_CACHE_DIR" "$LOCAL_MODELS_DIR"

if [[ ! -f "$VLLM_FORK/vllm_neuron/model/qwen3/model_bf16.py" ]]; then
    echo "ERROR: vLLM-Neuron fork not found at: $VLLM_FORK" >&2
    exit 1
fi

# Load HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
    if grep -q "HF_TOKEN" ~/.bashrc 2>/dev/null; then
        source ~/.bashrc
    fi
fi

# Stop existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: $CONTAINER_NAME"
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

if [[ ${#NEURON_DEVICES[@]} -eq 0 ]]; then
    echo "WARNING: No Neuron devices found at /dev/neuron*" >&2
fi

echo "Starting container: $CONTAINER_NAME ($MODE)"
echo "  Image: $FULL_IMAGE_NAME"
echo "  Neuron cores: 0-7"
echo "  Workspace: $ZIGENG_ROOT -> /dev3/zigeng"
echo "  Opt project: $REPO_ROOT -> /dev3/zigeng/bc/opt"
echo "  vLLM-Neuron: $VLLM_FORK"
echo ""

DOCKER_ARGS=(
    --name "$CONTAINER_NAME"
    --network host
    --privileged
    "${NEURON_DEVICES[@]}"
    -v "$ZIGENG_ROOT:/dev3/zigeng"
    -v "$REPO_ROOT:/dev3/zigeng/bc/opt"
    -v "$HF_CACHE_DIR:/root/.cache/huggingface"
    -v "$NKI_CACHE_DIR:/var/tmp/nki-intermediate-cache"
    -v "$VLLM_FORK:/opt/vllm-neuron:ro"
    -v "$LOCAL_MODELS_DIR:/dev3/zigeng/bc/opt/local-models"
    -w /dev3/zigeng/bc/opt
    -e "HF_HOME=/root/.cache/huggingface"
    -e "TRANSFORMERS_CACHE=/root/.cache/huggingface"
    -e "PYTHONPATH=/opt/vllm-neuron"
    -e "PYTHONDONTWRITEBYTECODE=1"
    -e "VLLM_CACHE_ROOT=$VLLM_CACHE_DIR"
    -e "NEURON_VISIBLE_DEVICES=$NEURON_VISIBLE_DEVICES_RANGE"
    -e "FIRST_CORE=0"
    -e "NEURON_SKIP_EFA_AFFINITY=1"
)

# Mount base nkilib patches (production baseline from BrowseComp-Plus)
if [[ -d "$BASE_PATCHES_DIR" ]]; then
    for patch_file in "$BASE_PATCHES_DIR"/*.py; do
        if [[ -f "$patch_file" ]]; then
            fname="$(basename "$patch_file")"
            DOCKER_ARGS+=(-v "$patch_file:$CONTAINER_NKILIB/$fname:ro")
        fi
    done
    echo "  Mounted $(ls "$BASE_PATCHES_DIR"/*.py 2>/dev/null | wc -l) base nkilib TKG patches"
fi

if [[ -d "$BASE_UTILS_PATCHES_DIR" ]]; then
    for patch_file in "$BASE_UTILS_PATCHES_DIR"/*.py; do
        if [[ -f "$patch_file" ]]; then
            fname="$(basename "$patch_file")"
            DOCKER_ARGS+=(-v "$patch_file:$CONTAINER_NKILIB_UTILS/$fname:ro")
        fi
    done
    echo "  Mounted $(ls "$BASE_UTILS_PATCHES_DIR"/*.py 2>/dev/null | wc -l) base nkilib utils patches"
fi

# Mount kernel patches from this repo (AI-editable, override base patches)
if [[ -d "$KERNEL_PATCHES_DIR" ]] && ls "$KERNEL_PATCHES_DIR"/*.py &>/dev/null 2>&1; then
    for patch_file in "$KERNEL_PATCHES_DIR"/*.py; do
        if [[ -f "$patch_file" ]]; then
            fname="$(basename "$patch_file")"
            DOCKER_ARGS+=(-v "$patch_file:$CONTAINER_NKILIB/$fname")
        fi
    done
    echo "  Mounted $(ls "$KERNEL_PATCHES_DIR"/*.py 2>/dev/null | wc -l) opt kernel patches (overrides)"
fi

# Mount vLLM-neuron patches from this repo (AI-editable)
if [[ -d "$VLLM_PATCHES_DIR" ]] && ls "$VLLM_PATCHES_DIR"/*.py &>/dev/null 2>&1; then
    for patch_file in "$VLLM_PATCHES_DIR"/*.py; do
        if [[ -f "$patch_file" ]]; then
            fname="$(basename "$patch_file")"
            DOCKER_ARGS+=(-v "$patch_file:$CONTAINER_VLLM/$fname")
        fi
    done
    echo "  Mounted $(ls "$VLLM_PATCHES_DIR"/*.py 2>/dev/null | wc -l) vLLM-neuron patches (overrides)"
fi

[[ -n "${HF_TOKEN:-}" ]] && DOCKER_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")

verify_vllm_fork() {
    local container="$1"
    if docker exec "$container" python3 -c \
        "import inspect, vllm_neuron; from vllm_neuron.model.qwen3.model_bf16 import Qwen3ForCausalLM; assert inspect.getfile(vllm_neuron).startswith('/opt/vllm-neuron/')" 2>/dev/null; then
        echo "  vLLM-Neuron fork is active"
    else
        echo "  vLLM-Neuron fork verification failed" >&2
        exit 1
    fi
}

if [[ "$MODE" == "daemon" ]]; then
    docker run -d "${DOCKER_ARGS[@]}" "$FULL_IMAGE_NAME" /bin/bash -c "tail -f /dev/null"
    echo "  Container started in background"
    verify_vllm_fork "$CONTAINER_NAME"
    echo ""
    echo "Attach: docker exec -it $CONTAINER_NAME bash"
    echo "Stop:   docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
else
    docker run -d "${DOCKER_ARGS[@]}" "$FULL_IMAGE_NAME" /bin/bash -c "tail -f /dev/null"
    echo "  Container started"
    verify_vllm_fork "$CONTAINER_NAME"
    echo ""
    echo "Attaching (exit to stop and remove)..."
    docker exec -it "$CONTAINER_NAME" bash
    docker stop "$CONTAINER_NAME" &> /dev/null || true
    docker rm "$CONTAINER_NAME" &> /dev/null || true
fi
