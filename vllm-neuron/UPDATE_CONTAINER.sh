#!/bin/bash
# 快速更新容器中的 Qwen3/Qwen3MoE 实现
# 用法: bash UPDATE_CONTAINER.sh [容器名称]

set -e

CONTAINER_NAME="${1:-browsecomp-neuron}"
VLLM_PATH="/opt/conda/lib/python3.13/site-packages/vllm_neuron"

echo "════════════════════════════════════════════════════════════"
echo "  更新 Qwen3/Qwen3MoE 实现到容器: $CONTAINER_NAME"
echo "════════════════════════════════════════════════════════════"
echo

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ 容器 $CONTAINER_NAME 未运行"
    exit 1
fi

echo "[1/5] 复制 qwen3 模块..."
docker exec "${CONTAINER_NAME}" rm -rf "${VLLM_PATH}/model/qwen3"
docker cp vllm_neuron/model/qwen3 "${CONTAINER_NAME}:${VLLM_PATH}/model/"
echo "✓ 完成"

echo "[2/5] 更新 registry.py..."
docker cp vllm_neuron/model/registry.py "${CONTAINER_NAME}:${VLLM_PATH}/model/"
echo "✓ 完成"

echo "[3/5] 更新 MoE blockwise 边界修复..."
docker cp vllm_neuron/functional/moe/moe_blockwise.py "${CONTAINER_NAME}:${VLLM_PATH}/functional/moe/"
echo "✓ 完成"

echo "[4/5] 清除 Python 缓存..."
docker exec "${CONTAINER_NAME}" bash -c "
    rm -rf ${VLLM_PATH}/model/__pycache__
    rm -rf ${VLLM_PATH}/model/qwen3/__pycache__
    find ${VLLM_PATH} -name '*.pyc' -delete
"
echo "✓ 完成"

echo "[5/5] 验证安装..."
docker exec -i "${CONTAINER_NAME}" python3 <<'PYVERIFY'
from vllm_neuron.model.registry import get_models

model_names = [name for name, _ in get_models()]
qwen3_models = [name for name in model_names if "Qwen3" in name and "VL" not in name]
expected = {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"}
missing = expected.difference(qwen3_models)
if missing:
    raise SystemExit(f"缺少 Qwen3 registry 项: {sorted(missing)}")
print("✓ Qwen3 registry:", sorted(qwen3_models))
PYVERIFY

docker exec "${CONTAINER_NAME}" python3 -m py_compile \
    "${VLLM_PATH}/model/qwen3/model_bf16.py"

echo
echo "════════════════════════════════════════════════════════════"
echo "✅ 更新和验证完成"
echo "════════════════════════════════════════════════════════════"
