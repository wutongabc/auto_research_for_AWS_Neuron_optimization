# Tongyi-30B-A3B Prefill 性能优化报告

**硬件**: AWS Trainium 2 (trn2.48xlarge)  
**模型**: Qwen3MoE 30B total / 3B active, 128 experts, top-8

## 端到端结果

| 基准 | 未优化 | 第一轮后 | 第二轮后 | 总加速比 |
|------|--------|----------|----------|----------|
| Medium (TP=4, 32K ctx, 10×3000 tok) | 712 tok/s | 845 tok/s | 4,269 tok/s | **6.0×** |
| Full (TP=8, 128K ctx, 42×3000 tok) | 258 tok/s | 365 tok/s | 1,533 tok/s | **5.9×** |

所有测量保持 100% top-1 logit 正确性。未优化 baseline 使用原始 vLLM-Neuron 代码和默认参数在同一硬件上补测。

---

## 第一轮：短上下文优化

**基准配置**: TP=4, 16K context, 快速迭代（编译主导的工作负载）  
**阶段**: Phase 1 (参数) + Phase 2 (模型代码) + Phase 3 (MoE 内核)  
**结果**: 571 → 1,454 tok/s (2.5× 加速), 100% correctness

### 成功的优化

| # | 优化 | 增益 | 机制 |
|---|------|------|------|
| 1 | Prefill 分段 4096 → 512 tokens | +101% | 减少每 chunk QK 计算量和 padding 浪费 |
| 2 | BF16 KV cache (替换 FP8) | +0.4% | 消除量化/反量化开销 |
| 3 | KV block size 调到 64 | +0.8% | 平衡 cache 利用率与 gather 效率 |
| 4 | MoE block128 + skip-weight DMA | +6% | 128-token MoE 分块 + 跳过 padding block 的权重加载 |
| 5 | NKI 融合 router (sigmoid top-k mask) | +0.6% | 将 top-k 路由的 sigmoid/scatter 融合到单个 NKI 内核 |
| 6 | 注意力 scale 融合到 Q | +1.7% | 预乘 scale 到 Q tensor，消除 268M 元素乘法 |
| 7 | 去除冗余 mask 和 NaN 清理 | +16.2% | segmented prefill 中 in-sequence mask 和 nan_to_num 完全多余 |
| 8 | 六路 selective-expert SBUF 交织 | +0.1% | 微调 MoE CTE 引擎重叠度 |

### 失败的尝试

| 尝试 | 原因 |
|------|------|
| 2048-token prefill chunk | 每 chunk QK 计算翻倍，抵消减少 chunk 数的收益 |
| DMA/compute overlap (KV gather 前移) | XLA 编译器已自动调度独立操作的重叠 |
| FP8 KV cache 用于 segmented attention | 内核不支持 non-packed FP8 读取 |
| FP8 MoE expert weights (prefill) | MxFP8 CTE 仅 TRN3 硬件支持 |
| Expert parallel (EP=4) | 集合通信开销完全抵消并行收益 |
| BF16 softmax (内核级别) | 内核内部固定 FP32 softmax，外部改无效 |
| ~40 次 MoE CTE 内核微调 | SBUF 交织度、buffer 布局、DMA 顺序、expert scope——全部 <1% 且在噪声范围内 |

### 崩溃/不可行

| 尝试 | 失败原因 |
|------|----------|
| MoE block64 | NKI 要求最小 128-row tile |
| FP8 segmented attention | kernel assert: dtype mismatch |
| 融合 expert scaling | NKI source resolver 拒绝动态定义的 helper |
| 直接 transpose 到 gather indices | 运行时 GPSIMD 挂起 |

### 第一轮关键教训

Phase 3 花了大量精力（约 40 次实验）在 MoE CTE 内核微优化上，全部未能产生有意义的收益。**瓶颈不在 MoE**——而在 attention。但短上下文基准没有暴露这一点，因为 KV cache 很小。

---

## 第二轮：长上下文优化

**基准配置**: TP=4, **32K context, 10 轮 × 3000 tokens**（累积到 30K+ 的 prior KV）  
**阶段**: Phase 2 (模型代码，聚焦长上下文 attention) + Phase 3 (NKI 内核集成)  
**Baseline**: 845 tok/s（新基准配置下，已包含第一轮所有优化）  
**结果**: 845 → 4,269 tok/s (5.1× 加速), 100% correctness

重新设计基准以暴露真正瓶颈：长上下文 attention，每个 chunk 需要 attend 到 ~30K 的 prior KV tokens。

### 成功的优化

| # | 优化 | tok/s | 增益 | 机制 |
|---|------|-------|------|------|
| 1 | MAX_NUM_BATCHED_TOKENS=1024 + block128 | 900 | +6.3% | 减少内核调用次数，更好的块利用率 |
| 2 | GQA broadcast matmul | 1,032 | +14.7% | reshape + broadcast 避免 repeat_interleave 的 KV 4× 复制 |
| 3 | BF16 QK/PV matmul (FP32 softmax) | 1,125 | +9% | BF16 matmul 在 tensor engine 上 2× 吞吐 |
| 4 | **全 BF16 attention 路径 (去掉 FP32 softmax cast)** | 1,556 | **+38.4%** | 去掉 softmax 中的 FP32 cast，整个 attention 用 BF16 |
| 5 | Pre-scale Q (消除 score 级别乘法) | 1,612 | +3.6% | scale 乘在 Q [B,S,D] 上而非 scores [B,S,30K] 上 |
| 6 | All-BF16 RMSNorm | 1,626 | +0.8% | variance/rsqrt 也用 BF16 |
| 7 | **NKI flash_attention 集成** | 4,071 | **+150%** | 接入 nkilib `attention_cte` 内核（8K 分段 + 软件流水线 + GQA），通过 k_prior/v_prior 传入历史 KV |
| 8 | MAX_MODEL_LEN=30208 + O3 编译器 | 4,269 | +5% | 减少 KV padding，flash 分段数从 5 降到 4 |

### 失败的尝试

| 尝试 | 原因 |
|------|------|
| 16K flash attention section | 分段变少但每段更大，总计算量不变 |
| 10752 flash attention section | 结果不稳定 (4280/4222)，与 8K 基线持平 |
| 增大 chunk 到 3072 | total_K 增大导致分段数从 5 涨到 6 |
| int32 block-table 索引 | 编译器对 int64 生成了更好的代码 |
| broadcast KV heads in segmented GQA | 结果在噪声范围内 |
| Rank-3 GQA KV expand views | 比 repeat-interleave 慢 0.43% |

### 崩溃/不可行

| 尝试 | 失败原因 |
|------|----------|
| MAX_MODEL_LEN=30080 | decode megakernel 要求 256 对齐 |
| 128K KV budget 默认值 | OOM: 3.0 GiB needed, 1.1 GiB available |

---

## 两轮对比

| | 第一轮 | 第二轮 |
|---|--------|--------|
| 基准配置 | 16K context, 快速编译 | 32K context, 10×3000 token 轮次 |
| Baseline | 571 tok/s (未优化) | 845 tok/s (已含第一轮优化，更重的负载) |
| 最终结果 | 1,454 tok/s | 4,269 tok/s |
| 加速比 | **2.5×** | **5.1×** |
| 暴露的瓶颈 | MoE（误判） | Attention 内存带宽 |
| 总实验数 | ~60 | ~15 |
| 最大单步提升 | 去 mask/NaN (+16.2%) | NKI flash_attention (+150%) |
| 方法 | 穷举微调 | 定向瓶颈消除 |
| 关键教训 | MoE 内核已是最优 | Attention 带宽才是真正的墙 |

注：两轮使用不同的 benchmark，加速比不能简单相乘。第二轮的 baseline (845 tok/s) 已包含第一轮所有优化，但由于负载更重（10×3000 tokens 累积到 30K context），吞吐低于第一轮最终值 (1,454 tok/s)。

---

## 最终瓶颈分析

最终系统受**内存带宽**限制：
- 每 chunk 加载 ~30K prior KV (4 个 8K section)
- 注意力仅达峰值算力的 ~7%
- NKI 内核已用软件流水线、GQA、flash attention 最大化带宽利用
- 进一步提升需硬件升级或近似算法（会破坏 correctness）

---

## Full Model 验证

| 配置 | tok/s | Correctness | Baseline | 加速比 |
|------|-------|-------------|----------|--------|
| Medium (TP=4, 32K) | 4,269 | 100% | 845 | 5.1× |
| Full (TP=8, 128K) | 1,533 | 100% | 364.5 | 4.2× |
