# Qwen3 / Qwen3MoE 使用指南

先把工作区实现同步到运行中的容器：

```bash
cd /dev3/zigeng/bc/vllm-neuron
bash UPDATE_CONTAINER.sh browsecomp-neuron
```

## Dense Qwen3 快速验证

在容器内用小模型先迭代；下面的配置只编译一个 512-token prefill bucket
和一个 batch-1 decode graph，但允许请求通过 segmented prefill 超过 512 tokens：

```bash
NEURON_COMPILED_ARTIFACTS=/path/to/qwen3-artifacts \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --block-size 128 \
  --num-gpu-blocks-override 32 \
  --enable-prefix-caching \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[512],"num_seqs_buckets":[1],"kv_segment_size_buckets":[512],"on_device_sampling_config":{"do_sample":true}}}' \
  --port 9999
```

## Qwen3MoE / Tongyi

四核 trn2.3xlarge 上启用 EP，使 128 experts 按 32/rank 分片：

```bash
NEURON_COMPILED_ARTIFACTS=/path/to/tongyi-artifacts \
python3 -m vllm.entrypoints.openai.api_server \
  --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --tokenizer /path/to/a/Qwen3-tokenizer \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --block-size 128 \
  --num-gpu-blocks-override 32 \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[2048],"num_seqs_buckets":[1],"kv_segment_size_buckets":[2048],"on_device_sampling_config":{"do_sample":true}}}' \
  --port 9999
```

如果 checkpoint 自带 tokenizer，可省略 `--tokenizer`。本地
`Tongyi-DeepResearch-30B-A3B` 副本没有 tokenizer 文件，因此实测时显式用了
`Qwen/Qwen3-0.6B` 的 tokenizer。

## 请求验证

```bash
curl http://127.0.0.1:9999/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

避免用 `vocab_size=1024`、TP=4 的随机模型判断 sampler 精度：此时
`vocab_per_rank=256` 恰好等于默认 `max_top_k=256`。模型路径验证可使用
正常 Qwen 词表、teacher forcing 或原始 logits；真实 Qwen 大词表不触发该
人工边界。

## 常见问题

- 架构未识别：运行 `UPDATE_CONTAINER.sh`，确认 registry 与 qwen3 模块同步。
- MoE 内存过高：确认传入了 `--enable-expert-parallel`；否则每 rank 会加载
  全部 experts，再仅对 intermediate 维做 TP。
- 修改 bucket 后重新编译：为 `NEURON_COMPILED_ARTIFACTS` 使用新的目录，
  避免混淆不同图形状。
- 编译调试时先用 `max-model-len=1024`、512-token segment、
  `max-num-seqs=1`；至少覆盖 521 tokens（刚超过 bucket）、接近上限的
  prompt，以及重复前缀。通过后再扩展生产 buckets。
