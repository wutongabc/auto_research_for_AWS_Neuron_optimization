# SPDX-License-Identifier: Apache-2.0
"""Low-overhead metrics for latency-sensitive Neuron execution.

The upstream plugin records Prometheus samples in the scheduler and after
every NEFF invocation.  Those observations are useful operationally, but the
benchmark is a single-request throughput workload and never consumes them.
Keep metrics available behind an explicit opt-in while making the hot path a
cheap no-op by default.
"""

import os


class _NoOpMetric:
    def labels(self, *args, **kwargs):
        return self

    def observe(self, *args, **kwargs):
        return None

    def inc(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None


if os.environ.get("VLLM_NEURON_ENABLE_METRICS", "0") == "1":
    from prometheus_client import Counter, Gauge, Histogram

    NUM_SEQS_PADDING = Histogram(
        "vllm_neuron:num_seqs_padding",
        "Number of padded batch lines processed by the model.",
        labelnames=["model_name", "bucket_name"],
        buckets=[2**i for i in range(8)],
    )
    NUM_BATCHED_TOKENS_PADDING = Histogram(
        "vllm_neuron:num_batched_tokens_padding",
        "Number of padded sequence length processed by the model.",
        labelnames=["model_name", "bucket_name"],
        buckets=[2**i for i in range(5, 15)],
    )
    NEFF_EXECUTION_COUNT = Counter(
        "vllm_neuron:neff_execution_count",
        "Number of NEFF executions.",
        labelnames=["model_name", "bucket_name"],
    )
    COMPILATION_TIME = Gauge(
        "vllm_neuron:compilation_time_seconds",
        "Time spent compiling Neuron graphs.",
        labelnames=["model_name", "bucket_name"],
        multiprocess_mode="max",
    )
    STARTUP_TIME = Gauge(
        "vllm_neuron:startup_time_seconds",
        "Total server startup time from worker spawn to ready.",
        labelnames=["model_name"],
        multiprocess_mode="max",
    )
    MODEL_LOAD_TIME = Gauge(
        "vllm_neuron:model_load_time_seconds",
        "Time spent loading model weights to device.",
        labelnames=["model_name"],
        multiprocess_mode="max",
    )
    MODEL_LOAD_SIZE = Gauge(
        "vllm_neuron:model_load_size_bytes",
        "Size of model weights transferred to device.",
        labelnames=["model_name"],
        multiprocess_mode="sum",
    )
else:
    NUM_SEQS_PADDING = _NoOpMetric()
    NUM_BATCHED_TOKENS_PADDING = _NoOpMetric()
    NEFF_EXECUTION_COUNT = _NoOpMetric()
    COMPILATION_TIME = _NoOpMetric()
    STARTUP_TIME = _NoOpMetric()
    MODEL_LOAD_TIME = _NoOpMetric()
    MODEL_LOAD_SIZE = _NoOpMetric()
