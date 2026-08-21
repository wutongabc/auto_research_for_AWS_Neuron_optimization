FROM public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04

COPY vllm-neuron/ /opt/vllm-neuron/
COPY nkilib/ /opt/conda/lib/python3.13/site-packages/nkilib/

ENV PYTHONPATH=/opt/vllm-neuron
ENV PYTHONDONTWRITEBYTECODE=1
