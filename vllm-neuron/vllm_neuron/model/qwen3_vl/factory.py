# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3-VL model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.interfaces import SupportsMaxPixels, SupportsSpatialMerge
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


class Qwen3VLForConditionalGeneration(
    nn.Module, SupportsSpatialMerge, SupportsMaxPixels
):
    """Factory that validates config and selects the appropriate Qwen3-VL implementation.

    The model runner passes `text_neuron_config` and `vision_neuron_config`
    for multimodal models.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        vision_neuron_config: VisionNeuronConfig | None,
    ) -> nn.Module:
        cls._validate_config(hf_config, text_neuron_config)

        from .model_bf16 import Qwen3VLForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        return hf_config.vision_config.spatial_merge_size**2

    @classmethod
    def get_max_pixels_token_count(
        cls, hf_config: PretrainedConfig, max_pixels: int
    ) -> int:
        patch_size = hf_config.vision_config.patch_size
        return max_pixels // (patch_size**2)

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
    ) -> None:
        pass
