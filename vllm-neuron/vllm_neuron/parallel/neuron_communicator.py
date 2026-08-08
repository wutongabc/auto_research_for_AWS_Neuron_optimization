# SPDX-License-Identifier: Apache-2.0
import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class NeuronDeviceCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

        # Override base class behavior: enable All2All when EP>1, rather than when DP>1.
        # This override is necessary because Neuron EP degree is configurable, and we can
        # use All2All when DP=1.
        if self.is_ep_communicator:
            from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_degree

            self.use_all2all = get_neuron_ep_degree() > 1 and self.is_ep_communicator

        # TODO: determine behavior when all2all_backend != 'neuron' due to vllm defaults and/or user behavior
        if self.use_all2all and self.all2all_backend == "neuron":
            from vllm_neuron.parallel.all2all import NeuronAll2AllManager

            self.all2all_manager = NeuronAll2AllManager(self.cpu_group)
            logger.info_once(
                "Using NeuronAll2AllManager for all2all communication.",
                scope="global",
            )
