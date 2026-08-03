from __future__ import annotations

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

from .compressed_tensors_scheme import CompressedTensorsMoEScheme

__all__ = ["CompressedTensorsW4A4MxFp4MoE"]


class CompressedTensorsW4A4MxFp4MoE(CompressedTensorsMoEScheme):
    """AutoRound compressed-tensors MXFP4 expert scheme."""

    def __init__(self, prefix: str):
        self._method = Mxfp4MoEMethod(prefix=prefix)

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(self, *args, **kwargs):
        return self._method.create_weights(*args, **kwargs)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        return self._method.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        return self._method.process_weights_after_loading(layer)

    def apply_weights(self, layer: torch.nn.Module, dispatch_output):
        return self._method.apply(layer, dispatch_output)

    def get_triton_quant_info(self, layer: torch.nn.Module):
        return self._method.get_triton_quant_info(layer)

    def get_marlin_quant_info(self, layer: torch.nn.Module):
        return self._method.get_marlin_quant_info(layer)
