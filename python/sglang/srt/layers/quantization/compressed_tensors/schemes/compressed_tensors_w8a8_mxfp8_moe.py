from __future__ import annotations

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

from .compressed_tensors_scheme import CompressedTensorsMoEScheme

__all__ = ["CompressedTensorsW8A8MxFp8MoE"]


class CompressedTensorsW8A8MxFp8MoE(CompressedTensorsMoEScheme):
    """AutoRound compressed-tensors MXFP8 W8A8 expert scheme."""

    def __init__(self):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            use_mxfp8=True,
        )
        self._method = Fp8MoEMethod(quant_config)

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def _rename_parameter(layer: torch.nn.Module, old_name: str, new_name: str) -> None:
        param = layer._parameters.pop(old_name)
        layer.register_parameter(new_name, param)

    @staticmethod
    def _alias_autoround_scales(layer: torch.nn.Module) -> None:
        layer.w13_weight_scale_inv = layer.w13_weight_scale
        layer.w2_weight_scale_inv = layer.w2_weight_scale
        layer.w13_weight_scale_inv.format_ue8m0 = True
        layer.w2_weight_scale_inv.format_ue8m0 = True

    def create_weights(self, *args, **kwargs):
        self._method.create_weights(*args, **kwargs)
        layer = kwargs.get("layer")
        if layer is None and args:
            layer = args[0]
        if layer is None:
            raise ValueError("CompressedTensorsW8A8MxFp8MoE requires layer.")
        self._rename_parameter(layer, "w13_weight_scale_inv", "w13_weight_scale")
        self._rename_parameter(layer, "w2_weight_scale_inv", "w2_weight_scale")
        layer.w13_weight_scale.format_ue8m0 = True
        layer.w2_weight_scale.format_ue8m0 = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        return self._method.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._alias_autoround_scales(layer)
        self._method.process_weights_after_loading(layer)
        if hasattr(layer, "w13_weight_scale_inv"):
            layer.w13_weight_scale = layer.w13_weight_scale_inv
        if hasattr(layer, "w2_weight_scale_inv"):
            layer.w2_weight_scale = layer.w2_weight_scale_inv

    def apply_weights(self, layer: torch.nn.Module, dispatch_output):
        self._alias_autoround_scales(layer)
        return self._method.apply(layer, dispatch_output)

    def get_triton_quant_info(self, layer: torch.nn.Module):
        self._alias_autoround_scales(layer)
        return self._method.get_triton_quant_info(layer)

    def get_marlin_quant_info(self, layer: torch.nn.Module):
        self._alias_autoround_scales(layer)
        return self._method.get_marlin_quant_info(layer)
