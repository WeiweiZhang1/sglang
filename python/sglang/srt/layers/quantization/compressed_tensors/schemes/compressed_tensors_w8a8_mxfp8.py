from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp8_utils import dispatch_w8a8_mxfp8_linear

__all__ = ["CompressedTensorsW8A8MxFp8"]


class CompressedTensorsW8A8MxFp8(CompressedTensorsLinearScheme):
    """OCP MXFP8 linear scheme exported by AutoRound compressed-tensors.

    AutoRound stores MXFP8 weights as E4M3 FP8 plus UE8M0 uint8 scales with
    one scale per 32 input channels.  This differs from block-FP8 compressed
    tensors, so it needs an explicit grouped-scale loader.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        # CUDA MXFP8 dense kernels require Blackwell. ROCm gfx95 is handled by
        # the dispatched backend, but compressed-tensors capability checks are
        # CUDA-oriented.
        return 100

    def __init__(self):
        self.w8a8_mxfp8_linear = dispatch_w8a8_mxfp8_linear()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition % 32 != 0:
            raise ValueError(
                "MXFP8 compressed-tensors weights require the sharded input "
                f"dimension to be divisible by 32, got {input_size_per_partition}."
            )

        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 32,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale.format_ue8m0 = True
        layer.register_parameter("weight_scale", weight_scale)
        layer.input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)
        layer.weight_scale.format_ue8m0 = True
        layer.input_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.w8a8_mxfp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            output_dtype=getattr(layer, "orig_dtype", None),
        )
