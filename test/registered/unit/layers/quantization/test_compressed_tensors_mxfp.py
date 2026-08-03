import unittest

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8MxFp8,
    CompressedTensorsW8A8MxFp8MoE,
)


def _mxfp_args(bits: int, dynamic: bool) -> QuantizationArgs:
    return QuantizationArgs.model_validate(
        {
            "num_bits": bits,
            "type": "float",
            "strategy": "group",
            "group_size": 32,
            "symmetric": True,
            "dynamic": dynamic,
        }
    )


class TestCompressedTensorsMXFP(unittest.TestCase):
    def test_autoround_mxfp8_scheme_is_detected(self):
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "mxfp8-quantized",
                "provider": "auto-round",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": _mxfp_args(bits=8, dynamic=False).model_dump(),
                        "input_activations": _mxfp_args(
                            bits=8, dynamic=True
                        ).model_dump(),
                    }
                },
            }
        )

        quant_config._check_scheme_supported = lambda *args, **kwargs: True
        scheme = quant_config._get_scheme_from_parts(
            quant_config.target_scheme_map["Linear"]["weights"],
            quant_config.target_scheme_map["Linear"]["input_activations"],
        )

        self.assertIsInstance(scheme, CompressedTensorsW8A8MxFp8)

    def test_mxfp8_detection_requires_group_size_32(self):
        quant_config = CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format=CompressionFormat.dense.value,
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )
        weight_quant = _mxfp_args(bits=8, dynamic=False)
        input_quant = _mxfp_args(bits=8, dynamic=True)
        input_quant.group_size = 16

        self.assertFalse(quant_config._is_mxfp8_w8a8(weight_quant, input_quant))

    def test_autoround_mxfp4_moe_scheme_is_detected(self):
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "provider": "auto-round",
                "config_groups": {
                    "group_0": {
                        "targets": ["RoutedExperts"],
                        "weights": _mxfp_args(bits=4, dynamic=False).model_dump(),
                        "input_activations": _mxfp_args(
                            bits=4, dynamic=True
                        ).model_dump(),
                    }
                },
            }
        )

        self.assertTrue(
            quant_config._is_mxfp4_w4a4(
                quant_config.target_scheme_map["RoutedExperts"]["weights"],
                quant_config.target_scheme_map["RoutedExperts"][
                    "input_activations"
                ],
            )
        )

    def test_autoround_mxfp8_moe_scheme_requires_blackwell(self):
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "mxfp8-quantized",
                "provider": "auto-round",
                "config_groups": {
                    "group_0": {
                        "targets": ["RoutedExperts"],
                        "weights": _mxfp_args(bits=8, dynamic=False).model_dump(),
                        "input_activations": _mxfp_args(
                            bits=8, dynamic=True
                        ).model_dump(),
                    }
                },
            }
        )

        quant_config._check_scheme_supported = lambda *args, **kwargs: False
        with self.assertRaisesRegex(NotImplementedError, "MXFP8.*MoE.*SM100/SM120"):
            quant_config.get_moe_scheme(
                layer=type("RoutedExperts", (), {})(),
                layer_name="model.layers.0.mlp",
            )

        quant_config._check_scheme_supported = lambda *args, **kwargs: True
        scheme = quant_config.get_moe_scheme(
            layer=type("RoutedExperts", (), {})(),
            layer_name="model.layers.0.mlp",
        )
        self.assertIsInstance(scheme, CompressedTensorsW8A8MxFp8MoE)


if __name__ == "__main__":
    unittest.main()
