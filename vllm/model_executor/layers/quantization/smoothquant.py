from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_world_size


class SmoothQuantConfig(QuantizationConfig):
    """Config class for SmoothQuant
    Reference: https://github.com/mit-han-lab/smoothquant
    """

    def __init__(
        self,
        weight_bits: int,
        quant_type: str = "tensor"
    ) -> None:
        self.weight_bits = weight_bits
        self.quant_type = quant_type

        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only w8a8 quantization is supported for "
                f"SmoothQuant, but got {self.weight_bits} bits.")
        if self.quant_type != "tensor":
            raise ValueError(
                "Currently, only tensor wise quantization is supported for "
                f"SmoothQuant, but got {self.quant_type} type quantization.")

    def __repr__(self) -> str:
        return (f"SmoothQuantConfig(weight_bits={self.weight_bits}, "
                f"quant_type={self.quant_type})")

    def get_name(self) -> str:
        return "smoothquant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.float]

    def get_min_capability(self) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        """List of filenames to search for in the model directory."""
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothQuantConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        quant_type = cls.get_from_keys(config, ["quant_type", "q_type"])
        return cls(weight_bits, quant_type)

    def get_linear_method(self) -> "SmoothLinearMethod":
        return SmoothLinearMethod(world_size=get_tensor_model_parallel_world_size())

    def get_scaled_act_names(self) -> List[str]:
        return []


class SmoothLinearMethod(LinearMethodBase):
    def __init__(self, world_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_dequant_after_row = world_size > 1
        self.dtpye = None

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=torch.int8),
                                       requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        self.dtpye = params_dtype
        return {"weight": weight}

    def apply_weights(
        self,
        weights: Dict[str, torch.Tensor],
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        scale: Optional[torch.Tensor] = None,
        dequant_scale: float = 1.0,
        is_row: bool = False,
    ) -> torch.Tensor:
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        weight = weights["weight"]
        y = torch.empty((x.shape[0], weight.shape[0]),dtype=torch.int32,device=x.device)
        ops.linear_a8_w8_o32_(x, weight, y)
        y = y.view(*x_shape[:-1], -1)
        if is_row and self.apply_dequant_after_row:
            # when tp > 1, duquant first(To improve accuracy?)
            out = torch.empty_like(y, dtype=self.dtpye)
            ops.dequant(out, y, scale, dequant_scale)
            y = out
        return y
