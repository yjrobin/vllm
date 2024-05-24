"""Custom activation functions."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm._C import ops
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.utils import set_weight_attrs


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.silu_and_mul(out, x)
        return out


class GeluAndMul(nn.Module):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d]) * x[..., d:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.gelu_and_mul(out, x)
        return out


class NewGELU(nn.Module):

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c *
                                           (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        ops.gelu_new(out, x)
        return out


class FastGELU(nn.Module):

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 *
                                           (1.0 + 0.044715 * x * x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        ops.gelu_fast(out, x)
        return out


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size,
                                                     tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype))
        set_weight_attrs(self.scales, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_fast": FastGELU(),
    "gelu_new": NewGELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
}


def get_act_fn(
    act_fn_name: str,
    quant_config: Optional[QuantizationConfig] = None,
    intermediate_size: Optional[int] = None,
    input_is_parallel: bool = True,
    params_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn_name!r} is not supported.")

    act_fn = _ACTIVATION_REGISTRY[act_fn_name]
    if (quant_config is not None
            and act_fn_name in quant_config.get_scaled_act_names()):
        if intermediate_size is None:
            raise ValueError("intermediate_size must be specified for scaled "
                             "activation functions.")
        return ScaledActivation(act_fn, intermediate_size, input_is_parallel,
                                params_dtype)
    return act_fn


# ↓ add for smoothquant
class DequantSiluAndMulQuant(nn.Module):
    """An activation function for SwiGLU.
    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.
    Shapes:
        x: (num_tokens, 2 * d)
        return: (num_tokens, d)
    """

    # TODO(Zhang Ying): use_per_token_quant
    def __init__(self,
                 gate_dequant_scale: float = 1.0,
                 up_dequant_scale: float = 1.0,
                 quant_scale: float = 1.0,
                 use_per_token_quant: bool = True) -> None:
        super().__init__()
        self.register_parameter(
            "gate_dequant_scale",
            torch.nn.Parameter(
                torch.tensor(gate_dequant_scale,dtype=torch.float32,requires_grad=False))
        )
        self.register_parameter(
            "up_dequant_scale",
            torch.nn.Parameter(
                torch.tensor(up_dequant_scale,dtype=torch.float32,requires_grad=False))
        )
        self.register_parameter(
            "quant_scale",
            torch.nn.Parameter(
                torch.tensor(quant_scale, dtype=torch.float32,requires_grad=False))
        )
        self.use_per_token_quant = use_per_token_quant

    def _apply(self, fn):
        super()._apply(fn)
        self.gate_dequant_scale.data = self.gate_dequant_scale.cpu()
        self.up_dequant_scale.data = self.up_dequant_scale.cpu()
        self.quant_scale.data = self.quant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gate_dequant_scale.data = self.gate_dequant_scale.to(*args, **kwargs)
        self.gate_dequant_scale.data = self.gate_dequant_scale.to(torch.float32)
        self.up_dequant_scale.data = self.up_dequant_scale.to(*args, **kwargs)
        self.up_dequant_scale.data = self.up_dequant_scale.to(torch.float32)
        self.quant_scale.data = self.quant_scale.to(*args, **kwargs)
        self.quant_scale.data = self.quant_scale.to(torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.numel() // x.shape[-1]
        d = x.shape[-1] // 2
        out = torch.empty(*x.shape[:-1], d, dtype=torch.int8, device=x.device)
        if self.use_per_token_quant:
            scale = torch.empty(num_tokens,
                                dtype=torch.float32,
                                device=x.device)
            # tmp is used in kernel func
            tmp = torch.empty(num_tokens,
                              d,
                              dtype=torch.float32,
                              device=x.device)
            ops.dequant_silu_and_mul_quant(
                out, x, self.gate_dequant_scale.item(), self.up_dequant_scale.item(),
                scale, tmp)
            return out, scale
        else:
            ops.dequant_silu_and_mul_quant(
                out, x, self.gate_dequant_scale.item(), self.up_dequant_scale.item(),
                self.quant_scale.item())
            return out

