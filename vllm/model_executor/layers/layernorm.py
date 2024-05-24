"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
                scale,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out


# ↓ add for smoothquant
class RMSNormQuant(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = torch.empty_like(x, dtype=torch.int8)
        ops.rms_norm_quant(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out


class AddResidualRMSNormQuant(nn.Module):
    """Root mean square normalization.
    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, 
                x: torch.Tensor, 
                residual: torch.Tensor, 
                scale: torch.Tensor = None) -> torch.Tensor:
        out = torch.empty_like(x, dtype=torch.int8)
        ops.fused_add_rms_norm_quant(out, x, residual, self.weight.data, self.variance_epsilon)
        return out, residual


class DequantAddResidualRMSNormQuant(nn.Module):
    """Root mean square normalization.
    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # TODO(Zhang Ying): use_per_token_dequant
    def __init__(self,
                 hidden_size: int,
                 dequant_scale: float = 1.0,
                 use_per_token_dequant: bool = True,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.register_parameter(
            "dequant_scale",
            torch.nn.Parameter(torch.tensor(dequant_scale,dtype=torch.float32,requires_grad=False))
        )
        self.use_per_token_dequant = use_per_token_dequant

    def _apply(self, fn):
        super()._apply(fn)
        self.dequant_scale.data = self.dequant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dequant_scale.data = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale.data = self.dequant_scale.to(torch.float32)
        return self

    def forward(self,
                x: torch.Tensor,
                residual: torch.Tensor,
                scale: torch.Tensor = None) -> torch.Tensor:
        out = torch.empty_like(x, dtype=torch.int8)
        if self.use_per_token_dequant and scale is not None:
            ops.dequant_fused_add_rms_norm_quant(
                out, x, residual, self.weight.data,self.variance_epsilon, 
                scale, self.dequant_scale.item())
        else:
            ops.dequant_fused_add_rms_norm_quant(
                out, x, residual, self.weight.data, self.variance_epsilon,
                None, self.dequant_scale.item())
        return out, residual


class DequantAddResidual(nn.Module):
    def __init__(self,
                 dequant_scale: float = 1.0,
                 use_per_token_dequant: bool = True) -> None:
        super().__init__()
        self.register_parameter(
            "dequant_scale",
            torch.nn.Parameter(torch.tensor(dequant_scale,dtype=torch.float32,requires_grad=False))
        )
        self.use_per_token_dequant = use_per_token_dequant

    def _apply(self, fn):
        super()._apply(fn)
        self.dequant_scale.data = self.dequant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dequant_scale.data = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale.data = self.dequant_scale.to(torch.float32)
        return self

    def forward(self,
                x: torch.Tensor,
                residual: torch.Tensor,
                scale: torch.Tensor = None) -> torch.Tensor:
        out = torch.empty_like(residual)
        if self.use_per_token_dequant and scale is not None:
            ops.dequant_add_residual(out, x, residual, scale, self.dequant_scale.item())
        else:
            ops.dequant_add_residual(out, x, residual, None, self.dequant_scale.item())
        return out


class AddResidual(DequantAddResidual):
    def __init__(self,
                 dequant_scale: float = 1.0,
                 use_per_token_dequant: bool = True):
        super().__init__(dequant_scale,use_per_token_dequant)
    
    def forward(self,
                x: torch.Tensor,
                residual: torch.Tensor,
                scale: torch.Tensor = None) -> torch.Tensor:
        return x + residual
