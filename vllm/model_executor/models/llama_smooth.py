# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import DequantSiluAndMulQuant
from vllm.model_executor.layers.attention import DequantPagedAttention
from vllm.model_executor.layers.layernorm import (RMSNorm, 
                                                  RMSNormQuant, 
                                                  AddResidualRMSNormQuant,
                                                  DequantAddResidualRMSNormQuant)

from vllm.model_executor.layers.quantization.smoothquant import SmoothLinearMethod

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               QuantMergedColumnParallelLinear,
                                               QuantQKVParallelLinear,
                                               QuantRowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_dequant_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, 
    ParallelLMHead)
from vllm.model_executor.layers.layernorm import DequantAddResidual, AddResidual
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class QuantLlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = QuantMergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method,
            skip_bias_add=True)
        self.down_proj = QuantRowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method,
                                           skip_bias_add=True)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = DequantSiluAndMulQuant()

    def forward(self, x):
        scale = None
        # int, half -> int32
        gate_up, _ = self.gate_up_proj(x)
        # int32 -> int, scale
        x, *scale = self.act_fn(gate_up)
        scale = scale[0] if scale is not None else None
        # int8, scale -> int32(when tp > 1, to half, scale for dequant before all reduce)
        x, _ = self.down_proj(x, scale)
        return x, scale


class QuantLlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
        
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QuantQKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
            skip_bias_add=True,
        )
        self.o_proj = QuantRowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
            skip_bias_add=True,
        )

        self.rotary_emb = get_dequant_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = DequantPagedAttention(self.num_heads,
                                   self.head_dim,
                                   self.scaling,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata
    ) -> torch.Tensor:
        # int8 -> int32
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # int32 -> half
        q, k, v = self.rotary_emb(positions, q, k, v, 
                               self.qkv_proj.q_dequant_scale.item(),
                               self.qkv_proj.k_dequant_scale.item(),
                               self.qkv_proj.v_dequant_scale.item())
        k_cache, v_cache = kv_cache
        scale = None
        # half - > int8, scale, 添加一个per channel 量化，并返回统计的scale
        attn_output, *scale = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        scale = scale[0] if scale is not None else None
        # int8, scale -> int32(when tp > 1, to half, scale for dequant before all reduce)
        output, _ = self.o_proj(attn_output, scale)
        return output, scale


class QuantLlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = QuantLlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
        )
        self.mlp = QuantLlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.apply_dequant_in_post = not linear_method.apply_dequant_after_row
        self.input_layernorm = RMSNormQuant(config.hidden_size,
                                       eps=config.rms_norm_eps)
        if self.apply_dequant_in_post:
            self.post_attention_layernorm = DequantAddResidualRMSNormQuant(config.hidden_size,
                                                    eps=config.rms_norm_eps)
            self.finally_add_residual = DequantAddResidual()
        else:
            self.post_attention_layernorm = AddResidualRMSNormQuant(config.hidden_size,
                                                    eps=config.rms_norm_eps)
            self.finally_add_residual = AddResidual()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # half
        residual = hidden_states
        # half -> int8
        hidden_states = self.input_layernorm(hidden_states)
        # int8 -> int32 ,scale (when tp > 1,to half, scale, this scale is useless) 
        hidden_states, scale = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata
        )

        # to = 1: int32, half, scale -> int8, half (scale for dequant)
        # tp > 1: half, half, scale -> int8, half
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual, scale)
        # int8 -> int32, scale (when tp > 1,to half, scale, this scale is useless) 
        hidden_states, scale = self.mlp(hidden_states)
        # ine32, half, scale -> half (when tp > 1, half, half, scale -> half)
        hidden_states = self.finally_add_residual(hidden_states, residual, scale)
        return hidden_states


class QuantLlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            QuantLlamaDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata
    ) -> torch.Tensor:
        # half
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata
            )
        # int32 , half, scale -> int8
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    
    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = QuantLlamaModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens
    
    
    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # process special params first
            ("qkv_proj.q_dequant_scale", "q_proj.dequant_scale", "-1"),
            ("qkv_proj.k_dequant_scale", "k_proj.dequant_scale", "-1"),
            ("qkv_proj.v_dequant_scale", "v_proj.dequant_scale", "-1"),
            ("act_fn.gate_dequant_scale", "gate_proj.dequant_scale", "-1"),
            ("act_fn.up_dequant_scale", "up_proj.dequant_scale", "-1"),
            
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        special_params_mapping = [
            ("post_attention_layernorm.dequant_scale", "self_attn.o_proj.dequant_scale"),
            ("finally_add_residual.dequant_scale","mlp.down_proj.dequant_scale")
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'bias' in name:
                    continue
                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader is default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight,shard_id)
                break
            else:
                for (param_name, weight_name) in special_params_mapping:
                    if weight_name not in name:
                        continue
                    # used in o_prof and down_proj when world_size > 1
                    if get_tensor_model_parallel_world_size() > 1:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        if weight_loader is default_weight_loader:
                            weight_loader(param, loaded_weight)
                        else:
                            weight_loader(param, loaded_weight,shard_id)
                    else:
                        param = params_dict[name.replace(weight_name, param_name)]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        if weight_loader is default_weight_loader:
                            weight_loader(param, loaded_weight)
                        else:
                            weight_loader(param, loaded_weight,shard_id)
                    break
                else:
                    if 'bias' not in name:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
