# coding=utf-8
# Copyright 2022 The OpenBMB team.
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

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from typing_extensions import TypedDict

import math


class CPMDragonflyConfig(PretrainedConfig):
    model_type = "cpm_dragonfly"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "scale_emb": "scale_emb",
        "scale_depth": "scale_depth",
        "scale": "scale",
        "attention_scale": "attention_scale",
        "qk_norm": "qk_norm",
        "ffn_gated": "ffn_gated",
    } # model specific to common

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        dim_head=128,
        intermediate_size=11008,
        num_hidden_layers=32,
        dropout_p=0.0,
        hidden_act="silu",
        scale=True,
        scale_emb: float=1.,
        scale_depth: float=-1,
        dim_model_base:int=None,
        rms_norm_eps=1e-5,
        init_std=0.02,
        half: bool = True,
        half_type = 'bf16',
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
        use_flash_attn: bool = True,
        flash_attn_mask_shape="1d",
        flash_impl="cuda",
        base=10000,
        non_checkpointing_layers_num:int = 0,
        attention_scale=1,
        qk_norm=False,
        ffn_gated=True,
        tie_lm_head=False,
        max_position_embeddings=2048,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.dim_head = dim_head
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.dropout_p = dropout_p
        self.hidden_act = hidden_act
        self.scale = scale
        self.scale_emb = scale_emb
        self.half = half
        self.half_type = half_type
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.rms_norm_eps = rms_norm_eps
        self.init_std = init_std
        self.flash_impl = flash_impl
        self.mask_modules = mask_modules
        self.use_flash_attn = use_flash_attn
        self.flash_attn_mask_shape = flash_attn_mask_shape
        self.base = base
        self.attention_scale=attention_scale
        self.qk_norm = qk_norm
        self.ffn_gated = ffn_gated
        self.non_checkpointing_layers_num = non_checkpointing_layers_num
        self.tie_lm_head = tie_lm_head
        self.use_bfloat16 = True if self.half_type == 'bf16' else False
        super().__init__(architectures=["CPMDragonflyForCausalLM"])
    
    @property
    def scale_width(self,):
        if self.scale:
            return self.hidden_size / self.dim_model_base
        else:
            return 1.
    
    @property
    def scale_states(self,):
        if self.scale:
            return self.scale_depth / math.sqrt(self.num_hidden_layers)
        else:
            return 1.