from typing import Dict,Any

import torch
import torch.nn.functional as F

import ixformer
import ixformer.functions as ixf_F
from ixformer._C import ReduceOp
from ixformer._C import _distributed as cdist
from ixformer._C._distributed import is_initialized, get_default_comm_group
from ixformer.contrib.torch.extension import ixformer_torch as ixft
from ixformer.contrib.torch.data_type_mapping import torch_to_ixformer_dtype


class ops():
    # activations
    @staticmethod
    def silu_and_mul(output, x):
        ixf_F.silu_and_mul(x, output)

    @staticmethod
    def gelu_and_mul(output, x):
        ixf_F.gelu_and_mul(x, output)

    @staticmethod
    def gelu_new(output, x): 
        output.copy_(F.gelu(x,approximate="tanh"))
        return output

    @staticmethod
    def gelu_fast(output, x):
        output.copy_(F.gelu(x,approximate="tanh"))
        return output

    # rms norm
    @staticmethod
    def rms_norm(output, x, weight, epsilon):
        ixf_F.rms_norm(x, weight, output, epsilon)

    @staticmethod
    def fused_add_rms_norm(input, residual, weight, epsilon, scale):
        ixf_F.fused_add_rms_norm(input, residual, weight, epsilon, scale)

    # rotary embedding
    @staticmethod
    def rotary_embedding(positions, query, key, head_size,
                         cos_sin_cache, is_neox_style):
        ixf_F.vllm_rotary_embedding_neox(positions, query, key, head_size,
                                         cos_sin_cache, is_neox_style)

    # paged attention
    @staticmethod
    def paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes=None,
        kv_cache_dtype=None,
    ):
        return ixf_F.vllm_single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )

    @staticmethod
    def paged_attention_v2(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes=None,
        kv_cache_dtype=None,
        use_sqrt_alibi=False,
    ):
        return ixf_F.vllm_single_query_cached_kv_attention_v2(
            output,
            256,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
            use_sqrt_alibi,
        )

    # awq
    @staticmethod
    def awq_gemm(x, qweight, scales, qzeros, pack_factor):
        return ixf_F.quantized_linear(x,qweight,scales,"awq",32 // pack_factor,qzeros,None,group_size=128)

    @staticmethod
    def awq_dequantize(qweight, scales, qzeros, holder1, holder2, holder3):
        raise NotImplementedError()

    # gqt-q
    @staticmethod
    def gptq_shuffle(qweights,g_idx,weight_bits):
        return ixf_F.vllm_gptq_shuffle(qweights,g_idx)

    @staticmethod
    def gptq_gemm(x, qweight, qzeros, scales, idx, status, weight_bits):
        batch = x.shape[0]
        if batch <= 8:
            return ixf_F.quantized_linear(x,qweight,scales,"gptq",4,qzeros,None,group_size=128)
        o_dtype_str = "fp16" if x.dtype == torch.half else "bf16"
        deq_w = ixf_F.quantized_weight_dequant(qweight,scales,"gptq",o_dtype_str,4,qzeros,group_size=128)
        return torch.matmul(x,deq_w)

    # squeezellm
    @staticmethod
    def squeezellm_gemm(reshaped_x, qweight, out_f, lookup_table):
        raise NotImplementedError()

    # marlin
    @staticmethod
    def marlin_gemm(x_2d, qweight, scales, workspace, size_m, size_n, size_k):
        raise NotImplementedError()

    # moe
    @staticmethod
    def moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad):
        block_size = 8 if topk_ids.shape[0] * topk_ids.shape[1] <= 8 else 64
        ixformer.functions.vllm_moe_align_block_size(
            topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
        )

    @staticmethod
    def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int,
                            config: Dict[str, Any]) -> None:
        block_size = 8 if topk_ids.shape[0] * topk_ids.shape[1] <= 8 else 64
        ixformer.functions.vllm_invoke_fused_moe_kernel(A,B,C,topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded,mul_routed_weight,top_k,block_size)

    # smoothquant    
    @staticmethod
    def quant(output,input,scale):
        ixf_F.vllm_smooth_quant(output,input,scale)
        return output

    @staticmethod
    def dequant(output,x,scale,global_scale):
        ixf_F.vllm_smooth_dequant(output,x,scale,global_scale)
        return output

    @staticmethod
    def dequant_add_residual(output,x,residual,scale,global_scale):
        if isinstance(x,torch.Tensor):
            ixf_F.vllm_smooth_dequant_add_residual(output,x,residual,scale,global_scale)
        return output

    @staticmethod
    def dequant_silu_and_mul_quant(output,x,gate_scale, up_scale, scale, temp = None):
        ixf_F.vllm_smooth_dequant_silu_and_mul_quant(output,x,gate_scale, up_scale, scale, temp)

    @staticmethod
    def rms_norm_quant(output, input, weight, epsilon):
        return ixf_F.vllm_smooth_rms_norm_quant(output, input, weight, epsilon)

    @staticmethod
    def fused_add_rms_norm_quant(output, input, residual, weight, epsilon):
        ixf_F.vllm_smooth_fused_add_rms_norm_quant(output, input, residual, weight, epsilon)

    @staticmethod
    def dequant_fused_add_rms_norm_quant(output, input, residual, weight, epsilon, scale, global_scale):
        ixf_F.vllm_smooth_dequant_fused_add_rms_norm_quant(output, input, residual, weight, epsilon, scale, global_scale)

    @staticmethod
    def dequant_rotary_embedding(positions, query, key, head_size,
                        cos_sin_cache, query_out, key_out, query_scale, key_scale, is_neox_style):
        ixf_F.vllm_smooth_dequant_rotary_embedding_neox(positions, query, key, head_size,
                        cos_sin_cache, query_out, key_out, query_scale, key_scale, is_neox_style)

    @staticmethod
    def linear_a8_w8_o32_(x, weight, output):
        return ixf_F.linear_i8w8o32(x,weight,output)


class cache_ops():
    
    @staticmethod
    def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
        ixf_F.vllm_cache_ops_reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping
        )

    @staticmethod
    def copy_blocks(key_caches, value_caches, block_mapping):
        ixf_F.vllm_copy_cache(
            key_caches, value_caches, block_mapping
        )

    @staticmethod
    def swap_blocks(src_key_cache, dst_key_cache, src_to_dst):
        ixf_F.vllm_swap_blocks(
            src_key_cache, dst_key_cache, src_to_dst
        )

class custom_ar():

    IS_INIT:bool = False

    @staticmethod
    def is_init():
        return_status = custom_ar.IS_INIT
        custom_ar.IS_INIT = True
        return return_status

    @staticmethod
    def init_cumtom_ar():
        if not is_initialized(get_default_comm_group()):
            group = ixft.create_ixformer_group_from_pg()
            ixformer.cuda.set_device(torch.cuda.current_device())
            cdist.update_default_comm_group(group)
        cdist.ipc.init_communicator_by_nccl()

    @staticmethod
    def all_reduce_reg(ptr,tensor,out = None):
        raise NotImplementedError()

    @staticmethod
    def all_reduce_unreg(ptr,tensor,buffer,out = None):
        dtype = tensor.dtype
        if torch.is_tensor(tensor):
            dtype = torch_to_ixformer_dtype(dtype)

        if out is None:
            out = tensor
        cdist.ipc.allreduce(
            tensor.data_ptr(), out.data_ptr(), dtype, tensor.numel(), ReduceOp.SUM
        )
        return out

    @staticmethod
    def dispose():
        ixformer.distributed.destroy_process_group()

    @staticmethod
    def should_custom_ar(tensor:torch.Tensor, max_size, world_size, full_nvlink):
        return cdist.ipc.should_custom_ar(tensor.numel(),tensor.element_size(),max_size,world_size)

class cuda_utils():
    @staticmethod
    def get_max_shared_memory_per_block_device_attribute(gpu):
        return 100000000
