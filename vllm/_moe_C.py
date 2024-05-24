import torch
import ixformer.functions as ixf_F

def topk_softmax(topk_weights,topk_ids,token_expert_indicies,gating_output):
    ixf_F.vllm_moe_topk_softmax(topk_weights,topk_ids,token_expert_indicies,gating_output)
