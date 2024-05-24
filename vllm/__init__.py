"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""
import os

# By default, to avoid memory fragmentation, disable UMD mempool
if os.getenv("UMD_ENABLEMEMPOOL") is None:
    os.environ["UMD_ENABLEMEMPOOL"] = "0"
os.environ["NCCL_FORCESYNC_DISABLE"] = "1"

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster
from vllm.entrypoints.llm import LLM
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

__version__ = "0.3.3"

__all__ = [
    "LLM",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_cluster",
]
