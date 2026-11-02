"""Configuration and credential boundaries for an owned local Dynamo RL run."""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Literal, TypedDict

from skyrl.train.config import SkyRLTrainConfig

from .discovery import is_loopback_host, validate_destination_url

# These names are forwarded, logged, or consumed as launch settings rather than secrets.
_RESERVED_AUTH_ENV_NAMES = frozenset(
    {
        "CUDA_HOME",
        "NCCL_CUMEM_ENABLE",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "FLA_TILELANG",
        "NVTE_FUSED_ATTN",
        "NVTE_DEBUG",
        "NVTE_DEBUG_LEVEL",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
        "VLLM_ALLOW_INSECURE_SERIALIZATION",
        "VLLM_DISABLE_COMPILE_CACHE",
        "VLLM_USE_V1",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "RAY_CGRAPH_get_timeout",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_NET_PLUGIN",
        "WANDB_API_KEY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_TOKEN",
        "DAYTONA_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "UV_CACHE_DIR",
        "UV_LINK_MODE",
        "UV_PYTHON",
        "UV_OFFLINE",
        "HF_HOME",
        "HF_TOKEN",
        "HF_HUB_OFFLINE",
        "HF_ENDPOINT",
        "PYTORCH_CUDA_ALLOC_CONF",
        "VLLM_ALLREDUCE_USE_FLASHINFER",
        "VLLM_ALLREDUCE_USE_SYMM_MEM",
        "OMP_NUM_THREADS",
        "CUDA_LAUNCH_BLOCKING",
        "PYTHONFAULTHANDLER",
        "TORCH_SHOW_CPP_STACKTRACES",
        "TORCH_USE_CUDA_DSA",
        "NCCL_DEBUG",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES",
        "VLLM_USE_DEEP_GEMM_E8M0",
        "RAY_ADDRESS",
        "RAY_AUTH_MODE",
        "RAY_AUTH_TOKEN",
        "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER",
    }
)


class RayInitKwargs(TypedDict, total=False):
    address: Literal["local"]
    _node_ip_address: str
    include_dashboard: bool


@dataclass(frozen=True)
class RayLaunch:
    """Validated Ray ownership policy; credentials stay in the parent environment."""

    owned: bool = False

    @property
    def init_kwargs(self) -> RayInitKwargs:
        if not self.owned:
            return {}
        return {"address": "local", "_node_ip_address": "127.0.0.1", "include_dashboard": False}


@dataclass(frozen=True)
class RuntimeEnvironmentPolicy:
    """Exclude parent-inherited credentials from Ray's persisted runtime metadata."""

    protected_names: frozenset[str] = frozenset()

    def filter(self, environment: dict[str, str]) -> dict[str, str]:
        return {name: value for name, value in environment.items() if name not in self.protected_names}


def _validate_auth_env_name(cfg: SkyRLTrainConfig) -> None:
    dyn = cfg.generator.inference_engine.dynamo
    if dyn is None:
        return
    name = dyn.auth_token_env
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Dynamo auth_token_env must name an environment variable")
    annotation_cfg = cfg.trainer.grafana_annotations
    if (
        name.startswith("SKYRL_")
        or name in _RESERVED_AUTH_ENV_NAMES
        or (annotation_cfg.enabled and name == annotation_cfg.token_env_var)
    ):
        raise ValueError("Dynamo auth_token_env uses a reserved runtime environment name")


def validate_dynamo_config(cfg: SkyRLTrainConfig) -> None:
    """Reject unsupported profiles and unsafe destinations before worker mutation."""
    ie_cfg = cfg.generator.inference_engine
    if ie_cfg.dynamo is None:
        if ie_cfg.weight_sync_backend == "ipc":
            raise ValueError("explicit weight_sync_backend='ipc' requires the qualified external Dynamo profile")
        return
    _validate_profile(cfg)
    _validate_destinations_and_deadlines(cfg)
    if ie_cfg.weight_sync_backend == "nccl":
        warnings.warn(
            "Dynamo NCCL weight transfer is accepted but not GPU-qualified; the qualified local profile uses weight_sync_backend='ipc'.",
            UserWarning,
            stacklevel=2,
        )


def _validate_profile(cfg: SkyRLTrainConfig) -> None:
    ie_cfg = cfg.generator.inference_engine
    unsupported = {
        "backend must be vllm": ie_cfg.backend != "vllm",
        "run_engines_locally must be false": ie_cfg.run_engines_locally,
        "external_proxy_url is required": not ie_cfg.external_proxy_url,
        "external_server_urls must be unset": ie_cfg.external_server_urls is not None,
        "colocate_all is unsupported": cfg.trainer.placement.colocate_all,
        "fully asynchronous training is unsupported": cfg.trainer.fully_async.enabled,
        "trainer.strategy must be fsdp": cfg.trainer.strategy != "fsdp",
        "LoRA is unsupported": cfg.trainer.policy.model.lora.rank != 0,
        "PD disaggregation is unsupported": ie_cfg.enable_pd
        or bool(ie_cfg.prefill_init_kwargs or ie_cfg.decode_init_kwargs),
        "TP, PP, DP and EP must be one": any(
            size != 1
            for size in (
                ie_cfg.tensor_parallel_size,
                ie_cfg.pipeline_parallel_size,
                ie_cfg.data_parallel_size,
                ie_cfg.expert_parallel_size,
            )
        ),
        "weight transport must be nccl or ipc": ie_cfg.weight_sync_backend not in {"nccl", "ipc"},
        "replay and sample-support metadata are unsupported": (
            ie_cfg.enable_return_routed_experts
            or ie_cfg.enable_return_sample_support_set
            or cfg.trainer.algorithm.enable_score_centering
            or cfg.trainer.algorithm.enable_sample_support_replay
        ),
        "speculative decoding is unsupported": ie_cfg.speculative_config is not None,
        "BF16 full-weight sync is required": ie_cfg.model_dtype != "bfloat16"
        or ie_cfg.fp8_weight_sync_mode is not None,
        "KV offload during weight sync is unsupported": ie_cfg.offload_kv_for_weight_sync,
    }
    for reason, invalid in unsupported.items():
        if invalid:
            raise ValueError(f"Dynamo integration: {reason}")
    kwargs = ie_cfg.engine_init_kwargs or {}
    if any(kwargs.get(key) is not None for key in ("quantization", "speculative_config", "kv_transfer_config")):
        raise ValueError(
            "Dynamo integration does not support quantization, speculative or KV-transfer engine overrides"
        )
    if (kwargs.get("hf_overrides") or {}).get("quantization_config") is not None:
        raise ValueError("Dynamo integration does not support quantized model overrides")
    if kwargs.get("logprobs_mode", "processed_logprobs") != "processed_logprobs":
        raise ValueError("Dynamo integration requires processed_logprobs")


def _validate_destinations_and_deadlines(cfg: SkyRLTrainConfig) -> None:
    ie_cfg = cfg.generator.inference_engine
    dyn = ie_cfg.dynamo
    assert dyn is not None
    if not isinstance(dyn.namespace, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", dyn.namespace):
        raise ValueError("Dynamo integration requires an exact nonempty run namespace")
    if type(dyn.expected_workers) is not int or dyn.expected_workers < 1:
        raise ValueError("Dynamo expected_workers must be a positive integer")
    if (
        not isinstance(dyn.allowed_hosts, list)
        or not dyn.allowed_hosts
        or any(not isinstance(host, str) or not host for host in dyn.allowed_hosts)
    ):
        raise ValueError("Dynamo allowed_hosts must contain nonempty destination hosts")
    if ie_cfg.weight_sync_backend == "ipc" and not all(is_loopback_host(host) for host in dyn.allowed_hosts):
        raise ValueError("Dynamo IPC administration requires loopback-only allowed_hosts")
    _validate_auth_env_name(cfg)
    for name in ("discovery_timeout_s", "control_timeout_s", "request_timeout_s"):
        value = getattr(dyn, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Dynamo {name} must be finite and positive")
    validate_destination_url(dyn.discovery_url, dyn.allowed_hosts)
    validate_destination_url(ie_cfg.external_proxy_url, dyn.allowed_hosts)


def validate_dynamo_environment(cfg: SkyRLTrainConfig) -> RuntimeEnvironmentPolicy:
    """Validate inherited credentials before any environment logging or platform probe."""
    dyn = cfg.generator.inference_engine.dynamo
    if dyn is None:
        return RuntimeEnvironmentPolicy()
    from .protocol import read_engine_token

    _validate_auth_env_name(cfg)
    if os.environ.get("RAY_ADDRESS") != "local":
        raise ValueError("Dynamo requires a newly owned local raylet with RAY_ADDRESS=local")
    read_engine_token(dyn.auth_token_env)
    return RuntimeEnvironmentPolicy(frozenset({dyn.auth_token_env, "RAY_AUTH_TOKEN"}))


def owned_ray_launch(cfg: SkyRLTrainConfig) -> RayLaunch:
    """Require authenticated, loopback-only Ray before launching an owned raylet."""
    dyn = cfg.generator.inference_engine.dynamo
    if dyn is None:
        return RayLaunch()
    import ray
    from ray._private import ray_constants

    from .protocol import read_engine_token

    validate_dynamo_environment(cfg)
    if ray.is_initialized():
        raise ValueError("Dynamo requires a newly owned local raylet with RAY_ADDRESS=local")
    if os.environ.get("RAY_AUTH_MODE") != "token":
        raise ValueError("Dynamo requires RAY_AUTH_MODE=token before starting its owned raylet")
    if os.environ.get("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER") != "0" or ray_constants.ENABLE_RAY_CLUSTER:
        raise ValueError("Dynamo requires RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0 before importing Ray")
    if read_engine_token("RAY_AUTH_TOKEN") == read_engine_token(dyn.auth_token_env):
        raise ValueError("Dynamo administration and Ray require independent authentication tokens")
    return RayLaunch(owned=True)
