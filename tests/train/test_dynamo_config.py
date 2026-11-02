"""Configuration boundaries for the certified Dynamo integration profile."""

import ast
import inspect
import re
import secrets

import pytest

from skyrl.train.config import DynamoInferenceConfig, SkyRLTrainConfig
from skyrl.train.utils import utils as train_utils
from skyrl.train.utils.utils import validate_inference_engine_cfg


def configured():
    return SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.strategy=fsdp",
            "trainer.placement.colocate_all=false",
            "generator.inference_engine.run_engines_locally=false",
            "generator.inference_engine.external_proxy_url=http://127.0.0.1:8000",
            "generator.inference_engine.dynamo.namespace=skyrl-test",
            "generator.inference_engine.weight_sync_backend=ipc",
        ]
    )


def test_nullable_typed_config_and_independent_default_hosts():
    assert SkyRLTrainConfig().generator.inference_engine.dynamo is None
    cfg = configured()
    assert isinstance(cfg.generator.inference_engine.dynamo, DynamoInferenceConfig)
    assert cfg.generator.inference_engine.dynamo.auth_token_env == "DYN_SYSTEM_ENGINE_AUTH_TOKEN"
    validate_inference_engine_cfg(cfg)
    left, right = DynamoInferenceConfig(), DynamoInferenceConfig()
    left.allowed_hosts.append("example.test")
    assert "example.test" not in right.allowed_hosts


@pytest.mark.parametrize(
    "path,value",
    [
        ("generator.inference_engine.run_engines_locally", True),
        ("generator.inference_engine.external_proxy_url", None),
        ("generator.inference_engine.external_server_urls", []),
        ("trainer.placement.colocate_all", True),
        ("trainer.fully_async.enabled", True),
        ("trainer.strategy", "megatron"),
        ("trainer.policy.model.lora.rank", 8),
        ("generator.inference_engine.enable_pd", True),
        ("generator.inference_engine.tensor_parallel_size", 2),
        ("generator.inference_engine.pipeline_parallel_size", 2),
        ("generator.inference_engine.data_parallel_size", 2),
        ("generator.inference_engine.expert_parallel_size", 2),
        ("generator.inference_engine.weight_sync_backend", "delta"),
        ("generator.inference_engine.enable_return_routed_experts", True),
        ("generator.inference_engine.enable_return_sample_support_set", True),
        ("trainer.algorithm.enable_score_centering", True),
        ("generator.inference_engine.speculative_config", {"method": "mtp"}),
        ("generator.inference_engine.fp8_weight_sync_mode", "blockwise"),
        ("generator.inference_engine.engine_init_kwargs", {"quantization": "fp8"}),
        ("generator.inference_engine.offload_kv_for_weight_sync", True),
        ("generator.inference_engine.dynamo.namespace", ""),
        ("generator.inference_engine.dynamo.expected_workers", 0),
        ("generator.inference_engine.dynamo.discovery_timeout_s", 0),
        ("generator.inference_engine.dynamo.control_timeout_s", float("inf")),
        ("generator.inference_engine.dynamo.request_timeout_s", float("nan")),
        ("generator.inference_engine.dynamo.allowed_hosts", []),
        ("generator.inference_engine.dynamo.allowed_hosts", ["external.example"]),
        ("generator.inference_engine.dynamo.auth_token_env", "token-value-with-dashes"),
        ("generator.inference_engine.dynamo.discovery_url", "http://untrusted.example"),
        ("generator.inference_engine.engine_init_kwargs", {"logprobs_mode": "raw_logprobs"}),
        ("generator.inference_engine.engine_init_kwargs", {"hf_overrides": {"quantization_config": {}}}),
        ("trainer.algorithm.enable_sample_support_replay", True),
    ],
)
def test_unsupported_profile_fails_before_side_effects(path, value):
    cfg = configured()
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)
    with pytest.raises(ValueError, match="Dynamo"):
        validate_inference_engine_cfg(cfg)


@pytest.mark.parametrize("env_name", ["DYN_SYSTEM_ENGINE_AUTH_TOKEN", "CUSTOM_DYNAMO_AUTH"])
def test_runtime_environment_inherits_secret_without_serializing_or_logging_it(monkeypatch, env_name):
    cfg = configured()
    cfg.generator.inference_engine.dynamo.auth_token_env = env_name
    secret = secrets.token_urlsafe(48)
    monkeypatch.setenv(env_name, secret)
    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **kwargs: True)
    messages = []
    monkeypatch.setattr(train_utils.logger, "info", messages.append)
    runtime_env = train_utils.prepare_runtime_environment(cfg)
    assert env_name not in runtime_env
    assert secret not in repr(runtime_env)
    assert secret not in "\n".join(messages)
    assert secret not in repr(cfg)


@pytest.mark.parametrize(
    "env_name",
    [
        "SKYRL_CUSTOM_TOKEN",
        "HF_TOKEN",
        "NCCL_DEBUG",
        "NCCL_NET_PLUGIN",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "VLLM_DISABLE_COMPILE_CACHE",
        "UV_CACHE_DIR",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES",
        "WANDB_API_KEY",
        "MLFLOW_TRACKING_TOKEN",
        "DAYTONA_API_KEY",
        "MODAL_TOKEN_SECRET",
        "RAY_AUTH_TOKEN",
    ],
)
def test_dynamo_rejects_auth_names_that_runtime_environment_forwards_or_logs(env_name):
    cfg = configured()
    cfg.generator.inference_engine.dynamo.auth_token_env = env_name
    with pytest.raises(ValueError, match="auth_token_env.*reserved"):
        validate_inference_engine_cfg(cfg)


def test_dynamo_rejects_enabled_annotation_token_name():
    cfg = configured()
    cfg.trainer.grafana_annotations.enabled = True
    cfg.trainer.grafana_annotations.token_env_var = "ANNOTATION_TOKEN"
    cfg.generator.inference_engine.dynamo.auth_token_env = "ANNOTATION_TOKEN"
    with pytest.raises(ValueError, match="auth_token_env.*reserved"):
        validate_inference_engine_cfg(cfg)


def test_auth_name_policy_covers_every_runtime_environment_literal():
    source = ast.parse(inspect.getsource(train_utils.prepare_runtime_environment))
    names = {
        node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"[A-Z][A-Za-z0-9_]*", node.value)
    }
    assert {"HF_TOKEN", "NCCL_DEBUG", "RAY_CGRAPH_get_timeout", "SKYRL_"} <= names
    for name in sorted(names):
        cfg = configured()
        cfg.generator.inference_engine.dynamo.auth_token_env = name
        with pytest.raises(ValueError, match="auth_token_env.*reserved"):
            validate_inference_engine_cfg(cfg)


def test_auth_name_collision_is_rejected_before_logs_or_platform_probes(monkeypatch):
    cfg = configured()
    cfg.generator.inference_engine.dynamo.auth_token_env = "NCCL_NET_PLUGIN"
    monkeypatch.setenv("NCCL_NET_PLUGIN", secrets.token_urlsafe(48))
    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.setattr(train_utils.logger, "info", lambda message: pytest.fail("logged before validation"))
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **kwargs: pytest.fail("probed before validation"))
    with pytest.raises(ValueError, match="auth_token_env.*reserved"):
        train_utils.prepare_runtime_environment(cfg)


@pytest.mark.parametrize("colocate_all", [False, True])
def test_explicit_ipc_is_not_available_to_unqualified_direct_profile(colocate_all):
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.generator.inference_engine.weight_sync_backend = "ipc"
    with pytest.raises(ValueError, match="explicit.*ipc.*Dynamo"):
        validate_inference_engine_cfg(cfg)


def test_colocated_logical_nccl_still_selects_internal_ipc():
    from skyrl.backends.skyrl_train.weight_sync import get_transfer_strategy

    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = True
    validate_inference_engine_cfg(cfg)
    assert get_transfer_strategy(cfg.generator.inference_engine.weight_sync_backend, True) == "ipc"


def test_dynamo_nccl_warns_that_transport_is_not_gpu_qualified():
    cfg = configured()
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    with pytest.warns(UserWarning, match="NCCL.*not GPU-qualified"):
        validate_inference_engine_cfg(cfg)


def test_direct_runtime_environment_forwarding_is_unchanged(monkeypatch):
    monkeypatch.setattr(
        train_utils.os,
        "environ",
        {
            "CUDA_HOME": "/runtime/cuda",
            "NCCL_NET_PLUGIN": "safe-plugin",
            "HF_TOKEN": "model-access-token",
            "UV_CACHE_DIR": "/cache/uv",
            "SKYRL_CUSTOM_OVERRIDE": "true",
        },
    )
    monkeypatch.setattr(train_utils, "SKYRL_LD_LIBRARY_PATH_EXPORT", False)
    monkeypatch.setattr(train_utils, "SKYRL_PYTHONPATH_EXPORT", False)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **kwargs: True)
    assert train_utils.prepare_runtime_environment(SkyRLTrainConfig()) == {
        "CUDA_HOME": "/runtime/cuda",
        "NCCL_NET_PLUGIN": "safe-plugin",
        "HF_TOKEN": "model-access-token",
        "UV_CACHE_DIR": "/cache/uv",
        "SKYRL_CUSTOM_OVERRIDE": "true",
        "NCCL_CUMEM_ENABLE": "0",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_USE_V1": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }


def test_runtime_policy_removes_both_credentials_without_mutating_input(monkeypatch):
    from skyrl.backends.skyrl_train.inference_servers.dynamo.launch import (
        validate_dynamo_environment,
    )

    cfg = configured()
    token_name = cfg.generator.inference_engine.dynamo.auth_token_env
    monkeypatch.setenv(token_name, secrets.token_urlsafe(48))
    monkeypatch.setenv("RAY_ADDRESS", "local")
    policy = validate_dynamo_environment(cfg)
    environment = {token_name: "engine-secret", "RAY_AUTH_TOKEN": "ray-secret", "SAFE_SETTING": "1"}
    assert policy.filter(environment) == {"SAFE_SETTING": "1"}
    assert len(environment) == 3


def test_environment_logs_only_names_and_preserves_forwarded_values(monkeypatch):
    marker = secrets.token_urlsafe(48)
    monkeypatch.setenv("NCCL_DEBUG", marker)
    monkeypatch.setenv("HF_TOKEN", marker)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **kwargs: True)
    messages = []
    monkeypatch.setattr(train_utils.logger, "info", messages.append)
    result = train_utils.prepare_runtime_environment(SkyRLTrainConfig())
    assert result["NCCL_DEBUG"] == result["HF_TOKEN"] == marker
    assert marker not in "\n".join(messages)
    assert any("NCCL_DEBUG" in message for message in messages)


@pytest.mark.parametrize("secret", [None, "short", "x" * 32 + "\n", "x" * 32 + " ", "x" * 32 + "\x7f"])
def test_runtime_environment_validates_dynamo_secret_before_platform_probes(monkeypatch, secret):
    cfg = configured()
    name = cfg.generator.inference_engine.dynamo.auth_token_env
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAY_ADDRESS", "local")
    if secret is not None:
        monkeypatch.setenv(name, secret)
    monkeypatch.setattr(
        train_utils,
        "peer_access_supported",
        lambda **kwargs: pytest.fail("token validation must precede platform probes"),
    )
    with pytest.raises(ValueError, match="token"):
        train_utils.prepare_runtime_environment(cfg)


@pytest.mark.parametrize("address", [None, "auto", "ray://127.0.0.1:10001", "127.0.0.1:6379"])
def test_dynamo_runtime_requires_an_owned_local_raylet(monkeypatch, address):
    cfg = configured()
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    if address is not None:
        monkeypatch.setenv("RAY_ADDRESS", address)
    monkeypatch.setenv(
        cfg.generator.inference_engine.dynamo.auth_token_env, "test-only-auth-012345678901234567890123456789"
    )
    with pytest.raises(ValueError, match="RAY_ADDRESS=local"):
        train_utils.prepare_runtime_environment(cfg)
