"""Tinker sampling is an independent producer, outside the strict RL cohort."""

from unittest.mock import Mock

import pytest

import skyrl.backends.skyrl_train_backend as backend_module
from skyrl.backends.skyrl_train.inference_servers import setup
from skyrl.backends.skyrl_train.inference_servers.dynamo import discovery
from skyrl.backends.skyrl_train.weight_sync import get_transfer_strategy


@pytest.mark.parametrize(
    "overrides",
    [
        {"generator.inference_engine.dynamo": {}},
        {"generator.inference_engine.dynamo.namespace": "training"},
        {"generator": {"inference_engine": {"dynamo": {"namespace": "training"}}}},
    ],
)
def test_tinker_rejects_dynamo_before_ray_or_discovery(monkeypatch, overrides):
    initialize = Mock(side_effect=AssertionError("initialized Ray"))
    discover = Mock(side_effect=AssertionError("discovered fleet"))
    monkeypatch.setattr(backend_module, "initialize_ray", initialize)
    monkeypatch.setattr(discovery, "discover_worker_snapshot", discover)
    with pytest.raises(ValueError, match="Dynamo.*Tinker"):
        backend_module._build_skyrl_train_config("Qwen/Qwen3-0.6B", backend_module.FSDPBackendOverrides(**overrides))
    initialize.assert_not_called()
    discover.assert_not_called()


@pytest.mark.parametrize("colocated,transport", [(False, "nccl"), (True, "ipc")])
@pytest.mark.parametrize(
    "overrides",
    [{}, {"generator.inference_engine.weight_sync_backend": "nccl"}, {"generator.inference_engine.dynamo": None}],
)
def test_tinker_native_defaults_nccl_and_null_keep_placement_selection(overrides, colocated, transport):
    config = backend_module._build_skyrl_train_config(
        "Qwen/Qwen3-0.6B",
        backend_module.FSDPBackendOverrides(**{"trainer.placement.colocate_all": colocated, **overrides}),
    )
    assert config.generator.inference_engine.dynamo is None
    assert config.generator.inference_engine.weight_sync_backend == "nccl"
    assert config.trainer.placement.colocate_all is colocated
    assert get_transfer_strategy(config.generator.inference_engine.weight_sync_backend, colocated) == transport


@pytest.mark.parametrize("colocated", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"generator.inference_engine.weight_sync_backend": "ipc"},
        {"generator": {"inference_engine": {"weight_sync_backend": "ipc"}}},
    ],
)
def test_tinker_explicit_ipc_rejects_before_ray_and_client_construction(monkeypatch, overrides, colocated):
    initialize = Mock(side_effect=lambda *a: pytest.fail("initialized unsupported native IPC"))
    construct = Mock(side_effect=lambda *a, **kw: pytest.fail("constructed unsupported native IPC"))
    discover = Mock(side_effect=lambda *a: pytest.fail("discovered unsupported native IPC"))
    monkeypatch.setattr(backend_module, "initialize_ray", initialize)
    monkeypatch.setattr(setup, "build_new_inference_client", construct)
    monkeypatch.setattr(discovery, "discover_worker_snapshot", discover)
    with pytest.raises(ValueError, match="explicit.*ipc.*Dynamo"):
        backend_module._build_skyrl_train_config(
            "Qwen/Qwen3-0.6B",
            backend_module.FSDPBackendOverrides(**{"trainer.placement.colocate_all": colocated, **overrides}),
        )
    for boundary in (initialize, construct, discover):
        boundary.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"generator.inference_engine.weight_sync_backend": "ipc"},
        {"generator": {"inference_engine": {"weight_sync_backend": "ipc"}}},
    ],
)
def test_tinker_inference_startup_rejects_explicit_ipc_before_ray(monkeypatch, overrides):
    backend = object.__new__(backend_module.SkyRLTrainBackend)
    backend.base_model = "Qwen/Qwen3-0.6B"
    backend.config = backend_module.FSDPBackendOverrides(runtime_role="inference", **overrides)
    backend._cfg = None
    backend._inference_engines_initialized = False
    initialize = Mock(side_effect=lambda *a: pytest.fail("initialized unsupported native IPC"))
    construct = Mock(side_effect=lambda *a: pytest.fail("constructed unsupported native IPC"))
    monkeypatch.setattr(backend_module.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(backend_module, "initialize_ray", initialize)
    backend._create_new_inference_client = construct

    with pytest.raises(ValueError, match="explicit.*ipc.*Dynamo"):
        backend._ensure_inference_engines()

    initialize.assert_not_called()
    construct.assert_not_called()
