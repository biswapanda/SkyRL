"""CPU tests for Dynamo setup, trainer initialization, and update ownership."""

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.backends.skyrl_train.inference_servers import setup
from skyrl.backends.skyrl_train.inference_servers.dynamo import discovery
from skyrl.backends.skyrl_train.inference_servers.dynamo.client import (
    DynamoInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.weight_sync.control_spec import (
    DirectWeightSyncControlSpec,
)
from skyrl.backends.skyrl_train.workers.worker import Worker
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import (
    DynamoInferenceConfig,
    InferenceEngineConfig,
    SkyRLTrainConfig,
)


@pytest.fixture
def dynamo_setup(monkeypatch):
    events = []
    snapshot = SimpleNamespace(workers=(SimpleNamespace(system_url="http://127.0.0.1:9010"),))

    def discover(config, model_name):
        events.append(("discover", config, model_name))
        return snapshot

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.proxy_url = kwargs["proxy_url"]
            self.server_urls = kwargs["server_urls"]

        async def prepare_for_initialization(self):
            events.append("initial-pause")

    module = ModuleType("skyrl.backends.skyrl_train.inference_servers.dynamo.client")
    module.DynamoInferenceClient = Client
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(discovery, "discover_worker_snapshot", discover)
    monkeypatch.setattr(setup, "create_inference_servers", lambda *a, **kw: pytest.fail("launched servers"))
    config = SkyRLTrainConfig()
    config.trainer.placement.colocate_all = False
    config.generator.inference_engine = InferenceEngineConfig(
        dynamo=DynamoInferenceConfig(namespace="training"),
        external_proxy_url="http://127.0.0.1:8000",
        served_model_name="served-policy",
        weight_sync_backend="ipc",
        run_engines_locally=False,
    )
    monkeypatch.setattr(setup, "build_vllm_cli_args", lambda *a: pytest.fail("built local server arguments"))
    return config, snapshot, Client, events


def test_dynamo_setup_discovers_and_pauses_before_returning(dynamo_setup):
    config, snapshot, client_class, events = dynamo_setup
    client, servers = setup.build_new_inference_client(config, tokenizer="tokenizer")
    ie_cfg = config.generator.inference_engine
    assert isinstance(client, client_class)
    assert events == [("discover", ie_cfg.dynamo, "served-policy"), "initial-pause"]
    assert client.kwargs["cfg"] is ie_cfg.dynamo
    assert client.kwargs["snapshot"] is snapshot
    assert client.kwargs["transport"] == "ipc"
    assert client.kwargs["model_name"] == "served-policy"
    assert client.kwargs["tokenizer"] == "tokenizer"
    assert servers.server_urls == ["http://127.0.0.1:9010"]
    assert servers.router is None
    assert servers.server_groups == []


def test_dynamo_setup_propagates_initial_pause_failure(dynamo_setup, monkeypatch):
    config, _, client_class, _ = dynamo_setup
    monkeypatch.setattr(client_class, "prepare_for_initialization", AsyncMock(side_effect=RuntimeError("pause failed")))
    with pytest.raises(RuntimeError, match="pause failed"):
        setup.build_new_inference_client(config, tokenizer=None)


@pytest.mark.parametrize("unsupported", ["colocation", "lora"])
def test_dynamo_policy_gate_precedes_discovery_and_pause(dynamo_setup, monkeypatch, unsupported):
    config, _, client_class, events = dynamo_setup
    if unsupported == "colocation":
        config.trainer.placement.colocate_all = True
        reason = "colocate_all"
    else:
        config.trainer.policy.model.lora.rank = 8
        reason = "LoRA"
    monkeypatch.setattr(discovery, "discover_worker_snapshot", lambda *a: pytest.fail("discovered invalid profile"))
    monkeypatch.setattr(
        client_class, "prepare_for_initialization", AsyncMock(side_effect=AssertionError("paused invalid profile"))
    )
    with pytest.raises(ValueError, match=reason):
        setup.build_new_inference_client(config, tokenizer=None)
    client_class.prepare_for_initialization.assert_not_awaited()
    assert events == []


@pytest.mark.parametrize("colocated", [False, True])
def test_direct_explicit_ipc_gate_precedes_all_construction(monkeypatch, colocated):
    config = SkyRLTrainConfig()
    config.trainer.placement.colocate_all = colocated
    config.generator.inference_engine.weight_sync_backend = "ipc"
    assert config.generator.inference_engine.dynamo is None
    native_calls = []
    for name in ("build_new_inference_client_impl", "create_inference_servers", "build_vllm_cli_args"):
        boundary = MagicMock(side_effect=lambda *a, **kw: pytest.fail("constructed unsupported native IPC"))
        monkeypatch.setattr(setup, name, boundary)
        native_calls.append(boundary)
    discover = MagicMock(side_effect=lambda *a: pytest.fail("discovered unsupported native IPC"))
    pause = AsyncMock(side_effect=AssertionError("paused unsupported native IPC"))
    monkeypatch.setattr(discovery, "discover_worker_snapshot", discover)
    monkeypatch.setattr(DynamoInferenceClient, "prepare_for_initialization", pause)

    with pytest.raises(ValueError, match="explicit.*ipc.*Dynamo"):
        setup.build_new_inference_client(config, tokenizer=None)

    for boundary in [*native_calls, discover, pause]:
        boundary.assert_not_called()


def test_dynamo_initial_pause_precedes_trainer_factory(dynamo_setup, initialized_worker):
    config, _, _, events = dynamo_setup
    worker, _ = initialized_worker
    client, _ = setup.build_new_inference_client(config, tokenizer=None)
    client.data_parallel_size = 1
    client.dynamo_control_spec = object()
    client.get_world_size = AsyncMock(return_value=(1, 1))
    client.validate_trainer_device = AsyncMock(side_effect=lambda uuid: events.append("validate-device"))
    client.weight_sync_control_spec = MethodType(DynamoInferenceClient.weight_sync_control_spec, client)

    async def build(*args, **kwargs):
        events.append("trainer-factory")
        return "engine"

    worker._weight_sync_thread.side_effect = build
    asyncio.run(Worker.init_weight_sync_state(worker, client, config.generator.inference_engine))
    assert events[-3:] == ["initial-pause", "validate-device", "trainer-factory"]


@pytest.fixture
def initialized_worker(monkeypatch):
    module = ModuleType("skyrl.backends.skyrl_train.weight_sync.weight_senders")
    module.build_trainer_engine = MagicMock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.barrier", lambda: None)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 3)
    properties = MagicMock(return_value=SimpleNamespace(uuid="GPU-trainer"))
    monkeypatch.setattr("torch.cuda.get_device_properties", properties)
    worker = SimpleNamespace(
        cfg=SimpleNamespace(
            placement=SimpleNamespace(colocate_all=False),
            policy=SimpleNamespace(model=SimpleNamespace(path="policy")),
        ),
        _build_weight_source=MagicMock(),
        _build_draft_weight_source=MagicMock(),
        _weight_sync_thread=AsyncMock(return_value="engine"),
    )
    return worker, properties


@pytest.mark.asyncio
async def test_dynamo_worker_validates_actual_device_before_factory(initialized_worker):
    worker, properties = initialized_worker
    client = SimpleNamespace(
        server_urls=["http://127.0.0.1:9010"],
        data_parallel_size=1,
        dynamo_control_spec=object(),
        get_world_size=AsyncMock(return_value=(1, 1)),
        validate_trainer_device=AsyncMock(),
    )
    client.weight_sync_control_spec = MethodType(DynamoInferenceClient.weight_sync_control_spec, client)

    async def build(*args, **kwargs):
        client.validate_trainer_device.assert_awaited_once_with("GPU-trainer")
        assert kwargs["control_spec"] is client.dynamo_control_spec
        return "engine"

    worker._weight_sync_thread.side_effect = build
    await Worker.init_weight_sync_state(worker, client, InferenceEngineConfig(dynamo=DynamoInferenceConfig()))
    properties.assert_called_once_with(3)
    assert worker._weight_sync_engine == "engine"


@pytest.mark.asyncio
async def test_dynamo_device_validation_failure_prevents_factory(initialized_worker):
    worker, _ = initialized_worker
    client = SimpleNamespace(
        server_urls=[],
        data_parallel_size=1,
        dynamo_control_spec=object(),
        get_world_size=AsyncMock(return_value=(1, 1)),
        validate_trainer_device=AsyncMock(side_effect=ValueError("device mismatch")),
    )
    client.weight_sync_control_spec = MethodType(DynamoInferenceClient.weight_sync_control_spec, client)
    with pytest.raises(ValueError, match="device mismatch"):
        await Worker.init_weight_sync_state(worker, client, InferenceEngineConfig(dynamo=DynamoInferenceConfig()))
    worker._weight_sync_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_worker_factory_receives_default_control_spec(initialized_worker):
    worker, properties = initialized_worker
    client = RemoteInferenceClient("http://router:8000", ["http://direct:8000"], 1)
    client.get_world_size = AsyncMock(return_value=(1, 1))
    await Worker.init_weight_sync_state(worker, client, InferenceEngineConfig())
    assert "dynamo_control_spec" not in worker._weight_sync_thread.call_args.kwargs
    assert worker._weight_sync_thread.call_args.kwargs["control_spec"] == DirectWeightSyncControlSpec(
        ("http://direct:8000",), 1
    )
    properties.assert_called_once_with(3)


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamo", [False, True])
async def test_only_dynamo_suppresses_overlapping_worker_cache_reset(monkeypatch, dynamo):
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    worker = SimpleNamespace(_should_reset_prefix_cache=lambda config: True)
    client = SimpleNamespace(reset_prefix_cache=AsyncMock(), owns_weight_update_lifecycle=dynamo)
    config = InferenceEngineConfig(dynamo=DynamoInferenceConfig() if dynamo else None)
    reset = Worker._reset_prefix_cache_task(worker, client, config)
    if dynamo:
        assert reset is None
        client.reset_prefix_cache.assert_not_called()
    else:
        await reset
        client.reset_prefix_cache.assert_awaited_once_with(reset_running_requests=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_dynamo_dispatch_transaction_owns_resume_and_version(fails):
    events = []
    client = MagicMock()
    client.owns_weight_update_lifecycle = True
    client.pause_generation = AsyncMock()
    client.resume_generation = AsyncMock()
    client.version = 0

    @asynccontextmanager
    async def transaction():
        events.append("transaction-pause")
        try:
            yield
        except BaseException:
            events.append("transaction-failed")
            raise
        events.append("transaction-cache-reset")
        events.append("transaction-resume")
        client.version += 1

    client.weight_update_transaction = transaction
    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.colocate_all = False
    dispatch.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp", policy=SimpleNamespace(model=SimpleNamespace(lora=SimpleNamespace(rank=0)))
        ),
        generator=SimpleNamespace(
            inference_engine=SimpleNamespace(
                dynamo=DynamoInferenceConfig(), offload_kv_for_weight_sync=False, weight_sync_backend="ipc"
            )
        ),
    )
    dispatch._inference_engine_client = client
    dispatch._prepare_for_weight_sync = AsyncMock()
    dispatch.ensure_active_adapter = MagicMock()
    dispatch._finish_weight_sync = MagicMock(side_effect=lambda **kwargs: events.append("finish"))

    def broadcast(*args, **kwargs):
        events.append("broadcast")
        if fails:
            raise RuntimeError("send failed")

    dispatch._broadcast_to_inference_engines = broadcast
    if fails:
        with pytest.raises(RuntimeError, match="send failed"):
            await dispatch.save_weights_for_sampler()
        assert events == ["transaction-pause", "broadcast", "transaction-failed"]
        assert client.version == 0
    else:
        await dispatch.save_weights_for_sampler()
        assert events == ["transaction-pause", "broadcast", "finish", "transaction-cache-reset", "transaction-resume"]
        assert client.version == 1
        assert dispatch.last_weight_sync_seconds >= 0
    client.pause_generation.assert_not_awaited()
    client.resume_generation.assert_not_awaited()
    client.increment_weight_version.assert_not_called()


@pytest.mark.asyncio
async def test_dynamo_dispatch_cancellation_drains_sender_without_resuming():
    events = []
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    client = MagicMock()
    client.owns_weight_update_lifecycle = True
    client.resume_generation = AsyncMock()

    @asynccontextmanager
    async def transaction():
        events.append("pause")
        try:
            yield
        except BaseException:
            events.append("failed")
            raise
        events.append("resume")

    def broadcast(*args, **kwargs):
        events.append("broadcast")
        loop.call_soon_threadsafe(started.set)
        if not release.wait(5):
            raise RuntimeError("sender was not released")
        events.append("sender-complete")

    client.weight_update_transaction = transaction
    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.colocate_all = False
    dispatch.cfg = SimpleNamespace(
        generator=SimpleNamespace(inference_engine=SimpleNamespace(dynamo=DynamoInferenceConfig()))
    )
    dispatch._inference_engine_client = client
    dispatch._prepare_for_weight_sync = AsyncMock()
    dispatch.ensure_active_adapter = MagicMock()
    dispatch._broadcast_to_inference_engines = broadcast
    dispatch._finish_weight_sync = MagicMock(side_effect=lambda **kwargs: events.append("finish"))
    task = asyncio.create_task(dispatch.save_weights_for_sampler())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["pause", "broadcast", "sender-complete", "finish", "failed"]
    client.resume_generation.assert_not_awaited()
    client.increment_weight_version.assert_not_called()
