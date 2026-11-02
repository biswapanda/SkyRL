"""Opt-in real worker loss after a packed IPC chunk is acknowledged.

The owned supervisor consumes ``worker-loss-request`` and writes ``worker-lost``
only after stopping its worker group and observing its GPU contexts withdraw.
The test never obtains process identifiers from discovery or kills processes.
"""

import asyncio
import gc
import json
import os
import time
import uuid
import weakref
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
import torch
from transformers import AutoModelForCausalLM

from skyrl.backends.skyrl_train.inference_servers.dynamo.client import (
    DynamoInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.control_plane import (
    DynamoControlError,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.discovery import (
    discover_worker_snapshot,
    validate_destination_url,
)
from skyrl.backends.skyrl_train.weight_sync.ipc_lifetime import FAILED_IPC_PRODUCERS
from skyrl.train.config import DynamoInferenceConfig
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.gpu_ci.inference_servers.test_dynamo_weight_sync import (
    Qualification,
    _generate,
    _preserve_until_receiver_fence,
    _sampling,
    _sender,
)

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.timeout(600),
    pytest.mark.skipif(
        os.environ.get("SKYRL_DYNAMO_WORKER_LOSS_QUALIFICATION") != "1",
        reason="requires an explicitly owned worker-loss qualification fleet",
    ),
]


def _wait_worker_lost(evidence, request, timeout_s=90.0):
    (evidence / "worker-loss-request").write_text(json.dumps(request))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            acknowledgment = json.loads((evidence / "worker-lost").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        assert acknowledgment.get("nonce") == request["nonce"], "worker fence acknowledgment belongs to another attempt"
        assert acknowledgment.get("receiver_fenced") is True, "worker loss must include GPU-context withdrawal"
        return acknowledgment
    raise TimeoutError("Owned supervisor did not acknowledge the requested worker loss")


@pytest_asyncio.fixture
async def worker_loss_qualification(ray_init_fixture):
    required = [
        "SKYRL_DYNAMO_FRONTEND_URL",
        "SKYRL_DYNAMO_DISCOVERY_URL",
        "SKYRL_DYNAMO_NAMESPACE",
        "SKYRL_DYNAMO_MODEL_PATH",
        "SKYRL_DYNAMO_MODEL_NAME",
        "SKYRL_DYNAMO_EVIDENCE_DIR",
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"missing owned worker-loss qualification inputs: {missing}")
    evidence = Path(os.environ["SKYRL_DYNAMO_EVIDENCE_DIR"])
    assert evidence.is_dir()
    assert not (evidence / "worker-loss-request").exists() and not (evidence / "worker-lost").exists()
    cfg = DynamoInferenceConfig(
        namespace=os.environ["SKYRL_DYNAMO_NAMESPACE"],
        discovery_url=os.environ["SKYRL_DYNAMO_DISCOVERY_URL"],
        control_timeout_s=10.0,
        request_timeout_s=30.0,
    )
    model_name, model_path = os.environ["SKYRL_DYNAMO_MODEL_NAME"], os.environ["SKYRL_DYNAMO_MODEL_PATH"]
    snapshot = await asyncio.to_thread(discover_worker_snapshot, cfg, model_name)
    assert len(snapshot.workers) == 1
    if os.environ.get("SKYRL_DYNAMO_SYSTEM_URL"):
        assert snapshot.workers[0].system_url == os.environ["SKYRL_DYNAMO_SYSTEM_URL"].rstrip("/")
    client = DynamoInferenceClient(
        cfg=cfg,
        snapshot=snapshot,
        transport="ipc",
        proxy_url=validate_destination_url(os.environ["SKYRL_DYNAMO_FRONTEND_URL"], cfg.allowed_hosts),
        server_urls=[snapshot.workers[0].system_url],
        data_parallel_size=1,
        model_name=model_name,
        tokenizer=get_tokenizer(model_path, local_files_only=True),
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
        )
        .to(device)
        .eval()
    )
    env = Qualification(client, model, model_path, "", device, cfg)
    try:
        status = await env.status()
        assert status["session_id"] is None and status["install_complete"] is False
        assert status["device_uuids"] == [str(torch.cuda.get_device_properties(device).uuid)]
        await client.validate_trainer_device(status["device_uuids"][0])
        yield env, evidence
    finally:
        await asyncio.to_thread(_preserve_until_receiver_fence, evidence)
        await client.teardown()


def _status_sync(env):
    worker = env.client.snapshot.workers[0]
    with httpx.Client(trust_env=False, timeout=env.cfg.control_timeout_s) as connection:
        response = connection.post(
            worker.url_for("get_weight_version"),
            json={},
            headers={"Authorization": f"Bearer {os.environ[env.cfg.auth_token_env]}"},
        )
    response.raise_for_status()
    result = response.json()
    assert result["status"] == "ok" and result["failed"] is False
    return result


@pytest.mark.asyncio
async def test_dynamo_worker_loss_mid_update_fences_epoch_and_retains_ipc_producer(worker_loss_qualification):
    env, evidence_dir = worker_loss_qualification
    await env.client.prepare_for_initialization()
    sender = await asyncio.to_thread(_sender, env)
    try:
        async with env.client.weight_update_transaction():
            await asyncio.to_thread(sender.send_weights)
        committed = await env.status()
        committed_epoch = env.client.weight_version
        assert committed_epoch == 1 and env.client.admission_open
        prompt = env.client.tokenizer.encode("The capital of France is", add_special_tokens=False)
        baseline = await _generate(env, prompt, _sampling(max_tokens=4, ignore_eos=True))
        assert baseline["response_ids"][0]
        changed = set()
        with torch.no_grad():
            for name, parameter in env.model.named_parameters():
                before = parameter.flatten()[0].item()
                parameter.flatten()[0].add_(0.125)
                assert parameter.flatten()[0].item() != before
                changed.add(name)
        torch.cuda.synchronize(env.device)
        observed = {"update_attempts": 0, "acknowledged_chunks": 0, "routes": []}
        buffers = []
        original_empty, original_update, original_fanout = torch.empty, sender.client.update_weights, env.client._fanout

        def capture_empty(*args, **kwargs):
            tensor = original_empty(*args, **kwargs)
            if (
                args == (sender.packed_buffer_size_bytes,)
                and kwargs.get("dtype") == torch.uint8
                and kwargs.get("device") == "cuda"
            ):
                buffers.append(weakref.ref(tensor))
            return tensor

        async def record_fanout(route, body):
            observed["routes"].append(route)
            return await original_fanout(route, body)

        def lose_worker_after_first_ack(update_info):
            observed["update_attempts"] += 1
            original_update(update_info)
            observed["acknowledged_chunks"] += 1
            assert observed["acknowledged_chunks"] == 1, "a chunk was acknowledged after the worker fence"
            assert set(update_info["names"]) & changed, "the acknowledged chunk did not contain perturbed tensors"
            assert len(buffers) == 1 and buffers[0]() is not None
            assert env.client.admission_open is False and env.client.weight_version == committed_epoch
            status = _status_sync(env)
            assert status["install_complete"] is False and status["next_seq"] == 2
            assert status["pending_session_id"] and status["pending_session_id"] != committed["session_id"]
            assert status["pending_weight_version"] and status["pending_weight_version"] != committed["version"]
            request = {
                "nonce": uuid.uuid4().hex,
                "worker_identity": env.client.snapshot.workers[0].identity,
                "committed_epoch": committed_epoch,
                "acknowledged_chunks": observed["acknowledged_chunks"],
                "pending_session_id": status["pending_session_id"],
                "pending_weight_version": status["pending_weight_version"],
                "next_seq": status["next_seq"],
            }
            observed["loss_request"] = request
            observed["loss_acknowledgment"] = _wait_worker_lost(evidence_dir, request)
            observed["worker_fenced_at"] = time.monotonic()

        data_client = env.client._get_generate_client()
        with (
            patch.object(torch, "empty", side_effect=capture_empty),
            patch.object(sender.client, "update_weights", side_effect=lose_worker_after_first_ack),
            patch.object(env.client, "_fanout", side_effect=record_fanout),
            patch.object(data_client, "_post", wraps=data_client._post) as generation_posts,
        ):
            with pytest.raises(DynamoControlError, match="update/update_weights"):
                async with env.client.weight_update_transaction():
                    await asyncio.to_thread(sender.send_weights)
            failure_delay = time.monotonic() - observed["worker_fenced_at"]
            assert failure_delay < env.cfg.control_timeout_s + 5.0
            assert observed["update_attempts"] == 2 and observed["acknowledged_chunks"] == 1
            assert "resume_generation" not in observed["routes"]
            assert "flush_cache" not in observed["routes"]
            assert env.client.weight_version == committed_epoch and not env.client.admission_open
            assert sender.client.failed and sender._skyrl_ipc_failed
            for _ in range(2):
                with pytest.raises(RuntimeError, match="admission is closed"):
                    await _generate(env, prompt, _sampling())
            with pytest.raises(RuntimeError, match="client failed"):
                async with env.client.weight_update_transaction():
                    pytest.fail("failed client opened another update transaction")
            with pytest.raises(RuntimeError, match="outcome is uncertain"):
                await asyncio.to_thread(sender.send_weights)
            assert generation_posts.call_count == 0

        engine_reference = weakref.ref(sender)
        producer = weakref.ref(sender._skyrl_ipc_producer)
        assert producer() is not None and producer().gi_frame is not None
        assert producer() in FAILED_IPC_PRODUCERS
        sender.client.close()
        sender = None
        await asyncio.sleep(0)
        gc.collect()
        assert engine_reference() is None, "engine collection must not own the quarantined buffer lifetime"
        assert producer() is not None and producer().gi_frame is not None
        assert buffers[0]() is not None, "failed producer's exported CUDA buffer was released"
        assert producer() in FAILED_IPC_PRODUCERS
        observed.update(
            committed_epoch=committed_epoch,
            committed_policy_uid=committed["version"],
            generation_posts_after_loss=0,
            failed_policy_committed=False,
            sender_permanently_failed=True,
            native_producer_globally_quarantined=True,
            exported_buffer_survived_engine_collection=True,
            next_chunk_failure_delay_s=failure_delay,
        )
        (evidence_dir / "worker-loss-result.json").write_text(json.dumps(observed, indent=2))
    finally:
        if sender is not None:
            sender.client.close()
