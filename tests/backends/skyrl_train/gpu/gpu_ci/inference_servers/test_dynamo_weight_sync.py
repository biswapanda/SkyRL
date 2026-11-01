"""Opt-in qualification against fresh, externally owned Dynamo/vLLM processes.

The supervisor supplies SKYRL_DYNAMO_{FRONTEND,DISCOVERY,SYSTEM,DIRECT}_URL,
SKYRL_DYNAMO_NAMESPACE, SKYRL_DYNAMO_MODEL_{PATH,NAME}, and the control-token
environment variable. Tests never launch or terminate serving processes.
"""

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
import torch
from transformers import AutoModelForCausalLM

from skyrl.backends.skyrl_train.inference_servers.dynamo.client import (
    DynamoInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.discovery import (
    discover_worker_snapshot,
    validate_destination_url,
)
from skyrl.backends.skyrl_train.weight_sync.sources import FsdpWeightSource
from skyrl.backends.skyrl_train.weight_sync.weight_senders import build_trainer_engine
from skyrl.train.config import DynamoInferenceConfig
from skyrl.utils.tok import get_tokenizer

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.timeout(600),
    pytest.mark.skipif(
        os.environ.get("SKYRL_DYNAMO_QUALIFICATION") != "1",
        reason="requires an explicitly owned Dynamo qualification fleet",
    ),
]


@dataclass
class Qualification:
    client: DynamoInferenceClient
    model: torch.nn.Module
    model_path: str
    direct_url: str
    device: torch.device
    cfg: DynamoInferenceConfig

    async def status(self, checksums=False):
        worker = self.client.snapshot.workers[0]
        async with httpx.AsyncClient(trust_env=False, timeout=self.cfg.control_timeout_s) as connection:
            response = await connection.post(
                worker.url_for("get_weight_checksums" if checksums else "get_weight_version"),
                json={},
                headers={"Authorization": f"Bearer {os.environ[self.cfg.auth_token_env]}"},
            )
        response.raise_for_status()
        result = response.json()
        assert result["status"] == "ok" and result["failed"] is False
        return result


def _preserve_until_receiver_fence(evidence, timeout_s=60.0):
    """Keep IPC producers alive until owned receivers have been fenced."""
    (evidence / "fence-request").write_text(json.dumps({"reason": "qualification cleanup"}))
    deadline = time.monotonic() + timeout_s
    reported = False
    while not (evidence / "fenced").exists():
        if not reported and time.monotonic() >= deadline:
            (evidence / "fence-timeout").write_text("Receiver fence unacknowledged; preserving producer lifetime.\n")
            reported = True
        time.sleep(0.25)


@pytest_asyncio.fixture
async def qualification(ray_init_fixture):
    required = [
        "SKYRL_DYNAMO_FRONTEND_URL",
        "SKYRL_DYNAMO_DISCOVERY_URL",
        "SKYRL_DYNAMO_NAMESPACE",
        "SKYRL_DYNAMO_DIRECT_URL",
        "SKYRL_DYNAMO_MODEL_PATH",
        "SKYRL_DYNAMO_MODEL_NAME",
        "SKYRL_DYNAMO_EVIDENCE_DIR",
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"missing owned qualification inputs: {missing}")
    evidence_dir = Path(os.environ["SKYRL_DYNAMO_EVIDENCE_DIR"])
    assert evidence_dir.is_dir()
    cfg = DynamoInferenceConfig(
        namespace=os.environ["SKYRL_DYNAMO_NAMESPACE"], discovery_url=os.environ["SKYRL_DYNAMO_DISCOVERY_URL"]
    )
    frontend = validate_destination_url(os.environ["SKYRL_DYNAMO_FRONTEND_URL"], cfg.allowed_hosts)
    direct = validate_destination_url(os.environ["SKYRL_DYNAMO_DIRECT_URL"], cfg.allowed_hosts)
    model_name = os.environ["SKYRL_DYNAMO_MODEL_NAME"]
    snapshot = await asyncio.to_thread(discover_worker_snapshot, cfg, model_name)
    assert len(snapshot.workers) == 1
    if os.environ.get("SKYRL_DYNAMO_SYSTEM_URL"):
        assert snapshot.workers[0].system_url == os.environ["SKYRL_DYNAMO_SYSTEM_URL"].rstrip("/")
    client = DynamoInferenceClient(
        cfg=cfg,
        snapshot=snapshot,
        transport="ipc",
        proxy_url=frontend,
        server_urls=[snapshot.workers[0].system_url],
        data_parallel_size=1,
        model_name=model_name,
        tokenizer=get_tokenizer(os.environ["SKYRL_DYNAMO_MODEL_PATH"], local_files_only=True),
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    model = (
        AutoModelForCausalLM.from_pretrained(
            os.environ["SKYRL_DYNAMO_MODEL_PATH"],
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    env = Qualification(client, model, os.environ["SKYRL_DYNAMO_MODEL_PATH"], direct, device, cfg)
    try:
        status = await env.status()
        assert status["session_id"] is None, "qualification requires fresh workers, not an existing training fleet"
        assert status["device_uuids"] == [str(torch.cuda.get_device_properties(device).uuid)]
        assert status["model_max_len"] == 512
        await client.validate_trainer_device(status["device_uuids"][0])
        yield env
    finally:
        await asyncio.to_thread(_preserve_until_receiver_fence, evidence_dir)
        await client.teardown()


def _sender(env):
    config = SimpleNamespace(
        weight_sync_backend="ipc",
        model_dtype="bfloat16",
        weight_transfer_threshold_cuda_ipc_GB=0.125,
        speculative_config=None,
    )
    with torch.cuda.device(env.device):
        return build_trainer_engine(
            ie_cfg=config,
            colocate_all=False,
            rank=0,
            inference_world_size=1,
            source_factory=lambda dtype, _: FsdpWeightSource(env.model, dtype),
            draft_source_factory=None,
            server_urls=env.client.server_urls,
            data_parallel_size=1,
            base_model_path=env.model_path,
            control_spec=env.client.dynamo_control_spec,
        )


def _send_overwriting_buffer_after_ack(env, engine):
    """Poison the native producer's actual reusable buffer after each route ACK."""
    original_empty = torch.empty
    original_update = engine.client.update_weights
    buffers = []
    acknowledged = 0

    def capture_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if (
            args == (engine.packed_buffer_size_bytes,)
            and kwargs.get("dtype") == torch.uint8
            and kwargs.get("device") == "cuda"
        ):
            buffers.append(tensor)
        return tensor

    def overwrite_after_ack(update_info):
        nonlocal acknowledged
        original_update(update_info)
        assert len(buffers) == 1, "must observe the native packed IPC producer buffer"
        buffers[0].fill_(0xA5)
        torch.cuda.synchronize(env.device)
        acknowledged += 1

    with (
        torch.cuda.device(env.device),
        patch.object(torch, "empty", side_effect=capture_empty),
        patch.object(engine.client, "update_weights", side_effect=overwrite_after_ack),
    ):
        engine.send_weights()
    assert acknowledged >= 2, "qualification must exercise real packed-buffer reuse"
    return acknowledged


def _sampling(**overrides):
    return {
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": 17,
        "max_tokens": 16,
        "min_tokens": 0,
        "ignore_eos": False,
        "stop": [],
        "stop_token_ids": [],
        "include_stop_str_in_output": False,
        "logprobs": 1,
        **overrides,
    }


async def _direct(env, prompt, sampling):
    async with httpx.AsyncClient(trust_env=False, timeout=env.cfg.request_timeout_s) as connection:
        response = await connection.post(
            f"{env.direct_url}/inference/v1/generate",
            json={"model": env.client.model_name, "token_ids": prompt, "sampling_params": sampling},
        )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] in {"stop", "length"}
    probabilities = [entry["logprob"] for entry in (choice["logprobs"] or {"content": []})["content"]]
    assert len(probabilities) == len(choice["token_ids"]) and all(math.isfinite(value) for value in probabilities)
    return choice, probabilities


async def _generate(env, prompt, sampling, require_tokens=True):
    result = await env.client.generate({"prompt_token_ids": [prompt], "sampling_params": sampling})
    if require_tokens:
        assert result["response_ids"][0]
    probabilities = result["response_logprobs"][0] or []
    assert len(probabilities) == len(result["response_ids"][0])
    assert all(math.isfinite(value) for value in probabilities)
    return result


def _qwen_eos_stop_ids(tokenizer):
    markers = ("<|endoftext|>", "<|im_end|>")
    ids = [tokenizer.convert_tokens_to_ids(marker) for marker in markers]
    if (
        any(type(token) is not int or token < 0 for token in ids)
        or len(set(ids)) != 2
        or tokenizer.eos_token_id != ids[1]
        or [tokenizer.convert_ids_to_tokens(token) for token in ids] != list(markers)
    ):
        raise ValueError("Qwen EOS metadata must identify both exact end tokens")
    return ids


def _eos_sampler_probes(stop_ids):
    explicit = [
        {
            "stop_token_ids": stop_ids,
            "allowed_token_ids": [token],
            "min_tokens": 0,
            "max_tokens": 4,
            "ignore_eos": ignore_eos,
        }
        for token in stop_ids
        for ignore_eos in (False, True)
    ]
    implicit = [
        {
            "stop_token_ids": [],
            "allowed_token_ids": [stop_ids[1]],
            "min_tokens": 0,
            "max_tokens": 4,
            "ignore_eos": ignore_eos,
        }
        for ignore_eos in (False, True)
    ]
    return [*explicit, *implicit]


async def _assert_sampler_parity(env, prompt):
    base, logs = await _direct(env, prompt, _sampling())
    repeated, repeated_logs = await _direct(env, prompt, _sampling())
    assert repeated["token_ids"] == base["token_ids"], "baseline greedy tokens must be stable"
    baseline_drift = max((abs(left - right) for left, right in zip(logs, repeated_logs)), default=0.0)
    tolerance = max(1e-5, baseline_drift * 5)
    stop_string = env.client.tokenizer.decode(base["token_ids"][:3], skip_special_tokens=False)
    assert stop_string, "baseline must produce a nonempty string-stop probe"
    cases = [
        {},
        {"ignore_eos": True, "max_tokens": 8, "min_tokens": 4},
        {"stop_token_ids": [base["token_ids"][0]]},
        {"stop": [stop_string], "include_stop_str_in_output": False},
        {"stop": [stop_string], "include_stop_str_in_output": True},
    ]
    qwen_stop_ids = _qwen_eos_stop_ids(env.client.tokenizer)
    cases.extend(_eos_sampler_probes(qwen_stop_ids))
    for case in cases:
        sampling = _sampling(**case)
        direct, direct_logs = await _direct(env, prompt, sampling)
        observed = await _generate(env, prompt, sampling, require_tokens=False)
        assert observed["response_ids"][0] == direct["token_ids"]
        assert observed["stop_reasons"][0] == direct["finish_reason"]
        assert (observed["response_logprobs"][0] or []) == pytest.approx(direct_logs, abs=tolerance)
        if "allowed_token_ids" in case:
            implicit_disabled = case["ignore_eos"] and not case["stop_token_ids"]
            count = case["max_tokens"] if implicit_disabled else 1
            assert direct["token_ids"] == case["allowed_token_ids"] * count
            assert direct["finish_reason"] == ("length" if implicit_disabled else "stop")
    return {
        "direct_repeat_max_logprob_delta": baseline_drift,
        "parity_tolerance": tolerance,
        "cases": len(cases),
        "explicit_qwen_stop_token_ids": qwen_stop_ids,
        "forced_eos_cases": 4,
        "implicit_eos_control_cases": 2,
    }


@pytest.mark.asyncio
async def test_dynamo_native_ipc_ack_reuse_sampler_parity_and_perturbed_install(qualification, tmp_path):
    env = qualification
    before = await env.status(checksums=True)
    prompt = env.client.tokenizer.encode("The capital of France is", add_special_tokens=False)
    await env.client.prepare_for_initialization()
    sender = await asyncio.to_thread(_sender, env)
    evidence = {"device_uuid": before["device_uuids"][0], "model_max_len": before["model_max_len"]}
    try:
        for round_index in range(2):
            async with env.client.weight_update_transaction():
                chunks = await asyncio.to_thread(_send_overwriting_buffer_after_ack, env, sender)
            status = await env.status(checksums=True)
            assert (
                status["weight_checksums"] == before["weight_checksums"]
            ), "poisoning an ACKed source buffer changed installed tensors"
            evidence[f"baseline_round_{round_index}_chunks"] = chunks
        evidence["sampler_parity"] = await _assert_sampler_parity(env, prompt)
        context_prompt = [prompt[0]] * (before["model_max_len"] - 2)
        context_sampling = _sampling(max_tokens=8, ignore_eos=True)
        direct_context, context_logs = await _direct(env, context_prompt, context_sampling)
        context_result = await _generate(env, context_prompt, context_sampling)
        assert direct_context["finish_reason"] == context_result["stop_reasons"][0] == "length"
        assert context_result["response_ids"][0] == direct_context["token_ids"]
        assert len(context_result["response_ids"][0]) == 2
        assert len(context_prompt) + len(context_result["response_ids"][0]) == before["model_max_len"]
        assert context_result["response_logprobs"][0] == pytest.approx(
            context_logs, abs=evidence["sampler_parity"]["parity_tolerance"]
        )
        evidence["context_boundary"] = {
            "prompt_tokens": len(context_prompt),
            "response_tokens": 2,
            "requested_max_tokens": 8,
            "actual_model_max_len": before["model_max_len"],
            "finish_reason": context_result["stop_reasons"][0],
        }
        original = await _generate(env, prompt, _sampling(max_tokens=4, ignore_eos=True))
        with torch.no_grad():
            env.model.get_output_embeddings().weight.zero_()
        expected_uniform_logprob = -math.log(env.model.config.vocab_size)
        skipped = await _generate(env, prompt, _sampling(max_tokens=4, ignore_eos=True))
        assert skipped["response_ids"] == original["response_ids"]
        skipped_gap = max(abs(value - expected_uniform_logprob) for value in skipped["response_logprobs"][0])
        assert skipped_gap > 0.5, "the skipped-install negative control must have a measured detectable logprob effect"
        async with env.client.weight_update_transaction():
            chunks = await asyncio.to_thread(_send_overwriting_buffer_after_ack, env, sender)
        installed = await _generate(env, prompt, _sampling(max_tokens=4, ignore_eos=True))
        after = await env.status(checksums=True)
        assert after["weight_checksums"] != before["weight_checksums"]
        assert installed["response_logprobs"][0] == pytest.approx(
            [expected_uniform_logprob] * len(installed["response_ids"][0]), abs=1e-4
        )
        evidence.update(
            perturbed_chunks=chunks,
            skipped_install_logprob_gap=skipped_gap,
            committed_epochs=env.client.weight_version,
            checksum_parameter_count=len(after["weight_checksums"]),
        )
        (Path(tmp_path) / "dynamo-qualification.json").write_text(json.dumps(evidence, indent=2))
    finally:
        sender.client.close()
