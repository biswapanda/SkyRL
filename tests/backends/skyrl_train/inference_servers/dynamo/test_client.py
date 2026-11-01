"""Admission and all-or-fail update transactions for the Dynamo driver."""

import asyncio
import json
import pickle
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import aiohttp
import pytest

from skyrl.backends.skyrl_train.inference_servers.dynamo.client import (
    DynamoGenerateClient,
    DynamoInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.discovery import (
    REQUIRED_ROUTES,
    WorkerRecord,
    WorkerSnapshot,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.membership import (
    CohortInvalidated,
)
from skyrl.train.config import DynamoInferenceConfig
from tests.backends.skyrl_train.inference_servers.dynamo.test_protocol import (
    status_body,
)


@asynccontextmanager
async def control_http_server():
    from aiohttp import web

    requests = []
    reply = {"status": 200, "body": {"status": "ok", "protocol_version": 1}, "headers": {}}

    async def handler(request):
        requests.append((request.path, dict(request.headers), await request.json()))
        return web.Response(
            status=reply["status"],
            text=json.dumps(reply["body"]),
            headers=reply["headers"],
            content_type="text/plain",
        )

    app = web.Application()
    app.router.add_post("/engine/{route:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}", requests, reply
    finally:
        await runner.cleanup()


def make_client(monkeypatch):
    monkeypatch.setenv("DYN_SYSTEM_ENGINE_AUTH_TOKEN", "test-control-secret-at-least-32-bytes")
    snapshot = WorkerSnapshot(
        (
            WorkerRecord(
                "run", "backend", "generate", 1, "model", "http://127.0.0.1:9001", tuple(sorted(REQUIRED_ROUTES)), 1, 1
            ),
        )
    )
    client = DynamoInferenceClient(
        cfg=DynamoInferenceConfig(namespace="run"),
        snapshot=snapshot,
        transport="ipc",
        proxy_url="http://127.0.0.1:8000",
        server_urls=["http://127.0.0.1:9001"],
        data_parallel_size=1,
        model_name="model",
    )
    client._check_membership = AsyncMock()
    client._fanout = AsyncMock(return_value=[{"status": "ok"}])
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "wrong-device", "rpc-error", "cancelled"])
async def test_one_shot_device_validation_closes_its_worker_copy_session(monkeypatch, outcome):
    client = make_client(monkeypatch)
    session = await client._control_session()
    client._fanout.return_value = [status_body()]
    try:
        if outcome == "success":
            await client.validate_trainer_device("GPU-a")
        elif outcome == "wrong-device":
            with pytest.raises(ValueError, match="GPU UUID"):
                await client.validate_trainer_device("GPU-b")
        else:
            failure = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
            client._fanout.side_effect = failure()
            with pytest.raises(failure):
                await client.validate_trainer_device("GPU-a")
        assert session.closed
        assert not client._control_sessions
    finally:
        await client.aclose()


def test_client_rejects_unallowlisted_worker_control_origin(monkeypatch):
    from dataclasses import replace

    original = make_client(monkeypatch)
    worker = replace(original.snapshot.workers[0], system_url="http://untrusted.invalid:9001")
    with pytest.raises(ValueError, match="allowlisted"):
        DynamoInferenceClient(
            cfg=original.cfg,
            snapshot=WorkerSnapshot((worker,)),
            transport="nccl",
            proxy_url=original.proxy_url,
            server_urls=[worker.system_url],
            data_parallel_size=1,
            model_name="model",
        )


@pytest.mark.asyncio
async def test_initialization_closes_admission_before_control_mutation(monkeypatch):
    client = make_client(monkeypatch)
    client._admission_open = True

    async def control(*args):
        assert not client.admission_open
        return [{"status": "ok"}]

    client._fanout.side_effect = control
    await client.prepare_for_initialization()
    assert not client.admission_open


@pytest.mark.asyncio
async def test_initialization_does_not_race_an_active_update(monkeypatch):
    client = make_client(monkeypatch)
    client._update_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="transaction"):
            await client.prepare_for_initialization()
    finally:
        client._update_lock.release()
    client._fanout.assert_not_awaited()


@pytest.mark.asyncio
async def test_membership_change_after_install_is_fenced_before_cache_or_resume(monkeypatch):
    client = make_client(monkeypatch)
    client._check_membership.side_effect = [None, RuntimeError("membership changed after install")]
    with pytest.raises(RuntimeError, match="membership changed after install"):
        async with client.weight_update_transaction():
            pass
    routes = [call.args[0] for call in client._fanout.await_args_list]
    assert routes == ["pause_generation", "update/begin_weight_update"]
    assert not client.admission_open
    assert client.weight_version == 0


@pytest.mark.asyncio
async def test_generation_initially_closed_and_worker_copy_cannot_open(monkeypatch):
    client = make_client(monkeypatch)
    with pytest.raises(RuntimeError, match="admission"):
        await client.generate({"prompt_token_ids": [[1]]})
    client._check_membership = None
    client._fanout = None
    restored = pickle.loads(pickle.dumps(client))
    assert restored.dynamo_control_spec == client.dynamo_control_spec
    with pytest.raises(RuntimeError, match="driver"):
        await restored.prepare_for_initialization()
    with pytest.raises(RuntimeError, match="admission"):
        await restored.generate({"prompt_token_ids": [[1]]})


@pytest.mark.asyncio
async def test_initialization_pauses_without_resuming(monkeypatch):
    client = make_client(monkeypatch)
    await client.prepare_for_initialization()
    client._check_membership.assert_awaited_once()
    client._fanout.assert_awaited_once_with("pause_generation", {"mode": "wait", "clear_cache": False})
    assert client.weight_version == 0


@pytest.mark.asyncio
async def test_successful_update_resets_verifies_resumes_then_commits(monkeypatch):
    client = make_client(monkeypatch)
    calls = []

    async def control(route, body=None):
        calls.append(route)
        if route == "update/begin_weight_update":
            control.version, control.session = body["weight_version"], body["session_id"]
        if route == "get_weight_version":
            return [
                status_body(
                    version=control.version,
                    session_id=control.session,
                    install_complete=True,
                    version_declared=True,
                    pending_session_id=None,
                    pending_weight_version=None,
                )
            ]
        return [{"status": "ok"}]

    client._fanout = control
    async with client.weight_update_transaction():
        assert client.weight_version == 0
        with pytest.raises(RuntimeError, match="admission"):
            await client.generate({"prompt_token_ids": [[1]]})
        calls.append("send")
    assert calls == [
        "pause_generation",
        "update/begin_weight_update",
        "send",
        "flush_cache",
        "get_weight_version",
        "resume_generation",
    ]
    assert client.weight_version == 1
    assert client.admission_open
    assert client._check_membership.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["send", "flush_cache", "get_weight_version", "resume_generation", "membership", "membership_after_resume"],
)
async def test_failure_permanently_closes_admission(monkeypatch, failure):
    client = make_client(monkeypatch)
    routes = []

    async def control(route, body=None):
        routes.append(route)
        if route == failure:
            raise RuntimeError("injected")
        if route == "get_weight_version":
            return [
                status_body(
                    version="wrong",
                    session_id="wrong",
                    install_complete=True,
                    version_declared=True,
                    pending_session_id=None,
                    pending_weight_version=None,
                )
            ]
        return [{"status": "ok"}]

    # Keep readback valid except in its own negative case.
    original = control

    async def valid_control(route, body=None):
        if route == "update/begin_weight_update":
            valid_control.version, valid_control.session = body["weight_version"], body["session_id"]
        if route == "get_weight_version" and failure != route:
            routes.append(route)
            return [
                status_body(
                    version=valid_control.version,
                    session_id=valid_control.session,
                    install_complete=True,
                    version_declared=True,
                    pending_session_id=None,
                    pending_weight_version=None,
                )
            ]
        return await original(route, body)

    client._fanout = valid_control
    if failure == "membership":
        client._check_membership.side_effect = [None, RuntimeError("changed")]
    elif failure == "membership_after_resume":
        client._check_membership.side_effect = [None, None, RuntimeError("changed after resume")]
    with pytest.raises(RuntimeError):
        async with client.weight_update_transaction():
            if failure == "send":
                raise RuntimeError("send failed")
    assert not client.admission_open
    assert client.weight_version == 0
    if failure in {"send", "flush_cache", "get_weight_version"}:
        assert "resume_generation" not in routes
    elif failure in {"resume_generation", "membership_after_resume"}:
        assert routes[-1] == "pause_generation"
    with pytest.raises(RuntimeError, match="failed"):
        async with client.weight_update_transaction():
            pytest.fail("failed client reopened")


@pytest.mark.asyncio
async def test_cancellation_during_send_fences_without_resume(monkeypatch):
    client = make_client(monkeypatch)
    with pytest.raises(asyncio.CancelledError):
        async with client.weight_update_transaction():
            raise asyncio.CancelledError
    assert not client.admission_open
    assert all(call.args[0] != "resume_generation" for call in client._fanout.await_args_list)


@pytest.mark.asyncio
async def test_active_generation_prevents_update(monkeypatch):
    client = make_client(monkeypatch)
    client._active_attempts = 1
    with pytest.raises(RuntimeError, match="active"):
        async with client.weight_update_transaction():
            pytest.fail("update raced generation")
    client._fanout.assert_not_awaited()


@pytest.mark.asyncio
async def test_device_identity_qualifies_transport(monkeypatch):
    client = make_client(monkeypatch)
    client._fanout.return_value = [status_body()]
    await client.validate_trainer_device("GPU-a")
    with pytest.raises(ValueError, match="same physical"):
        await client.validate_trainer_device("GPU-b")
    client.transport = "nccl"
    with pytest.raises(ValueError, match="distinct"):
        await client.validate_trainer_device("GPU-a")


@pytest.mark.asyncio
async def test_membership_change_discards_generation_result(monkeypatch):
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )

    client = make_client(monkeypatch)
    client._admission_open = True
    client._check_membership.side_effect = [None, RuntimeError("membership changed")]
    generate = AsyncMock(return_value={"response_ids": [[2]]})
    monkeypatch.setattr(RemoteInferenceClient, "generate", generate)
    with pytest.raises(CohortInvalidated) as error:
        await client.generate({"prompt_token_ids": [[1]]})
    assert str(error.value.__cause__) == "membership changed"
    assert client._active_attempts == 0


@pytest.mark.asyncio
async def test_failed_request_closes_admission_while_its_cohort_drains(monkeypatch):
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )

    client = make_client(monkeypatch)
    client._admission_open = True
    both_started, first_failed, release_peer = (asyncio.Event() for _ in range(3))
    count = 0

    async def request(*args, **kwargs):
        nonlocal count
        count += 1
        index = count
        if count == 2:
            both_started.set()
        await both_started.wait()
        if index == 1:
            first_failed.set()
            raise RuntimeError("uncertain request")
        await release_peer.wait()
        return {"response_ids": [[2]]}

    monkeypatch.setattr(RemoteInferenceClient, "generate", request)
    tasks = [asyncio.create_task(client.generate({"prompt_token_ids": [[1]]})) for _ in range(2)]
    try:
        await first_failed.wait()
        await asyncio.sleep(0)
        assert not client.admission_open
        assert client._active_attempts == 2
    finally:
        release_peer.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert client._active_attempts == 0
    assert not client.admission_open


@pytest.mark.asyncio
async def test_fanout_drains_other_workers_on_failure_and_cancellation(monkeypatch):
    from dataclasses import replace

    for cancel in (False, True):
        client = make_client(monkeypatch)
        worker = client.snapshot.workers[0]
        client.snapshot = WorkerSnapshot(
            (worker, replace(worker, instance_id=2, system_url="http://127.0.0.1:9002", rank_offset=2))
        )
        del client._fanout
        started, release = asyncio.Event(), asyncio.Event()
        completed = []

        async def request(worker, route, body):
            if worker.instance_id == 1:
                started.set()
                if not cancel:
                    raise RuntimeError("first failure")
            await release.wait()
            completed.append(worker.instance_id)
            return {"status": "ok"}

        client._request_control = request
        task = asyncio.create_task(client._fanout("flush_cache", {}))
        await started.wait()
        if cancel:
            task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await task
        assert 2 in completed


@pytest.mark.parametrize(
    "choice",
    [
        {"token_ids": [2], "finish_reason": "stop", "logprobs": {"content": []}},
        {"token_ids": [2], "finish_reason": "stop", "logprobs": {"content": [{"logprob": float("nan")}]}},
        {"token_ids": [2], "finish_reason": "abort"},
        {"token_ids": [True], "finish_reason": "stop"},
        {"token_ids": [2], "finish_reason": "stop"},
    ],
)
def test_native_output_rejects_invalid_or_substituted_probabilities(choice):
    with pytest.raises(ValueError):
        DynamoGenerateClient._validate_output({"choices": [choice]}, require_logprobs=True)


@pytest.mark.asyncio
async def test_alternate_generation_and_legacy_controls_cannot_bypass_gate(monkeypatch):
    client = make_client(monkeypatch)
    with pytest.raises(ValueError, match="token batches"):
        await client.sample({})
    with pytest.raises(RuntimeError, match="transaction"):
        await client.resume_generation()
    with pytest.raises(RuntimeError, match="committed"):
        client.increment_weight_version()


@pytest.mark.asyncio
async def test_real_http_native_sampling_payload_and_no_retries(monkeypatch):
    from aiohttp import web

    requests = []

    async def handler(request):
        requests.append((dict(request.headers), await request.json()))
        if len(requests) == 2:
            return web.Response(status=503)
        return web.json_response(
            {"choices": [{"token_ids": [2], "finish_reason": "stop", "logprobs": {"content": [{"logprob": -0.3}]}}]}
        )

    app = web.Application()
    app.router.add_post("/inference/v1/generate", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = DynamoGenerateClient(f"http://127.0.0.1:{port}", 3)
    try:
        result = await client.generate(
            prompt_token_ids=[1], sampling_params={"logprobs": 0, "temperature": 0.7}, session_id=None, model="model"
        )
        assert result.response_ids == [2]
        assert result.response_logprobs == [-0.3]
        headers, payload = requests[0]
        assert payload["sampling_params"]["temperature"] == 0.7
        assert payload["sampling_params"]["repetition_penalty"] == 1.0
        assert payload["sampling_params"]["top_k"] == -1
        assert headers["X-Request-ID"]
        assert "Authorization" not in headers
        with pytest.raises(aiohttp.ClientResponseError) as error:
            await client.generate(prompt_token_ids=[1], sampling_params={}, session_id=None, model="model")
        assert error.value.status == 503
        assert len(requests) == 2
    finally:
        await client.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_real_http_controls_authenticate_and_parse_legacy_text_json_without_retries(monkeypatch):
    from dataclasses import replace

    client = make_client(monkeypatch)
    async with control_http_server() as (origin, requests, reply):
        worker = replace(client.snapshot.workers[0], system_url=origin)
        try:
            first_session = await client._control_session()
            assert await client._control_session() is first_session
            result = await client._request_control(worker, "pause_generation", {"mode": "wait"})
            assert result == {"status": "ok", "protocol_version": 1}
            assert requests[0][0] == "/engine/pause_generation"
            assert requests[0][1]["Authorization"] == "Bearer test-control-secret-at-least-32-bytes"
            assert requests[0][2] == {"mode": "wait"}
            reply["status"] = 401
            reply["body"] = {"error": "authentication required"}
            with pytest.raises(RuntimeError, match="401") as error:
                await client._request_control(worker, "pause_generation", {})
            assert "test-control-secret" not in str(error.value)
            assert len(requests) == 2
            reply["status"] = 503
            with pytest.raises(RuntimeError, match="503"):
                await client._request_control(worker, "pause_generation", {})
            assert len(requests) == 3
            reply["status"] = 307
            reply["headers"] = {"Location": f"{origin}/engine/resume_generation"}
            with pytest.raises(RuntimeError, match="307"):
                await client._request_control(worker, "pause_generation", {})
            assert len(requests) == 4
        finally:
            await client.aclose()
            assert first_session.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_native_generation_rejects_unfollowed_redirect_with_valid_token_body(status):
    from aiohttp import web

    requests = []

    async def handler(request):
        requests.append(request.path)
        return web.json_response(
            {"choices": [{"token_ids": [2], "finish_reason": "stop"}]},
            status=status,
            headers={"Location": "/unexpected"},
        )

    app = web.Application()
    app.router.add_post("/inference/v1/generate", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = DynamoGenerateClient(f"http://127.0.0.1:{port}", 3)
    try:
        with pytest.raises(aiohttp.ClientResponseError) as error:
            await client.generate(prompt_token_ids=[1], sampling_params={}, session_id=None, model="model")
        assert error.value.status == status
        assert requests == ["/inference/v1/generate"]
    finally:
        await client.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_precheck_cancellation_fences_admission_before_drain_finishes(monkeypatch):
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )

    client = make_client(monkeypatch)
    client._admission_open = True
    checking, release = asyncio.Event(), asyncio.Event()

    async def check():
        checking.set()
        await release.wait()

    client._check_membership.side_effect = check
    generate = AsyncMock()
    monkeypatch.setattr(RemoteInferenceClient, "generate", generate)
    task = asyncio.create_task(client.generate({"prompt_token_ids": [[1]]}))
    await checking.wait()
    task.cancel()
    try:
        await asyncio.sleep(0)
        assert not client.admission_open
        assert not task.done()
        assert client._active_attempts == 1
        with pytest.raises(RuntimeError, match="admission"):
            await client.generate({"prompt_token_ids": [[1]]})
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [{"status": "ok", "success": False}, {"status": "error"}, [], {"status": "ok", "failed": True}]
)
async def test_real_http_controls_reject_unsuccessful_acknowledgements(monkeypatch, body):
    from dataclasses import replace

    client = make_client(monkeypatch)
    async with control_http_server() as (origin, _, reply):
        worker = replace(client.snapshot.workers[0], system_url=origin)
        reply["body"] = body
        try:
            with pytest.raises(RuntimeError, match="acknowledge"):
                await client._request_control(worker, "flush_cache", {})
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_real_http_fanout_attempts_all_workers_despite_one_failure(monkeypatch):
    from dataclasses import replace

    client = make_client(monkeypatch)
    del client._fanout
    async with control_http_server() as (first, first_requests, first_reply):
        async with control_http_server() as (second, second_requests, _):
            worker = client.snapshot.workers[0]
            client.snapshot = WorkerSnapshot(
                (
                    replace(worker, system_url=first),
                    replace(worker, instance_id=2, rank_offset=2, system_url=second),
                )
            )
            first_reply["status"] = 503
            try:
                with pytest.raises(RuntimeError, match="frozen fleet"):
                    await client._fanout("flush_cache", {})
                assert len(first_requests) == len(second_requests) == 1
                assert first_requests[0][1]["Authorization"] == second_requests[0][1]["Authorization"]
            finally:
                await client.aclose()


@pytest.mark.asyncio
async def test_real_membership_check_rejects_capability_changes(monkeypatch):
    from dataclasses import replace

    from skyrl.backends.skyrl_train.inference_servers.dynamo import (
        client as client_module,
    )

    client = make_client(monkeypatch)
    del client._check_membership
    worker = client.snapshot.workers[0]
    monkeypatch.setattr(client_module, "discover_worker_snapshot", lambda *args: client.snapshot)
    await client._check_membership()
    changed = WorkerSnapshot((replace(worker, instance_id=2),))
    monkeypatch.setattr(client_module, "discover_worker_snapshot", lambda *args: changed)
    with pytest.raises(RuntimeError, match="membership or capabilities changed"):
        await client._check_membership()
