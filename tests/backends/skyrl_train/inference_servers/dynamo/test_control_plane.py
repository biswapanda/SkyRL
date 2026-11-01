"""The blocking sender binds every chunk to one frozen, authenticated session."""

import pickle
import secrets
from dataclasses import FrozenInstanceError, replace
from threading import Barrier
from unittest.mock import Mock, patch

import pytest
import requests

from skyrl.backends.skyrl_train.inference_servers.dynamo.control_plane import (
    DynamoControlError,
    DynamoControlSpec,
    DynamoWeightSyncClient,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.discovery import (
    REQUIRED_ROUTES,
    WorkerRecord,
    WorkerSnapshot,
)
from tests.backends.skyrl_train.inference_servers.dynamo.test_protocol import (
    status_body,
)

TOKEN = secrets.token_urlsafe(48)


@pytest.fixture
def spec():
    workers = tuple(
        WorkerRecord(
            "run",
            "backend",
            "generate",
            i,
            "model",
            f"http://127.0.0.1:{9100 + i}",
            tuple(sorted(REQUIRED_ROUTES)),
            1,
            i,
        )
        for i in (1, 2)
    )
    return DynamoControlSpec(WorkerSnapshot(workers), "TEST_DYNAMO_TOKEN", 12.0, "ipc")


def state(**overrides):
    return status_body(**overrides)


def response(body=None, status_code=200):
    result = Mock(status_code=status_code)
    result.json.return_value = state() if body is None else body
    return result


@pytest.fixture
def client(spec, monkeypatch, request):
    spec = replace(spec, transport=getattr(request, "param", "ipc"))
    monkeypatch.setenv(spec.auth_token_env, TOKEN)
    with patch("requests.Session") as factory:
        session = factory.return_value
        session.headers = {}

        def request(url, *, json, **kwargs):
            if url.endswith(("get_weight_version", "init_weight_transfer_engine")):
                return response()
            if url.endswith("finish_weight_update"):
                return response(
                    state(
                        pending_session_id=None,
                        pending_weight_version=None,
                        session_id="session-1",
                        version="policy-1",
                        version_declared=True,
                        install_complete=True,
                        next_seq=json["seq"] + 1,
                    )
                )
            return response(state(next_seq=json["seq"] + 1))

        session.post.side_effect = request
        result = DynamoWeightSyncClient(spec)
        yield result, session
        result.close()


def initialize(client):
    client.init_weight_transfer_engine({"packed": True})


def test_spec_is_serializable_without_credentials(spec):
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert TOKEN not in repr(spec)
    with pytest.raises(FrozenInstanceError):
        spec.transport = "nccl"


def test_spec_is_a_carrier_that_builds_strict_client(spec, monkeypatch):
    monkeypatch.setenv(spec.auth_token_env, TOKEN)
    assert not hasattr(spec, "validate_sender")
    assert not hasattr(spec, "__post_init__")
    client = spec.build_client(None)
    try:
        assert client.spec is spec
        assert client.fail_closed_transfers is spec.fail_closed_transfers is True
    finally:
        client.close()


@pytest.mark.parametrize("token", [None, "short", "x" * 31, "x" * 32 + "\n", "x" * 32 + " ", "x" * 32 + "\x7f"])
def test_invalid_token_fails_before_network(spec, monkeypatch, token):
    monkeypatch.delenv(spec.auth_token_env, raising=False)
    if token is not None:
        monkeypatch.setenv(spec.auth_token_env, token)
    with patch("requests.Session") as factory, pytest.raises(ValueError, match="token"):
        DynamoWeightSyncClient(spec)
    factory.assert_not_called()


def test_lifecycle_sequence_and_auth_are_exact(client):
    sender, session = client
    initialize(sender)
    sender.start_weight_update()
    sender.update_weights({"names": ["w"]})
    sender.update_weights({"names": ["v"]})
    sender.finish_weight_update("policy-1")
    assert session.trust_env is False
    assert session.headers["Authorization"] == f"Bearer {TOKEN}"
    assert session.headers["Connection"] == "close"
    for call in session.post.call_args_list:
        assert call.kwargs["timeout"] == 12.0
        assert call.kwargs["allow_redirects"] is False
    for route, seq in [("start_weight_update", 0), ("finish_weight_update", 3)]:
        calls = [call for call in session.post.call_args_list if call.args[0].endswith("/" + route)]
        assert len(calls) == 2
        assert all(call.kwargs["json"] == {"session_id": "session-1", "seq": seq} for call in calls)
    updates = [call.kwargs["json"] for call in session.post.call_args_list if call.args[0].endswith("/update_weights")]
    assert sorted(body["seq"] for body in updates) == [1, 1, 2, 2]


@pytest.mark.parametrize("client", ["nccl"], indirect=True)
def test_nccl_offsets_come_from_frozen_records(client):
    sender, session = client
    sender.init_weight_transfer_engine({"world_size": 3, "rank_offset": 99})
    calls = session.post.call_args_list
    assert {call.args[0]: call.kwargs["json"]["init_info"]["rank_offset"] for call in calls} == {
        "http://127.0.0.1:9101/engine/update/init_weight_transfer_engine": 1,
        "http://127.0.0.1:9102/engine/update/init_weight_transfer_engine": 2,
    }


def test_ipc_handles_use_existing_json_encoder(client):
    sender, session = client
    initialize(sender)
    sender.start_weight_update()
    original = {"ipc_handles": {0: ("storage", b"handle")}, "packed": True}
    sender.update_weights(original)
    bodies = [
        call.kwargs["json"]["update_info"]
        for call in session.post.call_args_list
        if call.args[0].endswith("/update_weights")
    ]
    assert "ipc_handles" in original
    assert all("ipc_handles" not in body and isinstance(body["ipc_handles_pickled"], str) for body in bodies)


@pytest.mark.parametrize(
    "bad_body",
    [
        [],
        {},
        {"status": "error", "message": TOKEN},
        {"status": "ok", "success": False},
        {"status": "ok", "error": "failed"},
        {"status": "ok", "failed": True},
    ],
)
def test_body_failure_is_permanent_and_secret_safe(client, bad_body):
    sender, session = client
    session.post.side_effect = None
    session.post.return_value = response(bad_body)
    with pytest.raises(DynamoControlError) as error:
        initialize(sender)
    assert TOKEN not in str(error.value)
    before = session.post.call_count
    with pytest.raises(DynamoControlError, match="failed"):
        sender.start_weight_update()
    assert session.post.call_count == before


@pytest.mark.parametrize("status_code", [302, 401, 500])
def test_http_errors_and_redirects_fail_closed(client, status_code):
    sender, session = client
    session.post.side_effect = None
    session.post.return_value = response(status_code=status_code)
    with pytest.raises(DynamoControlError):
        initialize(sender)
    assert session.post.call_count == 2


def test_fanout_is_concurrent_and_drains_all_failures(client):
    sender, session = client
    arrived = Barrier(2, timeout=3)
    completed = []

    def fail(url, **kwargs):
        arrived.wait()
        completed.append(url)
        raise requests.Timeout(TOKEN)

    session.post.side_effect = fail
    with pytest.raises(DynamoControlError) as error:
        initialize(sender)
    assert len(completed) == 2
    assert TOKEN not in str(error.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"failed": True},
        {"pending_session_id": None},
        {"pending_weight_version": None},
        {"next_seq": 1},
        {"next_seq": True},
        {"install_complete": True},
    ],
)
def test_start_requires_prepared_session(client, overrides):
    sender, session = client
    initialize(sender)
    session.post.side_effect = None
    session.post.return_value = response(state(**overrides))
    with pytest.raises(DynamoControlError):
        sender.start_weight_update()
    assert all(not call.args[0].endswith("/start_weight_update") for call in session.post.call_args_list)


def test_start_requires_unanimous_session(client):
    sender, session = client
    initialize(sender)
    session.post.side_effect = lambda url, **kwargs: response(state(pending_session_id=url))
    with pytest.raises(DynamoControlError, match="session"):
        sender.start_weight_update()


def test_out_of_order_calls_fail_before_network(client):
    sender, session = client
    with pytest.raises(DynamoControlError):
        sender.update_weights({"names": ["w"]})
    session.post.assert_not_called()


def test_finish_requires_a_chunk_and_matching_version(client):
    sender, session = client
    initialize(sender)
    sender.start_weight_update()
    sender.update_weights({"names": ["w"]})
    with pytest.raises(DynamoControlError):
        sender.finish_weight_update("wrong-policy")
    assert all(not call.args[0].endswith("/finish_weight_update") for call in session.post.call_args_list)


def test_finish_rejects_incorrect_install_acknowledgement(client):
    sender, session = client
    initialize(sender)
    sender.start_weight_update()
    sender.update_weights({"names": ["w"]})
    session.post.side_effect = None
    session.post.return_value = response(state(install_complete=True, version="wrong-policy", next_seq=3))
    with pytest.raises(DynamoControlError):
        sender.finish_weight_update()


def test_close_is_idempotent_and_blocks_further_calls(client):
    sender, session = client
    sender.close()
    sender.close()
    session.close.assert_called_once()
    with pytest.raises(DynamoControlError):
        initialize(sender)


def test_repeated_prepared_session_is_rejected(client):
    sender, _session = client
    initialize(sender)
    sender.start_weight_update()
    sender.update_weights({"names": ["w"]})
    sender.finish_weight_update()
    with pytest.raises(DynamoControlError, match="already used"):
        sender.start_weight_update()


def test_finish_without_chunk_is_rejected(client):
    sender, _session = client
    initialize(sender)
    sender.start_weight_update()
    with pytest.raises(DynamoControlError, match="chunk"):
        sender.finish_weight_update()


def test_reinitialization_is_rejected(client):
    sender, session = client
    initialize(sender)
    before = session.post.call_count
    with pytest.raises(DynamoControlError, match="once"):
        initialize(sender)
    assert session.post.call_count == before


def test_invalid_update_acknowledgement_fences_sender(client):
    sender, session = client
    initialize(sender)
    sender.start_weight_update()
    session.post.side_effect = None
    session.post.return_value = response(state(next_seq=True))
    with pytest.raises(DynamoControlError):
        sender.update_weights({"names": ["w"]})
    assert sender.failed


def test_cancellation_outcome_is_drained_and_latched(client):
    sender, session = client
    session.post.side_effect = KeyboardInterrupt()
    with pytest.raises(DynamoControlError):
        initialize(sender)
    assert sender.failed
    assert session.post.call_count == 2


def test_malformed_json_response_is_masked(client):
    sender, session = client
    result = response()
    result.json.side_effect = ValueError(TOKEN)
    session.post.side_effect = None
    session.post.return_value = result
    with pytest.raises(DynamoControlError) as exc:
        initialize(sender)
    assert TOKEN not in str(exc.value)
    assert result.close.call_count == 2
