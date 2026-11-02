"""Both control transports consume the same strict, versioned wire contract."""

from dataclasses import FrozenInstanceError

import pytest

from skyrl.backends.skyrl_train.inference_servers.dynamo.protocol import (
    ControlAck,
    WeightTransferStatus,
    read_engine_token,
)


def status_body(**changes):
    return {
        "protocol_version": 1,
        "status": "ok",
        "version": None,
        "version_declared": False,
        "session_id": None,
        "install_complete": False,
        "pending_weight_version": "policy-1",
        "pending_session_id": "session-1",
        "failed": False,
        "next_seq": 0,
        "model_max_len": 512,
        "device_uuids": ["GPU-a"],
        **changes,
    }


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {},
        {"status": "ok"},
        {"protocol_version": True, "status": "ok"},
        {"protocol_version": 2, "status": "ok"},
        {"protocol_version": 1, "status": "error"},
        {"protocol_version": 1, "status": "ok", "success": 1},
        {"protocol_version": 1, "status": "ok", "failed": 0},
        {"protocol_version": 1, "status": "ok", "error": "secret"},
    ],
)
def test_ack_rejects_malformed_or_unversioned_bodies_without_echoing_payload(body):
    with pytest.raises(ValueError, match="control acknowledgement") as error:
        ControlAck.parse(body)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version_declared", 1),
        ("install_complete", 0),
        ("failed", 0),
        ("next_seq", True),
        ("next_seq", -1),
        ("model_max_len", 0),
        ("device_uuids", []),
        ("device_uuids", ["GPU-a", "GPU-b"]),
        ("device_uuids", [""]),
        ("pending_session_id", "\n"),
        ("pending_weight_version", "x" * 257),
    ],
)
def test_status_rejects_wrong_types_and_invalid_values(field, value):
    with pytest.raises(ValueError):
        WeightTransferStatus.parse(status_body(**{field: value}))


@pytest.mark.parametrize("field", list(status_body()))
def test_status_requires_every_contract_field(field):
    body = status_body()
    del body[field]
    with pytest.raises(ValueError):
        WeightTransferStatus.parse(body)


def test_typed_status_predicates_and_immutability():
    prepared = WeightTransferStatus.parse(status_body())
    assert prepared.is_prepared
    assert prepared.matches("session-1", "policy-1", 0)
    assert not prepared.matches("other", "policy-1", 0)
    installed = WeightTransferStatus.parse(
        status_body(
            version="policy-1",
            session_id="session-1",
            pending_session_id=None,
            pending_weight_version=None,
            version_declared=True,
            install_complete=True,
            next_seq=3,
        )
    )
    assert installed.matches("session-1", "policy-1", 3, finished=True)
    assert installed.is_installed("session-1", "policy-1")
    assert not installed.is_prepared
    with pytest.raises(FrozenInstanceError):
        prepared.next_seq = 2


@pytest.mark.parametrize("token", [None, "short", "a" * 32 + " ", "a" * 32 + "\n", "a" * 32 + "\x7f"])
def test_shared_token_validation_is_secret_safe(monkeypatch, token):
    monkeypatch.delenv("TEST_ENGINE_TOKEN", raising=False)
    if token is not None:
        monkeypatch.setenv("TEST_ENGINE_TOKEN", token)
    with pytest.raises(ValueError, match="token"):
        read_engine_token("TEST_ENGINE_TOKEN")


def test_shared_token_validation_reads_environment_only(monkeypatch):
    monkeypatch.setenv("TEST_ENGINE_TOKEN", "a" * 32)
    assert read_engine_token("TEST_ENGINE_TOKEN") == "a" * 32
