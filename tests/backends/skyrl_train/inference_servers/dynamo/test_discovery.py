"""Strict, immutable discovery for the fixed Dynamo RL fleet."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import pytest
import requests

from skyrl.backends.skyrl_train.inference_servers.dynamo.discovery import (
    REQUIRED_ROUTES,
    discover_worker_snapshot,
    is_loopback_host,
    parse_worker_snapshot,
)
from skyrl.train.config import DynamoInferenceConfig


@pytest.fixture
def config():
    return DynamoInferenceConfig(namespace="skyrl-test")


@pytest.fixture
def payload():
    return {
        "protocol_version": 1,
        "namespace": "skyrl-test",
        "workers": [
            {
                "namespace": "skyrl-test",
                "component": "backend",
                "endpoint": "generate",
                "instance_id": 11,
                "model": "test-model",
                "world_size": 1,
                "system_url": "http://127.0.0.1:9100",
                "routes": sorted(REQUIRED_ROUTES),
            }
        ],
    }


def test_snapshot_is_frozen_and_contains_exact_routes(payload, config):
    snapshot = parse_worker_snapshot(payload, config=config, model_name="test-model")
    worker = snapshot.workers[0]
    assert snapshot.world_size == 1
    assert worker.instance_id == 11
    assert worker.url_for("update/update_weights") == "http://127.0.0.1:9100/engine/update/update_weights"
    assert worker.rank_offset == 1
    assert isinstance(worker.routes, tuple)
    with pytest.raises(FrozenInstanceError):
        worker.instance_id = 12
    with pytest.raises(FrozenInstanceError):
        snapshot.workers = ()
    payload["workers"][0]["routes"].clear()
    assert worker.url_for("pause_generation").endswith("/engine/pause_generation")


def test_identity_is_stable_for_order_but_changes_for_routes_and_address(payload, config):
    config.expected_workers = 2
    second = deepcopy(payload["workers"][0])
    second.update(instance_id=2, system_url="http://localhost:9101")
    payload["workers"].append(second)
    first = parse_worker_snapshot(payload, config=config, model_name="test-model")
    payload["workers"].reverse()
    assert first.identity == parse_worker_snapshot(payload, config=config, model_name="test-model").identity
    assert [w.instance_id for w in first.workers] == [2, 11]
    assert [w.rank_offset for w in first.workers] == [1, 2]
    payload["workers"][0]["routes"].append("optional_route")
    assert first.identity != parse_worker_snapshot(payload, config=config, model_name="test-model").identity


@pytest.mark.parametrize(
    "field,value",
    [
        ("namespace", "another-run"),
        ("model", "another-model"),
        ("component", ""),
        ("endpoint", "../generate"),
        ("instance_id", True),
        ("instance_id", -1),
        ("instance_id", "11"),
        ("world_size", 2),
        ("world_size", True),
        ("world_size", None),
        ("error", "probe failed"),
        ("routes", ["pause_generation"]),
        ("routes", "pause_generation"),
        ("routes", ()),
        ("routes", []),
        ("system_url", "http://untrusted.example:9100"),
        ("system_url", "file:///tmp/admin"),
        ("system_url", "http://secret@127.0.0.1:9100"),
        ("system_url", "http://127.0.0.1:9100/../engine"),
        ("system_url", "http://127.0.0.1:9100/%2e%2e/engine"),
        ("system_url", "http://127.0.0.1:9100?secret=x"),
        ("system_url", "http://127.0.0.1:9100/#fragment"),
        ("system_url", "http://127.0.0.1:invalid"),
    ],
)
def test_rejects_invalid_workers(payload, config, field, value):
    payload["workers"][0][field] = value
    with pytest.raises(ValueError):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


@pytest.mark.parametrize(
    "route", ["../pause_generation", "/pause_generation", "update/%2e%2e/x", "https://localhost/x", "x?query", "x\\y"]
)
def test_rejects_malicious_advertised_routes(payload, config, route):
    payload["workers"][0]["routes"].append(route)
    with pytest.raises(ValueError):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(protocol_version=2),
        lambda p: p.update(protocol_version=True),
        lambda p: p.update(namespace="another-run"),
        lambda p: p.update(workers=[]),
        lambda p: p.update(workers="invalid"),
        lambda p: p.update(error="discovery failed"),
        lambda p: p.update(probe_errors=["failed"]),
        lambda p: p["workers"].append(deepcopy(p["workers"][0])),
    ],
)
def test_rejects_invalid_snapshot(payload, config, mutate):
    mutate(payload)
    with pytest.raises(ValueError):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


def test_rejects_duplicate_control_target(payload, config):
    config.expected_workers = 2
    other = deepcopy(payload["workers"][0])
    other["instance_id"] = 12
    payload["workers"].append(other)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


def test_rejects_duplicate_worker_identity_with_distinct_targets(payload, config):
    config.expected_workers = 2
    other = {**payload["workers"][0], "system_url": "http://localhost:9101"}
    payload["workers"].append(other)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


def test_rank_offsets_are_assigned_by_parser_not_untrusted_metadata(payload, config):
    payload["workers"][0]["rank_offset"] = 2
    snapshot = parse_worker_snapshot(payload, config=config, model_name="test-model")
    assert snapshot.workers[0].rank_offset == 1


def test_url_for_rejects_unadvertised_route(payload, config):
    worker = parse_worker_snapshot(payload, config=config, model_name="test-model").workers[0]
    with pytest.raises(ValueError, match="advertised"):
        worker.url_for("unadvertised")


def test_discovery_uses_bounded_get_without_redirect_or_retry(payload, config):
    response = Mock(status_code=200)
    response.json.return_value = payload
    with patch("requests.Session") as factory:
        get = factory.return_value.get
        get.return_value = response
        snapshot = discover_worker_snapshot(config, "test-model")
    assert snapshot.world_size == 1
    get.assert_called_once_with(
        "http://127.0.0.1:8001/v1/rl/workers",
        params={"namespace": "skyrl-test", "model": "test-model"},
        timeout=5.0,
        allow_redirects=False,
    )
    response.raise_for_status.assert_called_once()


def test_discovery_rejects_redirect_without_fetching_target(config):
    response = Mock(status_code=302)
    session = Mock()
    session.get.return_value = response
    with patch("requests.Session", return_value=session), pytest.raises(ValueError, match="redirect"):
        discover_worker_snapshot(config, "test-model")
    session.get.assert_called_once()
    session.close.assert_called_once()


@pytest.mark.parametrize("payload", [None, [], {"protocol_version": 1}])
def test_invalid_payload_shape(payload, config):
    with pytest.raises(ValueError):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


def test_zero_worker_configuration_cannot_admit_empty_snapshot(payload, config):
    config.expected_workers = 0
    payload["workers"] = []
    with pytest.raises(ValueError, match="expected_workers"):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


def test_missing_namespace_or_model_is_rejected(payload, config):
    with pytest.raises(ValueError, match="namespace and model"):
        parse_worker_snapshot(payload, config=config, model_name="")
    config.namespace = ""
    with pytest.raises(ValueError, match="namespace and model"):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


def test_duplicate_routes_rejected(payload, config):
    payload["workers"][0]["routes"].append("pause_generation")
    with pytest.raises(ValueError, match="duplicate routes"):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


@pytest.mark.parametrize("system_url", [None, "", "http://127.0.0.1:0", "http://127.0.0.1:9100\n"])
def test_invalid_destination_shape(payload, config, system_url):
    payload["workers"][0]["system_url"] = system_url
    with pytest.raises(ValueError):
        parse_worker_snapshot(payload, config=config, model_name="test-model")


def test_ipv6_loopback_is_supported(payload, config):
    payload["workers"][0]["system_url"] = "http://[::1]:9100/"
    worker = parse_worker_snapshot(payload, config=config, model_name="test-model").workers[0]
    assert worker.url_for("flush_cache") == "http://[::1]:9100/engine/flush_cache"


@pytest.mark.parametrize(
    "host,expected",
    [("127.0.0.1", True), ("::1", True), ("localhost", True), ("example.test", False), ("10.0.0.1", False)],
)
def test_loopback_validation_does_not_resolve_dns(host, expected):
    assert is_loopback_host(host) is expected


def test_failed_discovery_read_is_not_retried(config):
    session = Mock()
    session.get.side_effect = requests.Timeout("timed out")
    with patch("requests.Session", return_value=session), pytest.raises(requests.Timeout):
        discover_worker_snapshot(config, "test-model")
    session.get.assert_called_once()
    session.close.assert_called_once()


def test_failed_discovery_status_is_not_parsed(config):
    response = Mock(status_code=503)
    response.raise_for_status.side_effect = requests.HTTPError("unavailable")
    session = Mock()
    session.get.return_value = response
    with patch("requests.Session", return_value=session), pytest.raises(requests.HTTPError):
        discover_worker_snapshot(config, "test-model")
    response.json.assert_not_called()


def test_discovery_isolates_proxy_environment_and_closes_session(payload, config):
    response = Mock(status_code=200)
    response.json.return_value = payload
    session = Mock()
    session.get.return_value = response
    with (
        patch("requests.Session", return_value=session),
        patch("requests.get", side_effect=AssertionError("unisolated read")),
    ):
        discover_worker_snapshot(config, "test-model")
    assert session.trust_env is False
    assert session.get.call_args.kwargs["allow_redirects"] is False
    session.close.assert_called_once()
