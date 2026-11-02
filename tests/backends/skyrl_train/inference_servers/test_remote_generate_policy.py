"""Canonical data-plane transport policy hooks used by external providers."""

from unittest.mock import AsyncMock, Mock
from uuid import UUID

import aiohttp
import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteGenerateClient,
)


class _Response:
    status = 200
    method = "POST"
    url = "http://worker/generate"
    headers = {}
    reason = "Service Unavailable"

    async def read(self):
        return b'{"answer": 1}'

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_strict_policy_has_one_attempt_timeout_and_validator():
    validate = Mock()
    client = RemoteGenerateClient(
        "http://worker", retries=0, request_timeout_s=3.0, response_validator=validate, allow_redirects=False
    )
    session = Mock(post=Mock(return_value=_Response()))
    client._get_session = AsyncMock(return_value=session)
    assert await client._post("http://worker/generate", {"prompt": [1]}) == {"answer": 1}
    validate.assert_called_once_with({"answer": 1})
    kwargs = session.post.call_args.kwargs
    assert kwargs["timeout"].total == 3.0
    assert kwargs["allow_redirects"] is False


@pytest.mark.asyncio
async def test_zero_retries_does_not_replay_disconnected_generation(monkeypatch):
    client = RemoteGenerateClient("http://worker", retries=0)
    session = Mock(post=Mock(side_effect=aiohttp.ServerDisconnectedError()))
    client._get_session = AsyncMock(return_value=session)
    sleep = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep)
    with pytest.raises(aiohttp.ServerDisconnectedError):
        await client._post("http://worker/generate", {})
    session.post.assert_called_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_validator_failure_is_not_retried():
    client = RemoteGenerateClient("http://worker", response_validator=Mock(side_effect=ValueError("invalid response")))
    session = Mock(post=Mock(return_value=_Response()))
    client._get_session = AsyncMock(return_value=session)
    with pytest.raises(ValueError, match="invalid response"):
        await client._post("http://worker/generate", {})
    session.post.assert_called_once()


@pytest.mark.asyncio
async def test_non_json_http_failure_retains_status_without_strict_replay():
    class Unavailable(_Response):
        status = 503

        async def read(self):
            return b"backend unavailable"

        async def text(self):
            return "backend unavailable"

    client = RemoteGenerateClient("http://worker", retries=0)
    session = Mock(post=Mock(return_value=Unavailable()))
    client._get_session = AsyncMock(return_value=session)
    with pytest.raises(aiohttp.ClientResponseError, match="backend unavailable") as caught:
        await client._post("http://worker/generate", {})
    assert caught.value.status == 503
    session.post.assert_called_once()


def test_default_retry_budget_preserves_thirty_total_attempts():
    assert RemoteGenerateClient("http://worker").retries == 29
    assert RemoteGenerateClient("http://worker").allow_redirects is True


@pytest.mark.asyncio
async def test_every_generation_has_a_unique_request_id():
    client = RemoteGenerateClient("http://worker")
    client._post = AsyncMock(return_value={"choices": [{"token_ids": [1], "finish_reason": "stop"}]})
    request_ids = []
    for _ in range(2):
        await client.generate(prompt_token_ids=[1], sampling_params={}, session_id=None, model="model")
        request_ids.append(client._post.call_args.kwargs["headers"]["X-Request-ID"])
    assert len(set(request_ids)) == 2
    assert all(UUID(value).hex == value for value in request_ids)


@pytest.mark.parametrize(
    "kwargs", [{"retries": -1}, {"retries": True}, {"request_timeout_s": 0}, {"request_timeout_s": float("nan")}]
)
def test_invalid_transport_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RemoteGenerateClient("http://worker", **kwargs)
