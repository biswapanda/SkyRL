"""Validated, immutable discovery records for a fixed Dynamo RL worker fleet."""

import ipaddress
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from skyrl.train.config import DynamoInferenceConfig


REQUIRED_ROUTES = frozenset(
    {
        "update/init_weight_transfer_engine",
        "update/begin_weight_update",
        "update/start_weight_update",
        "update/update_weights",
        "update/finish_weight_update",
        "pause_generation",
        "resume_generation",
        "flush_cache",
        "get_weight_version",
    }
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")
_ROUTE = re.compile(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\Z")


def validate_destination_url(value: str, allowed_hosts: list[str]) -> str:
    """Validate an origin-only HTTP destination without resolving or following it."""
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError("Dynamo destination must be a nonempty HTTP origin")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Dynamo destination has an invalid host or port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.hostname.lower() not in {host.lower() for host in allowed_hosts}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or unquote(parsed.path) not in {"", "/"}
        or port == 0
    ):
        raise ValueError("Dynamo destination must be an allowlisted HTTP origin without credentials or paths")
    return value.rstrip("/")


def is_loopback_host(host: str) -> bool:
    """Accept localhost or a literal loopback address, never a DNS inference."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class WorkerRecord:
    """A frozen worker identity and its advertised administration capabilities."""

    namespace: str
    component: str
    endpoint: str
    instance_id: int
    model: str
    system_url: str
    routes: tuple[str, ...]
    world_size: int
    rank_offset: int

    @property
    def identity(self) -> tuple:
        return (
            self.namespace,
            self.component,
            self.endpoint,
            self.instance_id,
            self.model,
            self.system_url,
            self.routes,
            self.world_size,
            self.rank_offset,
        )

    def url_for(self, route: str) -> str:
        """Build only an exactly advertised route under the system engine prefix."""
        if not isinstance(route, str) or not _ROUTE.fullmatch(route) or route not in self.routes:
            raise ValueError("Dynamo engine route is invalid or not advertised")
        return f"{self.system_url}/engine/{route}"


@dataclass(frozen=True)
class WorkerSnapshot:
    """A deterministic fixed assignment, including all membership-sensitive metadata."""

    workers: tuple[WorkerRecord, ...]

    @property
    def identity(self) -> tuple:
        return tuple(worker.identity for worker in self.workers)

    @property
    def world_size(self) -> int:
        return sum(worker.world_size for worker in self.workers)


def _parse_worker(raw: Any, config: "DynamoInferenceConfig", model_name: str) -> dict:
    if not isinstance(raw, dict) or raw.get("error") or raw.get("probe_errors"):
        raise ValueError("Dynamo worker discovery probe failed or returned invalid metadata")
    if raw.get("namespace") != config.namespace or raw.get("model") != model_name:
        raise ValueError("Dynamo worker namespace or model does not match the requested fleet")
    for field in ("component", "endpoint"):
        if not isinstance(raw.get(field), str) or not _IDENTIFIER.fullmatch(raw[field]):
            raise ValueError(f"Dynamo worker has an invalid {field} identity")
    instance_id = raw.get("instance_id")
    if type(instance_id) is not int or not 0 <= instance_id < 2**64:
        raise ValueError("Dynamo worker instance_id must be an unsigned 64-bit integer")
    if type(raw.get("world_size")) is not int or raw["world_size"] != 1:
        raise ValueError("Dynamo profile requires a declared worker world_size of one")
    routes = raw.get("routes")
    if not isinstance(routes, list) or any(
        not isinstance(route, str) or not _ROUTE.fullmatch(route) for route in routes
    ):
        raise ValueError("Dynamo worker routes must be valid relative engine routes")
    if len(set(routes)) != len(routes) or not REQUIRED_ROUTES.issubset(routes):
        raise ValueError("Dynamo worker lacks required lifecycle capabilities or advertises duplicate routes")
    return {
        "namespace": raw["namespace"],
        "component": raw["component"],
        "endpoint": raw["endpoint"],
        "instance_id": instance_id,
        "model": model_name,
        "system_url": validate_destination_url(raw.get("system_url"), config.allowed_hosts),
        "routes": tuple(sorted(routes)),
        "world_size": 1,
    }


def parse_worker_snapshot(payload: Any, *, config: "DynamoInferenceConfig", model_name: str) -> WorkerSnapshot:
    """Require a complete protocol-v1 response for one exact namespace and model."""
    if not config.namespace or not isinstance(model_name, str) or not model_name:
        raise ValueError("Dynamo discovery requires an exact namespace and model")
    if type(config.expected_workers) is not int or config.expected_workers < 1:
        raise ValueError("Dynamo expected_workers must be a positive integer")
    if (
        not isinstance(payload, dict)
        or type(payload.get("protocol_version")) is not int
        or payload["protocol_version"] != 1
    ):
        raise ValueError("Dynamo discovery requires protocol_version 1")
    if payload.get("namespace") != config.namespace or payload.get("error") or payload.get("probe_errors"):
        raise ValueError("Dynamo discovery scope differs from the run namespace or a probe failed")
    raw_workers = payload.get("workers")
    if not isinstance(raw_workers, list) or len(raw_workers) != config.expected_workers:
        raise ValueError("Dynamo discovery worker count differs from expected_workers")
    workers = [_parse_worker(raw, config, model_name) for raw in raw_workers]
    identities = [(w["namespace"], w["component"], w["endpoint"], w["instance_id"]) for w in workers]
    targets = [w["system_url"] for w in workers]
    if len(set(identities)) != len(identities) or len(set(targets)) != len(targets):
        raise ValueError("Duplicate Dynamo worker identity or administration target")
    ordered = sorted(workers, key=lambda worker: (worker["instance_id"], worker["component"], worker["endpoint"]))
    return WorkerSnapshot(tuple(WorkerRecord(**worker, rank_offset=index + 1) for index, worker in enumerate(ordered)))


def discover_worker_snapshot(config: "DynamoInferenceConfig", model_name: str) -> WorkerSnapshot:
    """Read a single bounded discovery response without retries or redirects."""
    import requests

    origin = validate_destination_url(config.discovery_url, config.allowed_hosts)
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(
            f"{origin}/v1/rl/workers",
            params={"namespace": config.namespace, "model": model_name},
            timeout=config.discovery_timeout_s,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise ValueError("Dynamo discovery redirect is not allowed")
        response.raise_for_status()
        return parse_worker_snapshot(response.json(), config=config, model_name=model_name)
    finally:
        session.close()
