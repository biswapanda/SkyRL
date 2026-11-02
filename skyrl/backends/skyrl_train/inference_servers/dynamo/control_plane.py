"""Authenticated blocking control for frozen Dynamo weight-transfer sessions."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

from skyrl.backends.skyrl_train.inference_servers.dynamo.discovery import (
    WorkerRecord,
    WorkerSnapshot,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.protocol import (
    ControlAck,
    WeightTransferStatus,
    read_engine_token,
)
from skyrl.backends.skyrl_train.weight_sync.control_plane import json_safe
from skyrl.backends.skyrl_train.weight_sync.control_spec import InitPayloadFn


class DynamoControlError(RuntimeError):
    """A failed or ambiguous control call permanently fences this sender."""


@dataclass(frozen=True)
class DynamoControlSpec:
    """Frozen carrier of already validated configuration and discovery metadata.

    Configuration enters through ``validate_dynamo_config``; untrusted worker
    records enter through ``parse_worker_snapshot``. Those canonical boundaries
    own transport, deadline, credential-name, host-allowlist and worker validation.
    This internal carrier does not accept discovery payloads or credentials.
    """

    snapshot: WorkerSnapshot
    auth_token_env: str
    control_timeout_s: float
    transport: str

    @property
    def fail_closed_transfers(self) -> bool:
        return True

    def build_client(self, init_payload_fn: InitPayloadFn | None) -> "DynamoWeightSyncClient":
        return DynamoWeightSyncClient(self)


@dataclass(frozen=True)
class _SenderState:
    initialized: bool = False
    failed: bool = False
    closed: bool = False
    session_id: str | None = None
    version: str | None = None
    next_seq: int = 0
    chunks: int = 0
    used_sessions: frozenset[str] = frozenset()
    used_versions: frozenset[str] = frozenset()


class DynamoWeightSyncClient:
    """vLLM's synchronous four-method protocol over strict Dynamo engine controls."""

    fail_closed_transfers = True

    def __init__(self, spec: DynamoControlSpec) -> None:
        import requests

        token = read_engine_token(spec.auth_token_env)
        self._spec = spec
        self._state = _SenderState()
        self._lock = RLock()
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update({"Authorization": f"Bearer {token}", "Connection": "close"})
        self._pool = ThreadPoolExecutor(max_workers=len(spec.snapshot.workers), thread_name_prefix="dynamo-weight-sync")

    @property
    def spec(self) -> DynamoControlSpec:
        return self._spec

    @property
    def failed(self) -> bool:
        return self._state.failed

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._lock:
            if self._state.failed or self._state.closed:
                raise DynamoControlError("Dynamo sender is failed or closed; restart required")
            try:
                yield
            except BaseException as exc:
                self._state = replace(self._state, failed=True)
                if isinstance(exc, DynamoControlError):
                    raise
                raise DynamoControlError("Dynamo control operation failed; sender permanently fenced") from None

    def _post(self, worker: WorkerRecord, route: str, body: dict) -> dict:
        response = self._session.post(
            worker.url_for(route),
            json=body,
            timeout=self.spec.control_timeout_s,
            allow_redirects=False,
        )
        try:
            if not 200 <= response.status_code < 300:
                raise DynamoControlError("Dynamo worker returned an unsuccessful HTTP status")
            result = response.json()
            ControlAck.parse(result)
            return result
        finally:
            response.close()

    def _fanout(self, route: str, bodies: tuple[dict, ...]) -> tuple[dict, ...]:
        if len(bodies) != len(self.spec.snapshot.workers):
            raise DynamoControlError("Dynamo control fan-out does not match the frozen fleet")
        futures, results, failures = [], [], []
        for index, (worker, body) in enumerate(zip(self.spec.snapshot.workers, bodies)):
            try:
                futures.append((index, self._pool.submit(self._post, worker, route, body)))
            except BaseException:  # noqa: BLE001
                failures.append(index)
        for index, future in futures:
            while True:
                try:
                    results.append(future.result())
                    break
                except BaseException:  # noqa: BLE001
                    failures.append(index)
                    if future.done():
                        break
        if failures:
            raise DynamoControlError(f"Dynamo control {route} failed on frozen worker indices {failures}") from None
        return tuple(results)

    def _uniform(self, route: str, body: dict) -> tuple[dict, ...]:
        return self._fanout(route, tuple(dict(body) for _ in self.spec.snapshot.workers))

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        with self._operation():
            if self._state.initialized or not isinstance(init_info, dict):
                raise DynamoControlError("Dynamo weight transfer must be initialized exactly once")
            if self.spec.transport == "ipc" and init_info.get("packed") is not True:
                raise DynamoControlError("Dynamo IPC transfer requires packed buffers")
            if self.spec.transport == "nccl" and init_info.get("world_size") != self.spec.snapshot.world_size + 1:
                raise DynamoControlError("Dynamo NCCL world size does not match the frozen fleet")
            bodies = tuple(
                {
                    "init_info": {
                        **init_info,
                        **({"rank_offset": worker.rank_offset} if self.spec.transport == "nccl" else {}),
                    }
                }
                for worker in self.spec.snapshot.workers
            )
            self._fanout("update/init_weight_transfer_engine", bodies)
            self._state = replace(self._state, initialized=True)

    def _prepared_session(self) -> tuple[str, str]:
        results = self._uniform("get_weight_version", {})
        sessions = []
        for result in results:
            status = WeightTransferStatus.parse(result)
            if not status.is_prepared:
                raise DynamoControlError("Dynamo worker has no valid prepared weight session")
            sessions.append((status.pending_session_id, status.pending_weight_version))
        if len(set(sessions)) != 1:
            raise DynamoControlError("Dynamo workers disagree on the prepared weight session")
        return sessions[0]

    def _validate_ack(self, results: tuple[dict, ...], next_seq: int, *, finished: bool = False) -> None:
        for result in results:
            if not WeightTransferStatus.parse(result).matches(
                self._state.session_id, self._state.version, next_seq, finished=finished
            ):
                raise DynamoControlError("Dynamo worker acknowledgement does not match the weight session")

    def start_weight_update(self) -> None:
        with self._operation():
            if not self._state.initialized or self._state.session_id is not None:
                raise DynamoControlError("Dynamo start is outside the weight-transfer lifecycle")
            session, version = self._prepared_session()
            if session in self._state.used_sessions or version in self._state.used_versions:
                raise DynamoControlError("Dynamo prepared session or policy version was already used")
            self._state = replace(self._state, session_id=session, version=version, next_seq=0, chunks=0)
            results = self._uniform("update/start_weight_update", {"session_id": session, "seq": 0})
            self._validate_ack(results, 1)
            self._state = replace(
                self._state,
                next_seq=1,
                used_sessions=self._state.used_sessions | {session},
                used_versions=self._state.used_versions | {version},
            )

    def update_weights(self, update_info: dict[str, Any]) -> None:
        with self._operation():
            if self._state.session_id is None or not isinstance(update_info, dict):
                raise DynamoControlError("Dynamo update requires a started session and object metadata")
            results = self._uniform(
                "update/update_weights",
                {
                    "session_id": self._state.session_id,
                    "seq": self._state.next_seq,
                    "update_info": json_safe(update_info),
                },
            )
            self._validate_ack(results, self._state.next_seq + 1)
            self._state = replace(self._state, next_seq=self._state.next_seq + 1, chunks=self._state.chunks + 1)

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        with self._operation():
            if self._state.session_id is None or not self._state.chunks:
                raise DynamoControlError("Dynamo finish requires a session with at least one installed chunk")
            if weight_version is not None and weight_version != self._state.version:
                raise DynamoControlError("Dynamo finish version differs from the prepared policy")
            results = self._uniform(
                "update/finish_weight_update", {"session_id": self._state.session_id, "seq": self._state.next_seq}
            )
            self._validate_ack(results, self._state.next_seq + 1, finished=True)
            self._state = replace(self._state, session_id=None, version=None, next_seq=0, chunks=0)

    def close(self) -> None:
        with self._lock:
            if self._state.closed:
                return
            self._state = replace(self._state, closed=True)
            self._pool.shutdown(wait=True)
            self._session.close()
