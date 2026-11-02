"""Fixed-fleet Dynamo token client with driver-owned rollout admission."""

import asyncio
import math
import os
import threading
import uuid
from contextlib import asynccontextmanager

import aiohttp

from skyrl.backends.skyrl_train.inference_servers.dynamo.discovery import (
    WorkerSnapshot,
    discover_worker_snapshot,
    validate_destination_url,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.membership import (
    MembershipGate,
)
from skyrl.backends.skyrl_train.inference_servers.dynamo.protocol import (
    ControlAck,
    WeightTransferStatus,
    read_engine_token,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteGenerateClient,
    RemoteInferenceClient,
)
from skyrl.train.config import DynamoInferenceConfig


class DynamoGenerateClient(RemoteGenerateClient):
    """Native token requests without automatic replay of uncertain attempts."""

    def __init__(self, proxy_url: str, timeout_s: float):
        super().__init__(
            proxy_url=proxy_url,
            retries=0,
            request_timeout_s=timeout_s,
            allow_redirects=False,
            response_validator=lambda body: self._validate_output(body, require_logprobs=False),
        )

    async def generate(self, *, sampling_params, **kwargs):
        if any(
            kwargs.get(key)
            for key in (
                "return_routed_experts",
                "return_sample_support",
                "return_sample_support_logprobs",
                "mm_features",
            )
        ):
            raise ValueError("Dynamo strict RL supports native token generation only")
        params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            **sampling_params,
        }
        result = await super().generate(sampling_params=params, **kwargs)
        if params.get("logprobs") is not None and result.response_logprobs is None:
            raise ValueError("Dynamo sampled logprobs must align with returned tokens")
        return result

    @staticmethod
    def _validate_output(body, require_logprobs):
        if not isinstance(body, dict) or body.get("error"):
            raise ValueError("Dynamo returned a generation error")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("Dynamo requires exactly one native generation choice")
        choice = choices[0]
        ids = choice.get("token_ids")
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Dynamo returned invalid token IDs")
        if choice.get("finish_reason") not in {"stop", "length"}:
            raise ValueError("Dynamo generation did not finish normally")
        logs = choice.get("logprobs")
        if logs is None and not require_logprobs:
            return
        content = logs.get("content") if isinstance(logs, dict) else None
        if not isinstance(content, list) or len(content) != len(ids):
            raise ValueError("Dynamo sampled logprobs must align with returned tokens")
        for entry in content:
            value = entry.get("logprob") if isinstance(entry, dict) else None
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Dynamo sampled logprobs must be finite")


class DynamoInferenceClient(RemoteInferenceClient):
    """One synchronous rollout producer and one frozen external serving fleet."""

    owns_weight_update_lifecycle = True

    def __init__(self, *, cfg: DynamoInferenceConfig, snapshot: WorkerSnapshot, transport: str, **kwargs):
        from skyrl.backends.skyrl_train.inference_servers.dynamo.control_plane import (
            DynamoControlSpec,
        )

        super().__init__(**kwargs)
        validate_destination_url(self.proxy_url, cfg.allowed_hosts)
        for worker in snapshot.workers:
            validate_destination_url(worker.system_url, cfg.allowed_hosts)
        self.cfg = cfg
        self.snapshot = snapshot
        self.transport = transport
        self.dynamo_control_spec = DynamoControlSpec(
            snapshot=snapshot,
            auth_token_env=cfg.auth_token_env,
            control_timeout_s=cfg.control_timeout_s,
            transport=transport,
        )
        self._driver = True
        self._driver_pid = os.getpid()
        self._admission_open = False
        self._failed = False
        self._active_attempts = 0
        self._update_lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._run_id = uuid.uuid4().hex
        self._policy_uid = None
        self._control_sessions = {}
        self._membership_gate = MembershipGate(lambda: self._check_membership(), self._fence)
        self._auth_header()

    @property
    def admission_open(self):
        return self._admission_open and not self._failed and self._driver and self._driver_pid == os.getpid()

    def _require_driver(self):
        if not self._driver or self._driver_pid != os.getpid():
            raise RuntimeError("Only the original Dynamo driver can open rollout admission or mutate weights")
        if self._failed:
            raise RuntimeError("Dynamo client failed; restart the run with fresh workers")

    def _fence(self):
        self._failed = True
        self._admission_open = False

    def _auth_header(self):
        secret = read_engine_token(self.cfg.auth_token_env)
        return {"Authorization": f"Bearer {secret}"}

    async def _control_session(self):
        loop = asyncio.get_running_loop()
        session = self._control_sessions.get(loop)
        if session is None or session.closed:
            session = aiohttp.ClientSession(
                trust_env=False,
                connector=aiohttp.TCPConnector(force_close=True),
                timeout=aiohttp.ClientTimeout(total=self.cfg.control_timeout_s),
            )
            self._control_sessions[loop] = session
        return session

    async def _request_control(self, worker, route, body):
        session = await self._control_session()
        async with session.post(
            worker.url_for(route), json=body or {}, headers=self._auth_header(), allow_redirects=False
        ) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Dynamo control {route} failed (HTTP {response.status})")
            result = await response.json(content_type=None)
        try:
            ControlAck.parse(result)
        except ValueError:
            raise RuntimeError(f"Dynamo control {route} did not acknowledge success") from None
        return result

    async def _fanout(self, route, body=None):
        tasks = [asyncio.create_task(self._request_control(worker, route, body)) for worker in self.snapshot.workers]
        group = asyncio.gather(*tasks, return_exceptions=True)
        cancelled = False
        while not group.done():
            try:
                await asyncio.shield(group)
            except asyncio.CancelledError:
                cancelled = True
        results = group.result()
        if cancelled:
            raise asyncio.CancelledError
        if any(isinstance(result, BaseException) for result in results):
            raise RuntimeError(f"Dynamo control {route} failed on the frozen fleet") from None
        return results

    async def _check_membership(self):
        current = await asyncio.to_thread(discover_worker_snapshot, self.cfg, self.model_name)
        if current.identity != self.snapshot.identity:
            raise RuntimeError("Dynamo frozen worker membership or capabilities changed")

    async def prepare_for_initialization(self):
        self._require_driver()
        if not self._update_lock.acquire(blocking=False):
            raise RuntimeError("A Dynamo update transaction is already active")
        try:
            with self._admission_lock:
                self._admission_open = False
                if self._active_attempts:
                    raise RuntimeError("Cannot initialize Dynamo weights with active generation attempts")
            await self._check_membership()
            await self._fanout("pause_generation", {"mode": "wait", "clear_cache": False})
        except BaseException:
            self._fence()
            raise
        finally:
            try:
                await self._close_control_session()
            finally:
                self._update_lock.release()

    @asynccontextmanager
    async def weight_update_transaction(self):
        self._require_driver()
        if not self._update_lock.acquire(blocking=False):
            raise RuntimeError("A Dynamo update transaction is already active")
        resume_started = False
        try:
            with self._admission_lock:
                if self._active_attempts:
                    raise RuntimeError("Cannot update Dynamo weights with active generation attempts")
                self._admission_open = False
            await self._check_membership()
            await self._fanout("pause_generation", {"mode": "wait", "clear_cache": False})
            version = f"{self._run_id}:{self.weight_version + 1}"
            session_id = uuid.uuid4().hex
            await self._fanout("update/begin_weight_update", {"weight_version": version, "session_id": session_id})
            yield
            await self._check_membership()
            await self._fanout("flush_cache", {})
            results = await self._fanout("get_weight_version", {})
            if not all(WeightTransferStatus.parse(result).is_installed(session_id, version) for result in results):
                raise RuntimeError("Dynamo installed policy/session readback did not match the prepared update")
            resume_started = True
            await self._fanout("resume_generation", {})
            await self._check_membership()
            self._policy_uid = version
            super().increment_weight_version()
            self._admission_open = True
        except BaseException:
            self._fence()
            if resume_started:
                try:
                    await self._fanout("pause_generation", {"mode": "abort", "clear_cache": False})
                except BaseException:
                    pass
            raise
        finally:
            self._update_lock.release()

    async def generate(self, input_batch, model=None):
        if model is not None and model != self.model_name:
            raise ValueError("Dynamo generation requires the frozen policy model")
        with self._admission_lock:
            if not self.admission_open:
                raise RuntimeError("Dynamo rollout admission is closed")
            self._active_attempts += 1
            epoch = self.weight_version
        try:
            async with self._membership_gate.attempt():
                result = await super().generate(input_batch, model=model)
            if epoch != self.weight_version or not self.admission_open:
                raise RuntimeError("Discarding a Dynamo batch whose serving epoch changed")
            return result
        except BaseException:
            self._fence()
            raise
        finally:
            with self._admission_lock:
                self._active_attempts -= 1

    def _get_generate_client(self):
        if self._generate_client is None:
            self._generate_client = DynamoGenerateClient(self.proxy_url, self.cfg.request_timeout_s)
        return self._generate_client

    async def get_world_size(self):
        return self.snapshot.world_size, 1

    async def weight_sync_control_spec(self, device_uuid):
        await self.validate_trainer_device(device_uuid)
        return self.dynamo_control_spec

    async def validate_trainer_device(self, device_uuid):
        try:
            results = await self._fanout("get_weight_version", {})
            for result in results:
                devices = WeightTransferStatus.parse(result).device_uuids
                if self.transport == "ipc" and devices[0] != device_uuid:
                    raise ValueError("Dynamo IPC requires the same physical GPU UUID as the trainer")
                if self.transport == "nccl" and devices[0] == device_uuid:
                    raise ValueError("Dynamo NCCL requires distinct physical GPU UUIDs")
        finally:
            await self._close_control_session()

    async def finish_session(self, session_id):
        # The fixed-fleet certified profile does not allocate sticky routing sessions.
        return None

    async def sample(self, *args, **kwargs):
        raise ValueError("Dynamo strict RL supports generate() token batches only")

    async def _post(self, *args, **kwargs):
        raise ValueError("Dynamo strict RL requires local tokenization and native token generation")

    async def _call_all_servers(self, *args, **kwargs):
        raise RuntimeError("Dynamo engine controls are owned by the weight-update transaction")

    def increment_weight_version(self):
        raise RuntimeError("Dynamo policy epochs are committed only by a successful update transaction")

    async def _close_control_session(self):
        session = self._control_sessions.pop(asyncio.get_running_loop(), None)
        if session is not None:
            await session.close()

    async def teardown(self):
        self._admission_open = False
        await self._close_control_session()
        await super().teardown()

    def __getstate__(self):
        state = super().__getstate__()
        state.update(
            _driver=False,
            _admission_open=False,
            _active_attempts=0,
            _control_sessions={},
            _update_lock=None,
            _admission_lock=None,
            _membership_gate=None,
        )
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._driver = False
        self._admission_open = False
        self._update_lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._membership_gate = MembershipGate(lambda: self._check_membership(), self._fence)
