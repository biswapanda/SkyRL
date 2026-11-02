"""Version-one strict RL control contract shared by both client transports."""

import os
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1


def read_engine_token(env_name: str) -> str:
    token = os.environ.get(env_name, "")
    if len(token) < 32 or not all(33 <= ord(char) <= 126 for char in token):
        raise ValueError("Dynamo engine token must contain at least 32 printable ASCII characters without whitespace")
    return token


@dataclass(frozen=True)
class ControlAck:
    protocol_version: int = PROTOCOL_VERSION

    @classmethod
    def parse(cls, body: Any) -> "ControlAck":
        if (
            not isinstance(body, dict)
            or type(body.get("protocol_version")) is not int
            or body["protocol_version"] != PROTOCOL_VERSION
            or body.get("status") != "ok"
            or body.get("error")
            or ("failed" in body and body["failed"] is not False)
            or ("success" in body and body["success"] is not True)
        ):
            raise ValueError("Dynamo control acknowledgement is unsuccessful or has an unsupported protocol")
        return cls()


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 256 and value.isprintable()


@dataclass(frozen=True)
class WeightTransferStatus:
    version: str | None
    version_declared: bool
    session_id: str | None
    install_complete: bool
    pending_weight_version: str | None
    pending_session_id: str | None
    failed: bool
    next_seq: int
    model_max_len: int
    device_uuids: tuple[str, ...]

    @classmethod
    def parse(cls, body: Any) -> "WeightTransferStatus":
        ControlAck.parse(body)
        keys = cls.__dataclass_fields__
        if any(key not in body for key in keys):
            raise ValueError("Dynamo weight status is missing contract fields")
        for key in ("version_declared", "install_complete", "failed"):
            if type(body[key]) is not bool:
                raise ValueError("Dynamo weight status requires boolean lifecycle fields")
        for key in ("version", "session_id", "pending_weight_version", "pending_session_id"):
            if body[key] is not None and not _identifier(body[key]):
                raise ValueError("Dynamo weight status contains an invalid identifier")
        if type(body["next_seq"]) is not int or body["next_seq"] < 0:
            raise ValueError("Dynamo weight status requires a nonnegative sequence")
        if type(body["model_max_len"]) is not int or body["model_max_len"] <= 0:
            raise ValueError("Dynamo weight status requires a positive context limit")
        devices = body["device_uuids"]
        if not isinstance(devices, list) or len(devices) != 1 or not _identifier(devices[0]):
            raise ValueError("Dynamo must report exactly one physical device UUID per worker")
        return cls(**{key: body[key] for key in keys if key != "device_uuids"}, device_uuids=tuple(devices))

    @property
    def is_prepared(self) -> bool:
        return (
            not self.failed
            and not self.install_complete
            and self.next_seq == 0
            and self.pending_session_id is not None
            and self.pending_weight_version is not None
        )

    def is_installed(self, session_id: str, version: str) -> bool:
        return (
            not self.failed
            and self.install_complete
            and self.version_declared
            and self.session_id == session_id
            and self.version == version
            and self.pending_session_id is None
            and self.pending_weight_version is None
        )

    def matches(self, session_id: str, version: str, next_seq: int, *, finished: bool = False) -> bool:
        if finished:
            return self.next_seq == next_seq and self.is_installed(session_id, version)
        return (
            not self.failed
            and not self.install_complete
            and self.next_seq == next_seq
            and self.pending_session_id == session_id
            and self.pending_weight_version == version
        )
