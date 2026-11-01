"""Dedicated local Ray startup authenticates without serializing credentials."""

import os
import secrets
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import utils


def dynamo_config():
    return SkyRLTrainConfig.from_cli_overrides(["generator.inference_engine.dynamo.namespace=secure-local"])


@pytest.fixture
def isolated_initialization(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", "0")
    monkeypatch.setenv("RAY_AUTH_TOKEN", "ray-only-test-token-01234567890123456789")
    monkeypatch.setenv("DYN_SYSTEM_ENGINE_AUTH_TOKEN", "dynamo-only-test-token-0123456789012345")
    monkeypatch.setattr(utils, "prepare_runtime_environment", lambda cfg: {"SAFE_TEST_SETTING": "1"})
    monkeypatch.setattr(utils, "SKYRL_DUMP_INFRA_LOG_TO_STDOUT", True)
    monkeypatch.setattr(utils.ray, "is_initialized", lambda: False)
    from ray._private import ray_constants

    monkeypatch.setattr(ray_constants, "ENABLE_RAY_CLUSTER", False)
    start = Mock()
    monkeypatch.setattr(utils.ray, "init", start)
    from skyrl.backends.skyrl_train.utils import ppo_utils

    monkeypatch.setattr(ppo_utils, "sync_registries", Mock())
    return start


def test_dynamo_initialization_requires_loopback_owned_authenticated_cluster(isolated_initialization):
    utils.initialize_ray(dynamo_config())
    assert isolated_initialization.call_args.kwargs == {
        "address": "local",
        "_node_ip_address": "127.0.0.1",
        "include_dashboard": False,
        "runtime_env": {"env_vars": {"SAFE_TEST_SETTING": "1"}},
        "log_to_driver": True,
    }


def test_other_backends_keep_existing_initialization(isolated_initialization):
    utils.initialize_ray(SkyRLTrainConfig())
    assert isolated_initialization.call_args.kwargs == {
        "runtime_env": {"env_vars": {"SAFE_TEST_SETTING": "1"}},
        "log_to_driver": True,
    }


@pytest.mark.parametrize("mode", [None, "disabled", "TOKEN", ""])
def test_dynamo_rejects_disabled_ray_auth_before_starting(monkeypatch, isolated_initialization, mode):
    monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    if mode is not None:
        monkeypatch.setenv("RAY_AUTH_MODE", mode)
    with pytest.raises(ValueError, match="RAY_AUTH_MODE=token"):
        utils.initialize_ray(dynamo_config())
    isolated_initialization.assert_not_called()


@pytest.mark.parametrize("token", [None, "short", "x" * 32 + "\n", "x" * 32 + " "])
def test_dynamo_rejects_invalid_ray_secret(monkeypatch, isolated_initialization, token):
    monkeypatch.delenv("RAY_AUTH_TOKEN", raising=False)
    if token is not None:
        monkeypatch.setenv("RAY_AUTH_TOKEN", token)
    with pytest.raises(ValueError, match="token"):
        utils.initialize_ray(dynamo_config())
    isolated_initialization.assert_not_called()


def test_dynamo_rejects_shared_administration_and_ray_secret(monkeypatch, isolated_initialization):
    monkeypatch.setenv("RAY_AUTH_TOKEN", os.environ["DYN_SYSTEM_ENGINE_AUTH_TOKEN"])
    with pytest.raises(ValueError, match="independent"):
        utils.initialize_ray(dynamo_config())
    isolated_initialization.assert_not_called()


@pytest.mark.parametrize("setting", [None, "1", "false"])
def test_dynamo_requires_explicit_local_only_ray(monkeypatch, isolated_initialization, setting):
    monkeypatch.delenv("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", raising=False)
    if setting is not None:
        monkeypatch.setenv("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", setting)
    with pytest.raises(ValueError, match="before importing Ray"):
        utils.initialize_ray(dynamo_config())
    isolated_initialization.assert_not_called()


def test_dynamo_rejects_local_only_setting_applied_after_ray_import(monkeypatch, isolated_initialization):
    from ray._private import ray_constants

    monkeypatch.setattr(ray_constants, "ENABLE_RAY_CLUSTER", True)
    with pytest.raises(ValueError, match="before importing Ray"):
        utils.initialize_ray(dynamo_config())
    isolated_initialization.assert_not_called()


def test_dynamo_rejects_preinitialized_ray(monkeypatch, isolated_initialization):
    monkeypatch.setattr(utils.ray, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="newly owned"):
        utils.initialize_ray(dynamo_config())
    isolated_initialization.assert_not_called()


@pytest.mark.parametrize("address", [None, "auto", "ray://127.0.0.1:10001"])
def test_dynamo_rejects_external_ray_addresses(monkeypatch, isolated_initialization, address):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    if address is not None:
        monkeypatch.setenv("RAY_ADDRESS", address)
    with pytest.raises(ValueError, match="newly owned"):
        utils.initialize_ray(dynamo_config())
    isolated_initialization.assert_not_called()


def test_authenticated_owned_cpu_raylet_does_not_persist_secrets():
    ray_secret, dynamo_secret = secrets.token_urlsafe(48), secrets.token_urlsafe(48)
    script = textwrap.dedent(
        """
        import os
        import ipaddress
        from unittest.mock import patch
        import ray
        import psutil
        from ray._private.authentication.authentication_utils import is_token_auth_enabled
        from skyrl.train.config import SkyRLTrainConfig
        from skyrl.train.utils import utils
        from skyrl.backends.skyrl_train.inference_servers.dynamo.launch import owned_ray_launch

        assert is_token_auth_enabled()
        cfg = SkyRLTrainConfig.from_cli_overrides([
            "generator.inference_engine.dynamo.namespace=authenticated-cpu-proof",
        ])
        with patch.object(utils, "peer_access_supported", return_value=True):
            environment = utils.prepare_runtime_environment(cfg)
        assert "RAY_AUTH_TOKEN" not in environment
        assert "DYN_SYSTEM_ENGINE_AUTH_TOKEN" not in environment
        for name in ("RAY_AUTH_TOKEN", "DYN_SYSTEM_ENGINE_AUTH_TOKEN"):
            assert os.environ[name] not in repr(environment)

        def inherited():
            from ray._private.authentication.authentication_utils import is_token_auth_enabled
            return is_token_auth_enabled(), all(len(os.environ.get(name, "")) >= 32
                for name in ("RAY_AUTH_TOKEN", "DYN_SYSTEM_ENGINE_AUTH_TOKEN"))

        ray.init(**owned_ray_launch(cfg).init_kwargs, num_cpus=1, num_gpus=0,
                 _temp_dir=os.environ["SKYRL_TEST_RAY_DIR"],
                 runtime_env={"env_vars": environment})
        try:
            assert ray.get(ray.remote(num_cpus=1)(inherited).remote(), timeout=45) == (True, True)
            address = ray.get_runtime_context().gcs_address
            assert address.startswith("127.0.0.1:"), "Unexpected owned GCS address: " + address
            bindings = set()
            for child in psutil.Process().children(recursive=True):
                try:
                    for connection in child.net_connections(kind="inet"):
                        if connection.status == psutil.CONN_LISTEN:
                            bindings.add((child.name(), connection.laddr.ip))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            print("Owned Ray listen addresses: " + repr(sorted(bindings)), flush=True)
            assert bindings, "Owned Ray listener evidence is missing"
            for name, address in bindings:
                ip = ipaddress.ip_address(address)
                ip = getattr(ip, "ipv4_mapped", None) or ip
                assert ip.is_loopback, "Non-loopback owned Ray listener: " + repr((name, address))
        finally:
            ray.shutdown()
        """
    )
    with tempfile.TemporaryDirectory(prefix="sdr-auth-", dir="/tmp") as ray_directory:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "RAY_ADDRESS": "local",
                "RAY_AUTH_MODE": "token",
                "RAY_AUTH_TOKEN": ray_secret,
                "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER": "0",
                "DYN_SYSTEM_ENGINE_AUTH_TOKEN": dynamo_secret,
                "SKYRL_TEST_RAY_DIR": ray_directory,
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=150,
        )
        output = result.stdout + result.stderr
        assert ray_secret not in output and dynamo_secret not in output
        assert result.returncode == 0, output.replace(ray_secret, "[REDACTED]").replace(dynamo_secret, "[REDACTED]")
        for line in output.splitlines():
            if line.startswith("Owned Ray listen addresses:"):
                print(line)
        for path in Path(ray_directory).rglob("*"):
            if path.is_file():
                contents = path.read_bytes()
                assert (
                    ray_secret.encode() not in contents and dynamo_secret.encode() not in contents
                ), "Ray persisted an authentication credential in its owned temporary directory"
