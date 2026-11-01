"""A dedicated CPU raylet inherits credentials without runtime-env serialization."""

import os
import secrets
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def test_owned_cpu_raylet_inherits_secret_without_persisting_it():
    secret = secrets.token_urlsafe(48)
    script = textwrap.dedent(
        """
        import os
        from unittest.mock import patch
        import ray
        from skyrl.train.config import SkyRLTrainConfig
        from skyrl.train.utils import utils

        name = "DYNAMO_INHERITANCE_TEST_TOKEN"
        cfg = SkyRLTrainConfig.from_cli_overrides([
            "generator.inference_engine.dynamo.namespace=cpu-proof",
            "generator.inference_engine.dynamo.auth_token_env=" + name,
        ])
        with patch.object(utils, "peer_access_supported", return_value=True):
            environment = utils.prepare_runtime_environment(cfg)
        assert name not in environment
        assert os.environ[name] not in repr(environment)

        def inherited():
            value = os.environ.get("DYNAMO_INHERITANCE_TEST_TOKEN", "")
            return bool(value), len(value) >= 32

        ray.init(address="local", num_cpus=1, num_gpus=0,
                 include_dashboard=False, _temp_dir=os.environ["SKYRL_TEST_RAY_DIR"],
                 runtime_env={"env_vars": environment})
        try:
            assert ray.get(ray.remote(num_cpus=1)(inherited).remote(), timeout=45) == (True, True)
        finally:
            ray.shutdown()
    """
    )
    with tempfile.TemporaryDirectory(prefix="sdr-", dir="/tmp") as ray_directory:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "RAY_ADDRESS": "local",
                "DYNAMO_INHERITANCE_TEST_TOKEN": secret,
                "SKYRL_TEST_RAY_DIR": ray_directory,
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=150,
        )
        output = result.stdout + result.stderr
        assert secret not in output
        assert result.returncode == 0, output.replace(secret, "[REDACTED]")
        leaked = any(secret.encode() in path.read_bytes() for path in Path(ray_directory).rglob("*") if path.is_file())
        assert not leaked, "Ray persisted a credential in its owned temporary directory"
