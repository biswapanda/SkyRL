"""Dynamo device diagnostics must not depend on SkyRL's serving extension."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    NewInferenceWorkerWrap,
)

pytestmark = pytest.mark.vllm


def test_skyrl_extension_does_not_duplicate_dynamo_device_diagnostics():
    assert "get_weight_transfer_device_info" not in vars(NewInferenceWorkerWrap)
