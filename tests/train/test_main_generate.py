"""CPU role-boundary checks for the evaluation-only entrypoint."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyrl.backends.skyrl_train.inference_servers import setup
from skyrl.backends.skyrl_train.inference_servers.dynamo import discovery
from skyrl.train.entrypoints import main_generate


@pytest.fixture
def eval_boundaries(monkeypatch):
    events = []
    validate = Mock(side_effect=lambda cfg: events.append(("validate", cfg)))
    initialize = Mock(side_effect=lambda cfg: events.append(("initialize", cfg)))
    dispatch = Mock(side_effect=lambda cfg: events.append(("dispatch", cfg)) or "eval-result-ref")
    collect = Mock(side_effect=lambda ref: events.append(("collect", ref)) or {"score": 1})
    client = Mock(side_effect=AssertionError("constructed inference client"))
    discover = Mock(side_effect=AssertionError("discovered external fleet"))
    monkeypatch.setattr(main_generate, "validate_generator_cfg", validate)
    monkeypatch.setattr(main_generate, "initialize_ray", initialize)
    monkeypatch.setattr(main_generate, "eval_entrypoint", SimpleNamespace(remote=dispatch))
    monkeypatch.setattr(main_generate.ray, "get", collect)
    monkeypatch.setattr(setup, "build_new_inference_client", client)
    monkeypatch.setattr(discovery, "discover_worker_snapshot", discover)
    return SimpleNamespace(
        events=events,
        validate=validate,
        initialize=initialize,
        dispatch=dispatch,
        collect=collect,
        client=client,
        discover=discover,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        ["generator.inference_engine.dynamo={}"],
        ["generator.inference_engine.dynamo.namespace=training"],
        ['generator={"inference_engine": {"dynamo": {"namespace": "training"}}}'],
    ],
)
def test_eval_only_rejects_dynamo_before_validation_or_mutation(monkeypatch, eval_boundaries, overrides):
    monkeypatch.setattr(main_generate.sys, "argv", ["main_generate", *overrides])
    eval_boundaries.validate.side_effect = AssertionError("validated unsupported eval role")

    with pytest.raises(ValueError, match="Dynamo.*evaluation-only"):
        main_generate.main()

    for name in ("validate", "initialize", "dispatch", "collect", "client", "discover"):
        getattr(eval_boundaries, name).assert_not_called()
    assert eval_boundaries.events == []


@pytest.mark.parametrize("overrides", [[], ["generator.inference_engine.dynamo=null"]])
def test_eval_only_native_flow_is_unchanged(monkeypatch, eval_boundaries, overrides):
    monkeypatch.setattr(main_generate.sys, "argv", ["main_generate", *overrides])

    main_generate.main()

    cfg = eval_boundaries.validate.call_args.args[0]
    assert cfg.generator.inference_engine.dynamo is None
    assert eval_boundaries.events == [
        ("validate", cfg),
        ("initialize", cfg),
        ("dispatch", cfg),
        ("collect", "eval-result-ref"),
    ]
    eval_boundaries.client.assert_not_called()
    eval_boundaries.discover.assert_not_called()
