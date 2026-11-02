from types import SimpleNamespace

import pytest

from tests.backends.skyrl_train.gpu.gpu_ci.inference_servers import (
    test_dynamo_weight_sync as qualification,
)


def tokenizer(mapping, eos_token_id=151645):
    reverse = {value: key for key, value in mapping.items()}
    return SimpleNamespace(
        eos_token_id=eos_token_id,
        convert_tokens_to_ids=mapping.get,
        convert_ids_to_tokens=reverse.get,
    )


def test_qwen_eos_qualification_uses_both_actual_tokenizer_ids():
    observed = tokenizer({"<|endoftext|>": 151643, "<|im_end|>": 151645})
    assert qualification._qwen_eos_stop_ids(observed) == [151643, 151645]


def test_qwen_eos_qualification_rejects_wrong_reverse_roundtrip():
    observed = tokenizer({"<|endoftext|>": 151643, "<|im_end|>": 151645})
    skewed = SimpleNamespace(**{**vars(observed), "convert_ids_to_tokens": lambda _token: "wrong"})
    with pytest.raises(ValueError, match="Qwen EOS"):
        qualification._qwen_eos_stop_ids(skewed)


def test_eos_probes_distinguish_explicit_stops_from_implicit_defaults():
    probes = qualification._eos_sampler_probes([151643, 151645])
    assert len(probes) == 6
    assert all(probe["min_tokens"] == 0 and probe["max_tokens"] == 4 for probe in probes)
    assert [probe["stop_token_ids"] for probe in probes] == [[151643, 151645]] * 4 + [[], []]
    assert [probe["ignore_eos"] for probe in probes] == [False, True] * 3
    assert [probe["allowed_token_ids"] for probe in probes] == [[151643]] * 2 + [[151645]] * 4


@pytest.mark.parametrize(
    "mapping,eos",
    [
        ({"<|im_end|>": 151645}, 151645),
        ({"<|endoftext|>": 151643, "<|im_end|>": 151643}, 151643),
        ({"<|endoftext|>": True, "<|im_end|>": 151645}, 151645),
        ({"<|endoftext|>": 151643, "<|im_end|>": 151645}, 151643),
    ],
)
def test_qwen_eos_qualification_rejects_missing_ambiguous_or_skewed_metadata(mapping, eos):
    with pytest.raises(ValueError, match="Qwen EOS"):
        qualification._qwen_eos_stop_ids(tokenizer(mapping, eos))
