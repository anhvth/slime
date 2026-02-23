from types import SimpleNamespace

import pytest

from slime.utils.arguments import hf_validate_args


def _build_args(rotary_base):
    return SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=64,
        num_layers=64,
        ffn_hidden_size=25600,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        rotary_base=rotary_base,
    )


def _build_hf_config(rope_theta=10000.0, rope_parameters=None):
    return SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=64,
        num_hidden_layers=64,
        intermediate_size=25600,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_theta=rope_theta,
        rope_parameters=rope_parameters,
    )


def test_hf_validate_args_prefers_rope_parameters_rope_theta():
    args = _build_args(rotary_base=1_000_000)
    hf_config = _build_hf_config(rope_theta=10_000.0, rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"})

    hf_validate_args(args, hf_config)


def test_hf_validate_args_falls_back_to_rope_theta_when_rope_parameters_missing():
    args = _build_args(rotary_base=1_000_000)
    hf_config = _build_hf_config(rope_theta=10_000.0, rope_parameters=None)

    with pytest.raises(AssertionError, match="rope_theta in hf config 10000.0"):
        hf_validate_args(args, hf_config)


def test_hf_validate_args_uses_rope_parameters_value_for_mismatch_error():
    args = _build_args(rotary_base=10_000)
    hf_config = _build_hf_config(rope_theta=10_000.0, rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"})

    with pytest.raises(AssertionError, match="rope_parameters\\.rope_theta in hf config 1000000"):
        hf_validate_args(args, hf_config)
