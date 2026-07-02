"""GGUF flat names align with Qwen3.5 MoE VL ``named_parameters`` as ``model.*``."""
from sglang.srt.models.qwen3_5 import (
    _map_gguf_text_name_to_vl_param,
    _normalize_qwen35_checkpoint_name,
)


def test_flat_embed_maps_directly():
    assert (
        _map_gguf_text_name_to_vl_param("model.embed_tokens.qweight")
        == "model.embed_tokens.qweight"
    )


def test_flat_layer_maps_directly():
    name = _map_gguf_text_name_to_vl_param(
        "model.layers.3.mlp.experts.w13_qweight"
    )
    assert name == "model.layers.3.mlp.experts.w13_qweight"


def test_flat_norm_maps_directly():
    assert (
        _map_gguf_text_name_to_vl_param("model.norm.qweight")
        == "model.norm.qweight"
    )


def test_lm_head_maps_to_top_level():
    assert _map_gguf_text_name_to_vl_param("model.lm_head.weight") == "lm_head.weight"


def test_hf_language_model_name_not_double_prefixed():
    hf = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert _map_gguf_text_name_to_vl_param(hf) == hf


def test_causal_lm_normalize_strips_language_model():
    assert (
        _normalize_qwen35_checkpoint_name(
            "model.language_model.layers.1.linear_attn.in_proj_qkv.weight"
        )
        == "model.layers.1.linear_attn.in_proj_qkv.weight"
    )