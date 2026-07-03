"""HF GPU slice checkpoints use flat ``model.layers.*``; VL MoE params use ``model.language_model.*``."""

from sglang.srt.models.qwen3_5 import _map_hf_flat_text_to_vl_lm_param


def test_layers_prefix_unchanged_for_vl_moe():
    n = "model.layers.4.mlp.shared_expert.gate_proj.weight"
    assert _map_hf_flat_text_to_vl_lm_param(n) == n


def test_embed_and_norm_unchanged():
    assert _map_hf_flat_text_to_vl_lm_param("model.embed_tokens.weight") == (
        "model.embed_tokens.weight"
    )
    assert _map_hf_flat_text_to_vl_lm_param("model.norm.weight") == "model.norm.weight"


def test_nested_language_model_stripped_to_model():
    n = "model.language_model.layers.0.mlp.gate.weight"
    assert _map_hf_flat_text_to_vl_lm_param(n) == "model.layers.0.mlp.gate.weight"


def test_lm_head_unchanged():
    assert _map_hf_flat_text_to_vl_lm_param("lm_head.weight") == "lm_head.weight"