"""GGUF flat names vs HF nested names in load_weights routing."""

from sglang.srt.models.qwen3_5 import (
    _checkpoint_name_to_model_param,
    _is_gguf_flat_text_checkpoint_name,
)


def test_gguf_flat_layer_maps_directly():
    n = "model.layers.3.linear_attn.out_proj.qweight"
    assert _is_gguf_flat_text_checkpoint_name(n)
    assert (
        _checkpoint_name_to_model_param(n)
        == "model.layers.3.linear_attn.out_proj.qweight"
    )


def test_hf_nested_name_unchanged():
    n = "model.language_model.layers.3.linear_attn.out_proj.weight"
    assert _checkpoint_name_to_model_param(n) == n


def test_safetensors_strip_still_via_normalize_not_mapper():
    n = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert not _is_gguf_flat_text_checkpoint_name(n)
    assert _checkpoint_name_to_model_param(n) == n