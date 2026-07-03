"""Qwen3.5 MoE VL full-attn checkpoint names map to layer-level params."""


def test_vl_full_attention_checkpoint_strips_self_attn_segment_when_param_exists():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(layers_block_type=None, layer_types=["full_attention"])
    params = {"model.layers.0.qkv_proj.weight": None}
    name = "model.layers.0.self_attn.q_proj.weight"

    assert (
        _map_vl_lm_layer_param_to_sglang_attn_submodule(name, cfg, params)
        == "model.layers.0.q_proj.weight"
    )


def test_flat_hf_slice_preserves_self_attn_until_param_resolution():
    from sglang.srt.models.qwen3_5 import _map_hf_flat_text_to_vl_lm_param

    raw = "model.layers.15.self_attn.k_norm.weight"
    mapped = _map_hf_flat_text_to_vl_lm_param(raw)
    assert mapped == "model.layers.15.self_attn.k_norm.weight"
    assert ".self_attn." in mapped


def test_layer_norm_remapped_under_linear_attn_submodule():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(
        layers_block_type=None,
        layer_types=None,
        full_attention_interval=4,
    )
    flat = "model.layers.0.input_layernorm.weight"
    params = {
        "model.layers.0.linear_attn.input_layernorm.weight": None,
    }
    assert (
        _map_vl_lm_layer_param_to_sglang_attn_submodule(flat, cfg, params)
        == "model.layers.0.linear_attn.input_layernorm.weight"
    )


def test_mlp_weights_not_nested_under_linear_attn():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(
        layers_block_type=None,
        layer_types=None,
        full_attention_interval=4,
    )
    gate = "model.layers.0.mlp.shared_expert.gate_proj.weight"
    assert _map_vl_lm_layer_param_to_sglang_attn_submodule(gate, cfg, {}) == gate


def test_qwen35_moe_vl_disables_parent_hf_mapper():
    from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
    from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

    assert (
        Qwen3_5MoeForConditionalGeneration.hf_to_sglang_mapper.orig_to_new_prefix
        != Qwen3VLForConditionalGeneration.hf_to_sglang_mapper.orig_to_new_prefix
        or Qwen3_5MoeForConditionalGeneration.hf_to_sglang_mapper.orig_to_new_prefix
        == {}
    )
