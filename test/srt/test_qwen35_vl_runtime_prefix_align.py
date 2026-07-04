"""Failing tests for checkpoint model.layers.* to runtime model.language_model.layers.* prefix alignment."""

from types import SimpleNamespace

def test_align_text_stack_prefix_to_runtime_maps_model_layers_to_language_model():
    from sglang.srt.models.qwen3_5 import _align_text_stack_prefix_to_runtime

    # Checkpoint uses flat model.layers prefix
    ckpt_name = "model.layers.3.self_attn.qkv_proj.weight_packed"
    # Runtime params only have model.language_model.layers prefix
    params_dict = {
        "model.language_model.layers.3.self_attn.qkv_proj.weight_packed": None,
        "model.language_model.layers.3.self_attn.o_proj.weight": None,
    }

    aligned = _align_text_stack_prefix_to_runtime(ckpt_name, params_dict)
    assert aligned == "model.language_model.layers.3.self_attn.qkv_proj.weight_packed"
    assert aligned in params_dict


def test_align_text_stack_prefix_to_runtime_handles_non_layer_names():
    from sglang.srt.models.qwen3_5 import _align_text_stack_prefix_to_runtime

    # Top level names should pass through unchanged
    ckpt_name = "model.embed_tokens.weight"
    params_dict = {ckpt_name: None}
    assert _align_text_stack_prefix_to_runtime(ckpt_name, params_dict) == ckpt_name


def test_align_text_stack_prefix_to_runtime_handles_already_aligned_names():
    from sglang.srt.models.qwen3_5 import _align_text_stack_prefix_to_runtime

    # Already aligned names should pass through unchanged
    ckpt_name = "model.language_model.layers.5.mlp.gate.weight"
    params_dict = {ckpt_name: None}
    assert _align_text_stack_prefix_to_runtime(ckpt_name, params_dict) == ckpt_name


def test_align_text_stack_prefix_to_runtime_falls_back_when_no_match():
    from sglang.srt.models.qwen3_5 import _align_text_stack_prefix_to_runtime

    # No matching runtime key, should return original name
    ckpt_name = "model.layers.7.self_attn.qkv_proj.weight_packed"
    params_dict = {"model.language_model.layers.7.self_attn.o_proj.weight": None}
    assert _align_text_stack_prefix_to_runtime(ckpt_name, params_dict) == ckpt_name


def test_align_after_stacked_qkv_fusion_hits_language_model_runtime():
    from sglang.srt.models.qwen3_5 import (
        _align_text_stack_prefix_to_runtime,
        _replace_stacked_shard,
    )

    fused = _replace_stacked_shard(
        "model.layers.11.self_attn.q_proj.weight_packed",
        "qkv_proj",
        "q_proj",
    )
    params_dict = {
        "model.language_model.layers.11.self_attn.qkv_proj.weight_packed": None,
    }
    aligned = _align_text_stack_prefix_to_runtime(fused, params_dict)
    assert aligned == "model.language_model.layers.11.self_attn.qkv_proj.weight_packed"


def test_align_maps_to_language_model_model_layers_prefix():
    from sglang.srt.models.qwen3_5 import _align_text_stack_prefix_to_runtime

    ckpt = "model.layers.3.self_attn.o_proj.weight_packed"
    params = {"language_model.model.layers.3.self_attn.o_proj.weight_packed": None}
    assert _align_text_stack_prefix_to_runtime(ckpt, params) == (
        "language_model.model.layers.3.self_attn.o_proj.weight_packed"
    )


def test_align_suffix_match_when_runtime_prefix_unknown():
    from sglang.srt.models.qwen3_5 import _align_text_stack_prefix_to_runtime

    ckpt = "model.layers.11.self_attn.q_proj.weight_packed"
    params = {
        "foo.bar.model.language_model.layers.11.self_attn.q_proj.weight_packed": None,
    }
    aligned = _align_text_stack_prefix_to_runtime(ckpt, params)
    assert aligned.endswith("layers.11.self_attn.q_proj.weight_packed")
