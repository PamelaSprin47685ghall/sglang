"""Checkpoint BF16 ``.weight`` keys must map onto Marlin ``.weight_*`` params."""

from sglang.srt.models.qwen3_5 import _resolve_compressed_tensors_param_name


def test_weight_maps_to_weight_packed():
    params = {
        "model.language_model.layers.3.self_attn.q_proj.weight_packed": None,
    }
    ckpt = "model.language_model.layers.3.self_attn.q_proj.weight"
    assert _resolve_compressed_tensors_param_name(ckpt, params) == (
        "model.language_model.layers.3.self_attn.q_proj.weight_packed"
    )


def test_weight_packed_suffix_still_resolves_to_weight_when_present():
    params = {"model.layers.0.mlp.gate_proj.weight": None}
    ckpt = "model.layers.0.mlp.gate_proj.weight_packed"
    assert _resolve_compressed_tensors_param_name(ckpt, params) == (
        "model.layers.0.mlp.gate_proj.weight"
    )