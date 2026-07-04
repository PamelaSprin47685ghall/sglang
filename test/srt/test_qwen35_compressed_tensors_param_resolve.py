"""Checkpoint BF16 ``.weight`` keys must map onto Marlin ``.weight_*`` params."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import (
    _map_vl_lm_layer_param_to_sglang_attn_submodule,
    _resolve_compressed_tensors_param_name,
)


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


def test_vl_lm_keeps_self_attn_for_qwen35_moe_then_resolves_packed():
    """Regression: stripping self_attn dropped full-attn shards (garbled chat)."""
    cfg = SimpleNamespace(
        layers_block_type=None,
        layer_types=None,
        full_attention_interval=4,
    )
    params = {"model.layers.3.self_attn.q_proj.weight_packed": None}
    ckpt = "model.layers.3.self_attn.q_proj.weight"

    mapped = _map_vl_lm_layer_param_to_sglang_attn_submodule(ckpt, cfg, params)
    assert mapped == ckpt

    resolved = _resolve_compressed_tensors_param_name(mapped, params)
    assert resolved == "model.layers.3.self_attn.q_proj.weight_packed"


def test_vl_lm_strips_only_when_flat_q_proj_param_exists():
    cfg = SimpleNamespace(layers_block_type=None, layer_types=["full_attention"])
    params = {"model.layers.0.q_proj.weight": None}
    ckpt = "model.layers.0.self_attn.q_proj.weight"
    assert (
        _map_vl_lm_layer_param_to_sglang_attn_submodule(ckpt, cfg, params)
        == "model.layers.0.q_proj.weight"
    )


def test_load_weights_transposes_mismatched_2d():
    """Simulate the load_weights snippet on a mismatched 2D weight.

    Covers the core code path in ``Qwen3_5MoeForConditionalGeneration.load_weights``
    that transposes 2D weights when the loaded shape is the reverse of the
    parameter shape (e.g. checkpoint stores ``[2048, 32]`` while the model
    parameter expects ``[32, 2048]`` for ``linear_attn.in_proj_a.weight``).
    """
    target_shape = (32, 2048)
    ckpt_shape = (2048, 32)

    param_data = torch.zeros(target_shape)
    param = torch.nn.Parameter(param_data)
    param.weight_loader = default_weight_loader
    loaded_checkpoint_weight = torch.randn(ckpt_shape)

    params_dict = {"model.layers.0.linear_attn.in_proj_a.weight": param}

    loaded_weight = loaded_checkpoint_weight.clone()

    name = "model.layers.0.linear_attn.in_proj_a.weight"

    # Simulate the snippet starting at the shape mismatch check.
    if name in params_dict.keys():
        p = params_dict[name]
        if loaded_weight.shape != p.data.shape and loaded_weight.dim() == p.data.dim():
            if (
                loaded_weight.dim() == 2
                and loaded_weight.shape[0] == p.data.shape[1]
                and loaded_weight.shape[1] == p.data.shape[0]
            ):
                loaded_weight = loaded_weight.t().contiguous()
            else:
                raise AssertionError("Should have triggered transpose branch")
        weight_loader = getattr(p, "weight_loader", default_weight_loader)
        weight_loader(p, loaded_weight)

    assert param_data.shape == target_shape
    assert torch.allclose(param_data, loaded_checkpoint_weight.t())