"""TDD: Qwen3.5 GGUF 数值逆变换纯函数测试。"""
import math
import torch
import pytest

from sglang.srt.model_loader.gguf_qwen35moe import (
    norm_sub_one,
    alog_transform,
    conv1d_reorder,
    v_head_tiled_to_grouped,
    get_out_proj_activation_perm,
    apply_f32_transforms,
)


def test_norm_sub_one_applies_attn_norm():
    w = torch.randn(4096)
    r = norm_sub_one(w, "model.layers.0.attn_norm.weight")
    assert torch.allclose(r, w - 1.0)


def test_norm_sub_one_applies_output_norm():
    w = torch.randn(4096)
    r = norm_sub_one(w, "model.norm.weight")
    assert torch.allclose(r, w - 1.0)


def test_norm_sub_one_applies_post_attention_layernorm():
    w = torch.randn(4096)
    r = norm_sub_one(w, "model.layers.0.post_attention_layernorm.weight")
    assert torch.allclose(r, w - 1.0)


def test_norm_sub_one_applies_qk_norm():
    w = torch.randn(128)
    r = norm_sub_one(w, "model.layers.0.attn.q_norm.weight")
    assert torch.allclose(r, w - 1.0)


def test_norm_sub_one_skips_linear_attn_norm():
    """Mamba 内部 GroupNorm 不减 1"""
    w = torch.randn(4096)
    r = norm_sub_one(w, "model.layers.0.linear_attn.norm.weight")
    assert torch.allclose(r, w)


def test_norm_sub_one_skips_ssm_norm():
    w = torch.randn(128)
    r = norm_sub_one(w, "model.layers.0.linear_attn.ssm_norm.weight")
    assert torch.allclose(r, w)


def test_norm_sub_one_pure_no_mutation():
    w = torch.randn(4096)
    w_copy = w.clone()
    _ = norm_sub_one(w, "model.norm.weight")
    assert torch.equal(w, w_copy)


def test_alog_transform():
    w = -torch.exp(torch.randn(128))
    r = alog_transform(w)
    expected = torch.log(-w)
    assert torch.allclose(r, expected)


def test_alog_transform_nan_on_positive():
    w = torch.tensor([1.0, 0.5, -2.0, -0.3])
    r = alog_transform(w)
    assert torch.isfinite(r[2]) and torch.isfinite(r[3])
    assert torch.isnan(r[0]) or torch.isinf(r[0])
    assert torch.isnan(r[1]) or torch.isinf(r[1])


def test_alog_transform_pure():
    w = -torch.exp(torch.randn(16))
    w_copy = w.clone()
    _ = alog_transform(w)
    assert torch.equal(w, w_copy)


def test_conv1d_reorder_shape_and_v_flip():
    # The Qwen3.5 GatedDeltaNet ``ssm_conv1d`` is stored by llama.cpp as
    # ``(out=8192, kernel=4)``; the gguf-py reader already reverses
    # the axis order, so ``tensor.data`` is already the (out, kernel)
    # layout. ``conv1d_reorder`` therefore just returns a contiguous copy
    # without any axis swap. SGLang's Qwen3_5GatedDeltaNet treats this
    # buffer as the concat of two q/k convs plus the v conv and adds
    # the in_channel axis via ``unsqueeze(1)`` in its constructor.
    kernel, out_ch = 4, 12
    w = torch.arange(kernel * out_ch, dtype=torch.float32).reshape(out_ch, kernel)
    r = conv1d_reorder(w, num_v_per_k=2, k_heads=2, head_dim=2)
    assert r.shape == (out_ch, kernel)
    assert torch.equal(r, w)


def test_conv1d_reorder_pure():
    w = torch.randn(4, 8)
    w_copy = w.clone()
    _ = conv1d_reorder(w, num_v_per_k=1, k_heads=2, head_dim=2)
    assert torch.equal(w, w_copy)


def test_v_head_tiled_to_grouped():
    v_per_k, k_heads, head_dim = 4, 8, 128
    w = torch.arange(v_per_k * k_heads * head_dim, dtype=torch.float32).reshape(
        v_per_k, k_heads, head_dim
    )
    r = v_head_tiled_to_grouped(w, v_per_k, k_heads, head_dim)
    expected = w.permute(1, 0, 2).flatten()
    assert torch.equal(r, expected)


def test_v_head_tiled_to_grouped_pure():
    w = torch.randn(4, 8, 128, dtype=torch.float32)
    w_copy = w.clone()
    _ = v_head_tiled_to_grouped(w, 4, 8, 128)
    assert torch.equal(w, w_copy)


def test_get_out_proj_activation_perm_aligns_tiled_activation_to_grouped_layout():
    v_per_k, k_heads, head_dim = 2, 4, 8
    perm = get_out_proj_activation_perm(v_per_k, k_heads, head_dim)
    tiled = torch.arange(v_per_k * k_heads * head_dim, dtype=torch.float32)
    grouped = (
        tiled.reshape(v_per_k, k_heads, head_dim).permute(1, 0, 2).flatten()
    )
    # index_select(-1, perm): out[j] = tiled[perm[j]] must equal grouped layout.
    out_in = tiled.index_select(-1, perm)
    assert torch.equal(out_in, grouped)


# ---- apply_f32_transforms 分派测试 ----


def test_apply_f32_transforms_attn_norm():
    w = torch.randn(4096)
    r = apply_f32_transforms(w, "blk.0.attn_norm.weight")
    assert torch.allclose(r, w - 1.0)


def test_apply_f32_transforms_ssm_a():
    w = -torch.exp(torch.randn(128))
    r = apply_f32_transforms(w, "blk.3.ssm_a.weight")
    expected = torch.log(-w)
    assert torch.allclose(r, expected)


def test_apply_f32_transforms_conv1d():
    w = torch.arange(4 * 2, dtype=torch.float32).reshape(2, 4)
    r = apply_f32_transforms(w, "blk.2.ssm_conv1d.weight")
    # The gguf-py reader already reverses the axis order, so no transpose
    # is needed. SGLang's constructor adds the in_channel axis via
    # unsqueeze(1).
    assert r.shape == (2, 4)
    assert torch.equal(r, w)


def test_apply_f32_transforms_linear_attn_norm_passthrough():
    """Mamba GroupNorm 应为直通"""
    w = torch.randn(4096)
    r = apply_f32_transforms(w, "blk.0.linear_attn_norm.weight")
    assert torch.allclose(r, w)


def test_apply_f32_transforms_ssm_norm_passthrough():
    w = torch.randn(128)
    r = apply_f32_transforms(w, "blk.0.ssm_norm.weight")
    assert torch.allclose(r, w)


def test_apply_f32_transforms_ssm_dt_bias_passthrough():
    """dt_bias 直通"""
    w = torch.randn(128)
    r = apply_f32_transforms(w, "blk.0.ssm_dt.bias")
    assert torch.equal(r, w)


def test_apply_f32_transforms_output_norm():
    w = torch.randn(4096)
    r = apply_f32_transforms(w, "output_norm.weight")
    assert torch.allclose(r, w - 1.0)


def test_apply_f32_transforms_q_weight_passthrough():
    """量化张量直通(无 F32 变换)"""
    w = torch.arange(256, dtype=torch.uint8)
    r = apply_f32_transforms(w, "blk.0.attn_q.weight")
    assert torch.equal(r, w)
