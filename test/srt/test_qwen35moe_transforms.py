"""TDD: Qwen3.5 GGUF 数值逆变换纯函数测试。"""
import math
import torch
import pytest

from sglang.srt.model_loader.gguf_qwen35moe import (
    norm_sub_one,
    alog_transform,
    conv1d_reorder,
    v_head_tiled_to_grouped,
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
    kernel, qk_ch, v_ch = 4, 4, 8
    out_ch = qk_ch + v_ch
    w = torch.zeros(kernel, out_ch, dtype=torch.float32)
    w[:, qk_ch:] = torch.arange(kernel * v_ch, dtype=torch.float32).reshape(
        kernel, v_ch
    )
    r = conv1d_reorder(w, num_v_per_k=2, k_heads=2, head_dim=2)
    assert r.shape == (kernel, 1, out_ch)
    # Q/K 段不变
    assert torch.equal(r[:, 0, :qk_ch], w[:, :qk_ch])
    # V 段反序 (num_v_per_k=2)
    v_old = w[:, qk_ch:].reshape(kernel, 2, 2, 2)
    v_new = r[:, 0, qk_ch:].reshape(kernel, 2, 2, 2)
    assert torch.equal(v_new, v_old.flip(1))


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
    w = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    r = apply_f32_transforms(w, "blk.2.ssm_conv1d.weight")
    assert r.shape == (4, 1, 2)
    assert r[0, 0, 1] == w[0, 1]


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
