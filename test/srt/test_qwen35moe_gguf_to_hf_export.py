"""GGUF → HF export transforms (GGUF_on_the_fly.txt §3.3–3.4)."""

import torch

from sglang.srt.model_loader.gguf_qwen35moe import (
    apply_gguf_to_hf_weight,
    v_head_tiled_to_grouped,
)


def _cfg():
    c = dict(num_v_per_k=2, k_heads=16, head_k_dim=128, head_v_dim=128)
    c["num_value_heads"] = c["k_heads"] * c["num_v_per_k"]
    return c


def test_in_proj_qkv_v_segment_only_reordered():
    cfg = _cfg()
    key_dim = cfg["k_heads"] * cfg["head_k_dim"]
    value_dim = cfg["num_v_per_k"] * cfg["k_heads"] * cfg["head_v_dim"]
    lead = key_dim * 2 + value_dim
    out_ch = 2048
    w = torch.arange(out_ch * lead, dtype=torch.float32).reshape(out_ch, lead)
    hf = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    r = apply_gguf_to_hf_weight(w.clone(), hf, cfg)
    assert torch.equal(r[:, :key_dim], w[:, :key_dim])
    assert torch.equal(r[:, key_dim : 2 * key_dim], w[:, key_dim : 2 * key_dim])
    v_tiled = w[:, 2 * key_dim :]
    v_exp = (
        v_tiled.reshape(out_ch, cfg["num_v_per_k"], cfg["k_heads"], cfg["head_v_dim"])
        .permute(0, 2, 1, 3)
        .reshape(out_ch, value_dim)
    )
    assert torch.equal(r[:, 2 * key_dim :], v_exp)


def test_out_proj_dim1_v_reorder():
    cfg = _cfg()
    out_ch = 2048
    in_ch = cfg["k_heads"] * cfg["num_v_per_k"] * cfg["head_v_dim"]
    w = torch.randn(out_ch, in_ch)
    hf = "model.language_model.layers.0.linear_attn.out_proj.weight"
    r = apply_gguf_to_hf_weight(w.clone(), hf, cfg)
    assert r.shape == (out_ch, in_ch)
    # grouped input columns: permute tiled cols on dim=1
    tiled = w
    grouped = torch.empty_like(tiled)
    perm = (
        torch.arange(in_ch)
        .reshape(cfg["num_v_per_k"], cfg["k_heads"], cfg["head_v_dim"])
        .permute(1, 0, 2)
        .flatten()
    )
    assert torch.allclose(r, tiled[:, perm])


def test_conv1d_unsqueeze_after_v_reorder():
    w = torch.randn(8192, 4)
    hf = "model.language_model.layers.0.linear_attn.conv1d.weight"
    r = apply_gguf_to_hf_weight(w.clone(), hf, _cfg())
    assert r.shape == (8192, 1, 4)