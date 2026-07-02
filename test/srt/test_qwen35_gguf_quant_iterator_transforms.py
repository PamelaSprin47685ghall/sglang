"""GGUF on-the-fly: Q6_K linear_attn must use dequant + apply_gguf_to_hf_weight like export."""

import gguf
import numpy as np
import torch

from sglang.srt.model_loader.gguf_qwen35moe import (
    apply_gguf_to_hf_weight,

    qwen35_gguf_dequant_apply_for_load,
    qwen35moe_gguf_on_the_fly_needs_hf_transform,
    qwen35moe_linear_attn_vcfg,
)
from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator


def _vcfg():
    return qwen35moe_linear_attn_vcfg(
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )


def test_on_the_fly_needs_transform_for_in_proj_qkv_not_self_attn():
    cfg = _vcfg()
    assert qwen35moe_gguf_on_the_fly_needs_hf_transform(
        "model.layers.0.linear_attn.in_proj_qkv.weight", cfg
    )
    assert not qwen35moe_gguf_on_the_fly_needs_hf_transform(
        "model.layers.0.self_attn.q_proj.weight", cfg
    )


def test_dequant_apply_matches_dim1_export_on_loader_layout_in_proj_qkv():
    cfg = _vcfg()
    key_dim = cfg["k_heads"] * cfg["head_k_dim"]
    value_dim = cfg["num_value_heads"] * cfg["head_v_dim"]
    out_ch, lead = 64, key_dim * 2 + value_dim
    loader_layout = torch.randn(lead, out_ch)
    hf = "model.layers.0.linear_attn.in_proj_qkv.weight"
    ref = apply_gguf_to_hf_weight(loader_layout.t().clone(), hf, cfg).t().contiguous()
    got = qwen35_gguf_dequant_apply_for_load(loader_layout.clone(), hf, cfg)
    assert torch.allclose(got, ref)
    assert got.shape[0] == lead


def test_in_proj_z_hidden_major_dequant_transposed_before_v_reorder():
    cfg = _vcfg()
    value_dim = cfg["num_value_heads"] * cfg["head_v_dim"]
    hidden = 2048
    hf = "model.layers.0.linear_attn.in_proj_z.weight"
    row_major = qwen35_gguf_dequant_apply_for_load(
        torch.randn(value_dim, hidden), hf, cfg
    )
    col_major = qwen35_gguf_dequant_apply_for_load(
        torch.randn(hidden, value_dim), hf, cfg
    )
    assert row_major.shape == (value_dim, hidden)
    assert col_major.shape == (value_dim, hidden)
    assert not torch.allclose(row_major, col_major)


def test_on_the_fly_skips_out_proj_runtime_perm_path():
    cfg = _vcfg()
    assert not qwen35moe_gguf_on_the_fly_needs_hf_transform(
        "model.layers.0.linear_attn.out_proj.weight", cfg
    )


def test_iterator_yields_bf16_in_proj_qkv_for_layer0_slice():
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    reader = gguf.GGUFReader(path)
    gt = "blk.0.attn_qkv.weight"
    hf = "model.layers.0.linear_attn.in_proj_qkv.weight"
    mapping = {gt: hf}
    cfg = _vcfg()
    items = {
        n: t
        for n, t in gguf_quant_weights_iterator(
            path,
            mapping,
            qwen35_linear_attn_vcfg=cfg,
            qwen35_dense_storage_dtype=torch.bfloat16,
        )
    }
    qname = hf.replace("weight", "qweight")
    assert qname in items
    w = items[qname]
    assert w.dtype == torch.bfloat16
    assert w.shape == (2048, 8192)
    t = next(x for x in reader.tensors if x.name == gt)
    ref_dense = torch.from_numpy(
        np.array(gguf.dequantize(np.array(t.data), t.tensor_type))
    ).float()
    ref = qwen35_gguf_dequant_apply_for_load(ref_dense, hf, cfg)
    assert torch.allclose(w.float(), ref.t(), atol=1e-2, rtol=1e-2)


def test_iterator_conv1d_matches_export_v_reorder():
    from gguf import GGUFReader

    from sglang.srt.model_loader.gguf_qwen35moe import apply_gguf_to_hf_weight

    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    gt = "blk.0.ssm_conv1d.weight"
    hf = "model.layers.0.linear_attn.conv1d.weight"
    cfg = _vcfg()
    items = dict(
        gguf_quant_weights_iterator(path, {gt: hf}, qwen35_linear_attn_vcfg=cfg)
    )
    w = items[hf]
    assert w.ndim == 3 and w.shape[1] == 1
    t = next(x for x in GGUFReader(path).tensors if x.name == gt)
    raw = torch.from_numpy(np.array(t.data, copy=True)).float()
    ref = apply_gguf_to_hf_weight(raw, hf, cfg)
    assert torch.allclose(w.float(), ref.float(), atol=1e-5)


def test_in_proj_qkv_dim1_slices_transpose_for_linear_weight_layout():
    cfg = _vcfg()
    hidden = 2048
    key_dim = cfg["k_heads"] * cfg["head_k_dim"]
    value_dim = cfg["num_value_heads"] * cfg["head_v_dim"]
    lead = key_dim * 2 + value_dim
    loaded = torch.randn(hidden, lead)
    slices = (
        (0, key_dim),
        (key_dim, key_dim * 2),
        (key_dim * 2, key_dim * 2 + value_dim),
    )
    for sl in slices:
        piece = loaded[:, sl[0] : sl[1]].contiguous()
        w = piece.t().contiguous()
        assert w.shape[0] == sl[1] - sl[0]
        assert w.shape[1] == hidden
        torch.mm(torch.randn(4, hidden), w.t())