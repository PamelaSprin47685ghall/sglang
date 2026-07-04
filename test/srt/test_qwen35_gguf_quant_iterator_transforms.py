"""GGUF on-the-fly: Q6_K linear_attn must use dequant + apply_gguf_to_hf_weight like export."""

import gguf
import numpy as np
import torch
from safetensors import safe_open

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
    hidden, lead = 2048, key_dim * 2 + value_dim
    hf = "model.layers.0.linear_attn.in_proj_qkv.weight"
    logical = torch.randn(hidden, lead)
    dequant_rows = logical.t().contiguous()
    ref = apply_gguf_to_hf_weight(logical.clone(), hf, cfg)
    got = qwen35_gguf_dequant_apply_for_load(dequant_rows.clone(), hf, cfg)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    assert got.shape == (hidden, lead)


def _layer0_apply_gguf_ref(gt: str, hf: str, cfg):
    from gguf import GGUFReader

    from sglang.srt.model_loader.gguf_qwen35moe import (
        qwen35_gguf_dequant_apply_for_load,
        qwen35_gguf_dequant_from_reader_tensor,
    )

    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    t = next(x for x in GGUFReader(path).tensors if x.name == gt)
    dense = qwen35_gguf_dequant_from_reader_tensor(t)
    return qwen35_gguf_dequant_apply_for_load(dense, hf, cfg).float()


def test_in_proj_z_gguf_slice_dequant_is_out_in_rows():
    cfg = _vcfg()
    hf = "model.layers.0.linear_attn.in_proj_z.weight"
    ref = _layer0_apply_gguf_ref("blk.0.attn_gate.weight", hf, cfg)
    hidden = 2048
    value_dim = cfg["num_value_heads"] * cfg["head_v_dim"]
    assert ref.shape == (hidden, value_dim)


def test_on_the_fly_skips_out_proj_runtime_perm_path():
    cfg = _vcfg()
    assert not qwen35moe_gguf_on_the_fly_needs_hf_transform(
        "model.layers.0.linear_attn.out_proj.weight", cfg
    )


def test_out_proj_q6k_dequant_meta_layout_matches_export():
    import gguf as gguf_mod
    from sglang.srt.model_loader.gguf_qwen35moe import qwen35_gguf_dequant_from_reader_tensor, get_out_proj_activation_perm

    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    t = next(x for x in gguf_mod.GGUFReader(path).tensors if x.name == "blk.0.ssm_out.weight")
    dense = qwen35_gguf_dequant_from_reader_tensor(t).float()
    perm = get_out_proj_activation_perm(2, 16, 128)
    w = dense[:, perm].contiguous()
    exp_key = "model.language_model.layers.0.linear_attn.out_proj.weight"
    with safe_open(
        "/home/kunweiz/Desktop/Ornith/ornith-gpu-bf16-from-gguf/model-gpu-from-gguf.safetensors",
        framework="pt",
    ) as f:
        ref = f.get_tensor(exp_key).float()
    assert w.shape == ref.shape
    assert torch.allclose(w, ref, atol=0.05, rtol=0.02)


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
    assert ref.shape == (2048, 8192)
    assert torch.allclose(w.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_in_proj_qkv_iterator_matches_export_safetensors():
    from safetensors import safe_open

    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    exp_path = (
        "/home/kunweiz/Desktop/Ornith/ornith-gpu-bf16-from-gguf/model-gpu-from-gguf.safetensors"
    )
    gt = "blk.0.attn_qkv.weight"
    hf = "model.layers.0.linear_attn.in_proj_qkv.weight"
    cfg = _vcfg()
    items = dict(
        gguf_quant_weights_iterator(
            path,
            {gt: hf},
            qwen35_linear_attn_vcfg=cfg,
            qwen35_dense_storage_dtype=torch.bfloat16,
        )
    )
    w = items[hf.replace("weight", "qweight")].float()
    exp_key = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    with safe_open(exp_path, framework="pt") as f:
        exp = f.get_tensor(exp_key).float()
    assert w.shape == exp.shape == (2048, 8192)
    assert torch.allclose(w, exp, atol=0.01, rtol=0.01)


def test_iterator_in_proj_z_matches_apply_gguf_hidden_major():
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    gt = "blk.0.attn_gate.weight"
    hf = "model.layers.0.linear_attn.in_proj_z.weight"
    cfg = _vcfg()
    items = dict(
        gguf_quant_weights_iterator(
            path, {gt: hf}, qwen35_linear_attn_vcfg=cfg, qwen35_dense_storage_dtype=torch.bfloat16
        )
    )
    w = items[hf.replace("weight", "qweight")].float()
    ref = _layer0_apply_gguf_ref(gt, hf, cfg)
    assert ref.shape == (2048, 4096)
    assert w.shape == (4096, 2048)
    assert torch.allclose(w.float(), ref.t().float(), atol=0.05, rtol=0.02)


def test_iterator_in_proj_a_matches_apply_gguf_out_in_rows():
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    gt = "blk.0.ssm_alpha.weight"
    hf = "model.layers.0.linear_attn.in_proj_a.weight"
    cfg = _vcfg()
    items = dict(
        gguf_quant_weights_iterator(
            path, {gt: hf}, qwen35_linear_attn_vcfg=cfg, qwen35_dense_storage_dtype=torch.bfloat16
        )
    )
    w = items[hf.replace("weight", "qweight")].float()
    ref = _layer0_apply_gguf_ref(gt, hf, cfg)
    assert ref.shape == (2048, 32)
    assert w.shape == (32, 2048)
    assert torch.allclose(w.float(), ref.t().float(), atol=0.05, rtol=0.02)


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


def _shexp_q6k_fused_vs_dequant(gt: str, in_dim: int, out_dim: int, atol: float = 0.02):
    import numpy as np
    from gguf import GGUFReader

    from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

    if not torch.cuda.is_available():
        return
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    t = next(x for x in GGUFReader(path).tensors if x.name == gt)
    raw = torch.tensor(np.array(t.data)).cuda()
    dense = (
        torch.from_numpy(np.array(gguf.dequantize(np.array(t.data), t.tensor_type)))
        .float()
        .cuda()
    )
    assert dense.shape == (out_dim, in_dim)
    x = torch.randn(2, in_dim, dtype=torch.float16, device="cuda")
    y = fused_mul_mat_gguf(x, raw, int(t.tensor_type))
    y_ref = x.float() @ dense.T
    assert torch.allclose(y.float(), y_ref, atol=atol, rtol=0.01)


def test_shared_expert_gate_q6k_fused_matches_dequant():
    _shexp_q6k_fused_vs_dequant("blk.0.ffn_gate_shexp.weight", 2048, 512)


def test_shared_expert_up_q6k_fused_matches_dequant():
    _shexp_q6k_fused_vs_dequant("blk.0.ffn_up_shexp.weight", 2048, 512)


def test_shared_expert_down_q6k_fused_matches_dequant_row_layout():
    import numpy as np
    from gguf import GGUFReader

    from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

    if not torch.cuda.is_available():
        return
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    gt = "blk.0.ffn_down_shexp.weight"
    t = next(x for x in GGUFReader(path).tensors if x.name == gt)
    raw = torch.tensor(np.array(t.data)).cuda()
    dense = (
        torch.from_numpy(np.array(gguf.dequantize(np.array(t.data), t.tensor_type)))
        .float()
        .cuda()
    )
    assert dense.shape == (2048, 512)
    x = torch.randn(2, 512, dtype=torch.float16, device="cuda")
    y = fused_mul_mat_gguf(x, raw, int(t.tensor_type))
    y_ref = x.float() @ dense.T
    assert torch.allclose(y.float(), y_ref, atol=0.02, rtol=0.01)


def test_mlp_gate_f32_loader_layout_matches_linear():
    from sglang.srt.model_loader.gguf_qwen35moe import qwen35moe_gguf_to_hf

    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    gt = "blk.0.ffn_gate_inp.weight"
    hf = qwen35moe_gguf_to_hf(gt).replace("model.language_model.", "model.")
    cfg = _vcfg()
    items = dict(
        gguf_quant_weights_iterator(path, {gt: hf}, qwen35_linear_attn_vcfg=cfg)
    )
    w = items[hf]
    assert w.shape == (256, 2048)
    x = torch.randn(2, 2048)
    torch.mm(x, w.T)


def test_layer3_attn_q_q6k_fused_matches_dequant():
    import numpy as np
    from gguf import GGUFReader

    from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

    if not torch.cuda.is_available():
        return
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    gt = "blk.3.attn_q.weight"
    t = next(x for x in GGUFReader(path).tensors if x.name == gt)
    raw = torch.tensor(np.array(t.data)).cuda()
    qt = int(t.tensor_type)
    dense = (
        torch.from_numpy(np.array(gguf.dequantize(np.array(t.data), t.tensor_type)))
        .float()
        .cuda()
    )
    assert dense.shape == (8192, 2048)
    x = torch.randn(2, 2048, dtype=torch.float16, device="cuda")
    y = fused_mul_mat_gguf(x, raw, qt)
    y_ref = x.float() @ dense.T
    assert torch.allclose(y.float(), y_ref, atol=0.05, rtol=0.01)


def test_out_proj_q6k_fused_matches_dense_with_activation_perm():
    import numpy as np
    from gguf import GGUFReader

    from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf
    from sglang.srt.model_loader.gguf_qwen35moe import get_out_proj_activation_perm

    if not torch.cuda.is_available():
        return
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    t = next(x for x in GGUFReader(path).tensors if x.name == "blk.0.ssm_out.weight")
    raw = torch.tensor(np.array(t.data)).cuda()
    qt = int(t.tensor_type)
    dense = (
        torch.from_numpy(np.array(gguf.dequantize(np.array(t.data), t.tensor_type)))
        .float()
        .cuda()
    )
    perm = get_out_proj_activation_perm(2, 16, 128).cuda()
    x = torch.randn(2, 4096, dtype=torch.float16, device="cuda")
    x_perm = x.index_select(-1, perm)
    y_fused = fused_mul_mat_gguf(x_perm, raw, qt)
    y_dense = x_perm.float() @ dense.T
    assert torch.allclose(y_fused.float(), y_dense, atol=0.15, rtol=0.02)


def test_layer3_o_proj_q6k_fused_matches_dequant():
    import numpy as np
    from gguf import GGUFReader

    from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

    if not torch.cuda.is_available():
        return
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    gt = "blk.3.attn_output.weight"
    t = next(x for x in GGUFReader(path).tensors if x.name == gt)
    raw = torch.tensor(np.array(t.data)).cuda()
    qt = int(t.tensor_type)
    dense = (
        torch.from_numpy(np.array(gguf.dequantize(np.array(t.data), t.tensor_type)))
        .float()
        .cuda()
    )
    assert dense.shape == (2048, 4096)
    x = torch.randn(2, 4096, dtype=torch.float16, device="cuda")
    y = fused_mul_mat_gguf(x, raw, qt)
    y_ref = x.float() @ dense.T
    assert torch.allclose(y.float(), y_ref, atol=0.06, rtol=0.01)


def test_layer3_qk_norm_f32_iterator_applies_minus_one():
    from gguf import GGUFReader

    from sglang.srt.model_loader.gguf_qwen35moe import (
        apply_f32_transforms_hf,
        qwen35moe_gguf_to_hf,
    )

    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    for gt in ("blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight"):
        t = next(x for x in GGUFReader(path).tensors if x.name == gt)
        raw = torch.from_numpy(np.array(t.data, copy=True)).float()
        hf = qwen35moe_gguf_to_hf(gt).replace("model.language_model.", "model.")
        ref = apply_f32_transforms_hf(raw, hf)
        items = dict(
            gguf_quant_weights_iterator(path, {gt: hf}, qwen35_linear_attn_vcfg=_vcfg())
        )
        assert torch.allclose(items[hf].float(), ref.float(), atol=1e-6)


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


def test_token_embd_q6k_fused_embedding_matches_dequant_for_france_prompt_ids():
    import numpy as np
    from gguf import GGUFReader

    from sglang.srt.layers.quantization.gguf import apply_gguf_embedding

    if not torch.cuda.is_available():
        return
    path = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"
    gt = "token_embd.weight"
    t = next(x for x in GGUFReader(path).tensors if x.name == gt)
    raw_qweight = torch.tensor(np.array(t.data, copy=True)).cuda()
    qt = int(t.tensor_type)
    hidden_size = 2048
    dense = (
        torch.from_numpy(np.array(gguf.dequantize(np.array(t.data, copy=True), t.tensor_type)))
        .float()
        .cuda()
    )
    assert dense.shape == (248320, hidden_size)
    ids = torch.tensor([760, 6511, 314, 9338, 369], dtype=torch.long, device="cuda")
    emb = apply_gguf_embedding(ids, raw_qweight, qt, hidden_size, dtype=torch.float32)
    ref = dense[ids]
    assert emb.shape == (5, hidden_size)
    assert torch.allclose(emb.float(), ref.float(), atol=0.02)