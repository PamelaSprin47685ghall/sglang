"""RED: prove qwen35_gguf_dequant_apply_for_load uses reshape where transpose is required.

GGUF dequant output layout is (out_features, in_features) row-major.
The correct conversion to (in_features, out_features) is .t() — reshape
reinterprets the same contiguous memory block and scrambles the data.

Affected branches:
  - in_proj_qkv.weight: dense.reshape(hidden, lead) at line 380
  - in_proj_z.weight:   _reshape_dequant_rows_to_hidden_major at line 369
  - in_proj_a/b.weight: _reshape_dequant_rows_to_hidden_major at line 373

Each test constructs a random (out, in) tensor simulating real GGUF dequant,
computes reference via .t() + apply_gguf_to_hf_weight, then checks that
qwen35_gguf_dequant_apply_for_load produces a result with cosine > 0.99.
Current code fails with cosine ≈ 0.5 (scrambled data).

Additionally, in_proj_z/a/b tests assert that the v-head tiled→grouped
reorder is actually applied (not silently skipped).  The true reference
is computed by manually doing the grouped reorder on the dequant tensor.
Current code skips the reorder (because the numel guard in
apply_gguf_to_hf_weight only matches 1-D tensors), so these assertions
FAIL — proving the reorder is missing.
"""

import torch

from sglang.srt.model_loader.gguf_qwen35moe import (
    apply_gguf_to_hf_weight,
    qwen35_gguf_dequant_apply_for_load,
    qwen35moe_linear_attn_vcfg,
)


def _vcfg():
    return qwen35moe_linear_attn_vcfg(
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()


def test_dequant_apply_in_proj_qkv_matches_transpose_not_reshape():
    """in_proj_qkv: (lead, hidden) → .t() → (hidden, lead) + v-head reorder."""
    cfg = _vcfg()
    key_dim = cfg["k_heads"] * cfg["head_k_dim"]        # 2048
    value_dim = cfg["num_value_heads"] * cfg["head_v_dim"]  # 4096
    hidden, lead = 2048, key_dim * 2 + value_dim          # 8192
    hf = "model.layers.0.linear_attn.in_proj_qkv.weight"

    # Real GGUF dequant: row-major (out, in) = (lead, hidden)
    dequant = torch.randn(lead, hidden)

    # Reference: transpose then apply v-head reorder
    ref = apply_gguf_to_hf_weight(dequant.t().contiguous().clone(), hf, cfg)

    # Bug path: reshape instead of transpose
    got = qwen35_gguf_dequant_apply_for_load(dequant.clone(), hf, cfg)

    assert got.shape == (hidden, lead), f"shape mismatch: {got.shape}"
    cos = _cosine(got, ref)
    assert cos > 0.99, f"in_proj_qkv cosine={cos:.4f}, expected > 0.99 (reshape scrambles data)"


def _grouped_ref_in_proj_z(dequant, cfg):
    """Compute true grouped reference for in_proj_z.

    dequant: (value_dim, hidden) GGUF dequant output — row-major (out, in).
    Returns: (hidden, value_dim) with v-head tiled→grouped reorder applied.

    The tiled layout is (vpk, kh, hvd, hidden); the grouped layout is
    (kh, vpk, hvd, hidden).  Permute(1, 0, 2, 3) converts between them.
    """
    vpk = cfg["num_v_per_k"]
    kh = cfg["k_heads"]
    hvd = cfg["head_v_dim"]
    grouped = dequant.reshape(vpk, kh, hvd, -1).permute(1, 0, 2, 3).reshape(-1, dequant.shape[1])
    return grouped.t().contiguous()


def test_dequant_apply_in_proj_z_matches_transpose_not_reshape():
    """in_proj_z: (value_dim, hidden) → .t() → (hidden, value_dim) + v-head reorder."""
    cfg = _vcfg()
    hidden = 2048
    value_dim = cfg["num_value_heads"] * cfg["head_v_dim"]  # 4096
    hf = "model.layers.0.linear_attn.in_proj_z.weight"

    dequant = torch.randn(value_dim, hidden)
    ref = apply_gguf_to_hf_weight(dequant.t().contiguous().clone(), hf, cfg)
    got = qwen35_gguf_dequant_apply_for_load(dequant.clone(), hf, cfg)

    assert got.shape == (hidden, value_dim), f"shape mismatch: {got.shape}"
    cos = _cosine(got, ref)
    assert cos > 0.99, f"in_proj_z cosine={cos:.4f}, expected > 0.99"

    # TRUE reference: v-head tiled→grouped reorder must be applied.
    # Current code skips it (numel guard in apply_gguf_to_hf_weight only
    # matches 1-D tensors), so this assertion FAILS (RED).
    true_ref = _grouped_ref_in_proj_z(dequant.clone(), cfg)
    cos_true = _cosine(got, true_ref)
    assert cos_true > 0.99, (
        f"in_proj_z grouped reorder missing: cosine={cos_true:.4f}, expected > 0.99"
    )


def _grouped_ref_in_proj_a_b(dequant, cfg):
    """Compute true grouped reference for in_proj_a/b.

    dequant: (nvh, hidden) GGUF dequant output — row-major (out, in).
    Returns: (hidden, nvh) with v-head tiled→grouped reorder applied (hvd=1).

    The tiled layout is (vpk, kh, 1, hidden); the grouped layout is
    (kh, vpk, 1, hidden).  Permute(1, 0, 2, 3) converts between them.
    """
    vpk = cfg["num_v_per_k"]
    kh = cfg["k_heads"]
    grouped = dequant.reshape(vpk, kh, 1, -1).permute(1, 0, 2, 3).reshape(-1, dequant.shape[1])
    return grouped.t().contiguous()


def test_dequant_apply_in_proj_a_matches_transpose_not_reshape():
    """in_proj_a: (num_value_heads, hidden) → .t() → (hidden, num_value_heads) + v-head reorder."""
    cfg = _vcfg()
    hidden = 2048
    nvh = cfg["num_value_heads"]  # 32
    hf = "model.layers.0.linear_attn.in_proj_a.weight"

    dequant = torch.randn(nvh, hidden)
    ref = apply_gguf_to_hf_weight(dequant.t().contiguous().clone(), hf, cfg)
    got = qwen35_gguf_dequant_apply_for_load(dequant.clone(), hf, cfg)

    assert got.shape == (hidden, nvh), f"shape mismatch: {got.shape}"
    cos = _cosine(got, ref)
    assert cos > 0.99, f"in_proj_a cosine={cos:.4f}, expected > 0.99"

    # TRUE reference: v-head tiled→grouped reorder must be applied (hvd=1).
    # Current code skips it, so this assertion FAILS (RED).
    true_ref = _grouped_ref_in_proj_a_b(dequant.clone(), cfg)
    cos_true = _cosine(got, true_ref)
    assert cos_true > 0.99, (
        f"in_proj_a grouped reorder missing: cosine={cos_true:.4f}, expected > 0.99"
    )


def test_dequant_apply_in_proj_b_matches_transpose_not_reshape():
    """in_proj_b: (num_value_heads, hidden) → .t() → (hidden, num_value_heads) + v-head reorder."""
    cfg = _vcfg()
    hidden = 2048
    nvh = cfg["num_value_heads"]  # 32
    hf = "model.layers.0.linear_attn.in_proj_b.weight"

    dequant = torch.randn(nvh, hidden)
    ref = apply_gguf_to_hf_weight(dequant.t().contiguous().clone(), hf, cfg)
    got = qwen35_gguf_dequant_apply_for_load(dequant.clone(), hf, cfg)

    assert got.shape == (hidden, nvh), f"shape mismatch: {got.shape}"
    cos = _cosine(got, ref)
    assert cos > 0.99, f"in_proj_b cosine={cos:.4f}, expected > 0.99"

    # TRUE reference: v-head tiled→grouped reorder must be applied (hvd=1).
    # Current code skips it, so this assertion FAILS (RED).
    true_ref = _grouped_ref_in_proj_a_b(dequant.clone(), cfg)
    cos_true = _cosine(got, true_ref)
    assert cos_true > 0.99, (
        f"in_proj_b grouped reorder missing: cosine={cos_true:.4f}, expected > 0.99"
    )
