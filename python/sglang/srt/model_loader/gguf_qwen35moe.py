from __future__ import annotations

import re
from typing import Optional


_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_TOP_LEVEL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

_NEXTN_SUFFIXES = {
    "nextn.eh_proj.weight": "model.mtp.fc.weight",
    "nextn.enorm.weight": "model.mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "model.mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "model.mtp.norm.weight",
    "nextn.eh_norm.weight": "model.mtp.eh_norm.weight",
}

_BLOCK_SUFFIXES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "self_attn.ffn_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "q_norm.weight",
    "attn_k_norm.weight": "k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_a.weight": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "linear_attn_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert.gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
}

_EXPERT_SUFFIXES = {
    "ffn_gate_exps.weight",
    "ffn_down_exps.weight",
    "ffn_up_exps.weight",
}

_FUSED_EXPERT_TARGETS = {
    "ffn_gate_exps.weight": "mlp.experts.gate_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.down_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.up_proj.weight",
}


def _map_shared_expert_gguf_checkpoint_name(name: str) -> str:
    """Map shared_expert GGUF checkpoint names to SGLang model parameter names.

    GGUF stores shared_expert ``gate_proj`` and ``up_proj`` as separate tensors,
    but the model's ``Qwen2MoeMLP`` (``shared_expert``) exposes a single fused
    ``gate_up_proj`` (``MergedColumnParallelLinear``).  Similarly, the GGUF
    name for the shared-expert routing gate is ``shared_expert.gate.weight``,
    but the model parameter is ``shared_expert_gate.weight`` (one level, not
    nested under ``shared_expert.*``).

    Call this *before* the ``stacked_params_mapping`` loop in ``load_weights``
    so the fused name no longer contains ``gate_proj`` or ``up_proj``, causing
    ``stacked_params_mapping`` to skip it naturally (the guard
    ``"shared_expert" in name`` remains as a safety net).
    """
    # Only rename routing gate; gate_proj/up_proj are loaded via shard_id
    # into ``gate_up_proj`` in ``load_weights`` (both map to same target).
    name = name.replace(".shared_expert.gate.", ".shared_expert_gate.")
    return name


def is_qwen35moe_expert(gguf_name: str) -> bool:
    match = _BLOCK_RE.match(gguf_name)
    if not match:
        return False
    suffix = match.group(2)
    return suffix in _EXPERT_SUFFIXES or any(
        suffix.startswith(prefix[:-7] + ".") for prefix in _EXPERT_SUFFIXES
    )


def qwen35moe_gguf_to_hf(gguf_name: str) -> Optional[str]:
    if gguf_name in _TOP_LEVEL:
        return _TOP_LEVEL[gguf_name]

    if gguf_name in _NEXTN_SUFFIXES:
        return _NEXTN_SUFFIXES[gguf_name]

    match = _BLOCK_RE.match(gguf_name)
    if not match:
        return None

    layer, suffix = match.groups()
    if suffix in _NEXTN_SUFFIXES:
        return _NEXTN_SUFFIXES[suffix]
    if suffix in _FUSED_EXPERT_TARGETS:
        return f"model.layers.{layer}.{_FUSED_EXPERT_TARGETS[suffix]}"
    target = _BLOCK_SUFFIXES.get(suffix)
    if target is None:
        return None
    return f"model.layers.{layer}.{target}"


def make_qwen35moe_gguf_map(
    num_hidden_layers: int,
    num_experts: int,
    num_experts_per_tok: int,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_key_value_heads: int,
    num_attention_heads: int,
    num_expert_shared: int = 1,
) -> dict[str, str]:
    out = {}
    for gguf_name, hf_name in _TOP_LEVEL.items():
        out[gguf_name] = hf_name
    # Top-level nextn.* (no blk.N. prefix) — GGUF stores these flat in the file.
    for suffix, hf_name in _NEXTN_SUFFIXES.items():
        out[suffix] = hf_name
    for layer in range(num_hidden_layers):
        for suffix, hf_suffix in _BLOCK_SUFFIXES.items():
            gguf_name = f"blk.{layer}.{suffix}"
            out[gguf_name] = f"model.layers.{layer}.{hf_suffix}"
        out[f"blk.{layer}.ssm_a"] = f"model.layers.{layer}.linear_attn.A_log"
        # Fused expert tensors (``blk.L.ffn_gate_exps.weight`` etc.)
        # are yielded as a single ``[num_experts, ...]`` tensor; the
        # SGLang ``FusedMoE`` loader splits them along dim 0 itself.
        for suffix, hf_suffix in _FUSED_EXPERT_TARGETS.items():
            out[f"blk.{layer}.{suffix}"] = f"model.layers.{layer}.{hf_suffix}"
    # SGLang ``FusedMoE`` exposes ``w13_qweight`` / ``w2_qweight`` as
    # ``GGUFUninitializedParameter`` on the experts ``Module``; the
    # fork loader splits the fused GGUF buffer along dim 0 and uses
    # the ``load_fused_expert_weights`` helper to walk the per-expert
    # slice, so the iterator yields one entry per (layer, expert)
    # using the per-expert HF names.
    # Per-expert HF names are also yielded so the SGLang
    # ``Qwen3_5MoeForConditionalGeneration.load_weights`` hook
    # (or any other SGLang path) can address each expert
    # individually. The names here match the per-expert mapping
    # already produced above for ``mlp.experts.{N}.gate_proj.weight``
    # style tensors.
    for layer in range(num_hidden_layers):
        for exp in range(num_experts):
            out[f"blk.{layer}.ffn_gate_exps.{exp}.weight"] = (
                f"model.layers.{layer}.mlp.experts.{exp}.gate_proj.weight"
            )
            out[f"blk.{layer}.ffn_up_exps.{exp}.weight"] = (
                f"model.layers.{layer}.mlp.experts.{exp}.up_proj.weight"
            )
            out[f"blk.{layer}.ffn_down_exps.{exp}.weight"] = (
                f"model.layers.{layer}.mlp.experts.{exp}.down_proj.weight"
            )
    nextn_layer = num_hidden_layers
    for suffix, hf_name in _NEXTN_SUFFIXES.items():
        out[f"blk.{nextn_layer}.{suffix}"] = hf_name
    return out


def norm_sub_one(w, hf_name: str):
    if "linear_attn.norm" in hf_name or "ssm_norm" in hf_name:
        return w.clone()
    if "norm" not in hf_name and "layernorm" not in hf_name:
        return w.clone()
    return w - 1.0


def alog_transform(w):
    import torch

    return torch.log(-w)


def conv1d_reorder(
    w,
    num_v_per_k: int = 1,
    k_heads: int = 1,
    head_dim: int = 1,
):
    # Qwen3.5 GatedDeltaNet ``ssm_conv1d`` is stored by llama.cpp as
    # ``(out=8192, kernel=4)`` (the gguf-py reader already reverses the
    # axis order, so what comes through ``tensor.data`` is the (out, kernel)
    # layout). SGLang's Qwen3_5GatedDeltaNet treats this buffer as the
    # concat of two q/k convs plus the v conv (mirroring the mamba-v2
    # in_proj_qkv sharding it inherits), then ``unsqueeze(1)``'s in the
    # constructor to add the in-channel axis. So the data is already in
    # the SGLang-expected ``(out, kernel)`` shape and needs no further
    # rearrangement.
    if w.ndim != 2:
        raise ValueError(f"conv1d GGUF tensor must be 2D, got {tuple(w.shape)}")
    return w.contiguous()


def v_head_tiled_to_grouped(w, v_per_k: int, k_heads: int, head_dim: int):
    return w.reshape(v_per_k, k_heads, head_dim).permute(1, 0, 2).flatten()


def whole_row_q6k_permute(w, perm, block_bytes: int = 160):
    del block_bytes
    return w.index_select(0, perm)


def get_out_proj_activation_perm(num_v_per_k: int, k_heads: int, head_dim: int):
    import torch

    return (
        torch.arange(num_v_per_k * k_heads * head_dim)
        .reshape(num_v_per_k, k_heads, head_dim)
        .permute(1, 0, 2)
        .flatten()
    )


def qwen35moe_linear_attn_vcfg(
    *,
    linear_num_key_heads: int,
    linear_num_value_heads: int,
    linear_key_head_dim: int,
    linear_value_head_dim: int,
) -> dict:
    k_heads = int(linear_num_key_heads)
    v_heads = int(linear_num_value_heads)
    return {
        "k_heads": k_heads,
        "num_v_per_k": v_heads // k_heads,
        "num_value_heads": v_heads,
        "head_k_dim": int(linear_key_head_dim),
        "head_v_dim": int(linear_value_head_dim),
    }


_ON_THE_FLY_DEQUANT_SUFFIXES = (
    ".in_proj_qkv.weight",
    ".in_proj_z.weight",
    ".in_proj_a.weight",
    ".in_proj_b.weight",
    ".embed_tokens.weight",
    ".lm_head.weight",
)


def qwen35moe_gguf_on_the_fly_needs_hf_transform(hf_name: str, cfg: dict) -> bool:
    if any(hf_name.endswith(s) for s in (".embed_tokens.weight", ".lm_head.weight")):
        return True
    if not _needs_v_head_reorder(hf_name, cfg):
        return False
    return any(hf_name.endswith(s) for s in _ON_THE_FLY_DEQUANT_SUFFIXES)


def _qwen35_hidden_size(cfg: dict) -> int:
    return int(cfg.get("hidden_size", 2048))


def _qwen35_dequant_to_out_in_rows(w, hf_name: str, cfg: dict):
    """GGUF dequant may be (out,in) or (hidden,lead); qkv uses lead on dim1 for on-the-fly reorder."""
    import torch

    key_dim, value_dim, _ = _linear_attn_dims(cfg)
    lead = key_dim * 2 + value_dim
    hidden = _qwen35_hidden_size(cfg)
    if w.ndim != 2:
        return w
    if hf_name.endswith(".in_proj_qkv.weight"):
        if w.shape[0] != lead and w.shape[1] == lead:
            return w.t().contiguous()
        return w
    if hf_name.endswith(
        (
            ".in_proj_z.weight",
            ".in_proj_a.weight",
            ".in_proj_b.weight",
        )
    ):
        if w.shape[0] == hidden:
            return w
        if w.shape[1] == hidden:
            return w.t().contiguous()
    if hf_name.endswith((".embed_tokens.weight", ".lm_head.weight")):
        if w.shape[0] == hidden:
            return w.t().contiguous()
    return w


def _qwen35_on_the_fly_linear_attn_qweight(w, hf_name: str, cfg: dict):
    """GGUF dequant is (out_features, in_features); loader expects same layout as qweight rows."""
    import torch

    w = _qwen35_dequant_to_out_in_rows(w, hf_name, cfg)
    key_dim, value_dim, _ = _linear_attn_dims(cfg)
    vpk, kh, hvd = int(cfg["num_v_per_k"]), int(cfg["k_heads"]), int(cfg["head_v_dim"])
    lead = key_dim * 2 + value_dim

    if hf_name.endswith(".in_proj_qkv.weight"):
        if w.ndim == 2 and w.shape[0] == lead:
            tail = w.shape[1]
            q = w[:key_dim]
            k = w[key_dim : 2 * key_dim]
            v = w[2 * key_dim : 2 * key_dim + value_dim]
            v_g = (
                v.reshape(vpk, kh, hvd, tail)
                .permute(1, 0, 2, 3)
                .reshape(value_dim, tail)
            )
            return torch.cat([q, k, v_g], dim=0).contiguous()
        if w.ndim == 2 and w.shape[1] == lead:
            return _v_reorder_dim1_segment(w, key_dim, value_dim, vpk, kh, hvd)
        return w
    if hf_name.endswith(
        (
            ".in_proj_z.weight",
            ".in_proj_a.weight",
            ".in_proj_b.weight",
        )
    ):
        return w
    return w


def qwen35_gguf_dequant_from_reader_tensor(tensor) -> "torch.Tensor":
    """Same layout as kt ``gguf_gpu_slice_to_hf_awq_prep._tensor_to_torch`` before HF transforms."""
    import gguf
    import numpy as np
    import torch

    raw = np.asarray(tensor.data, dtype=np.uint8)
    w = torch.from_numpy(gguf.dequantize(raw, tensor.tensor_type)).float()
    logical = tuple(int(x) for x in tensor.shape)
    if w.ndim == 2 and w.shape != logical and w.shape == logical[::-1]:
        return w.contiguous()
    if w.numel() == int(np.prod(logical)):
        return w.reshape(logical).contiguous()
    return w


def qwen35_gguf_dense_for_gguf_linear(w, hf_name: str, cfg: dict):
    """On-the-fly GGUF linear: ``fused_mul_mat_gguf`` expects qweight rows ``(out_features, in_features)``."""
    import torch

    dense = w.float() if isinstance(w, torch.Tensor) else torch.from_numpy(w).float()
    return apply_gguf_to_hf_weight(dense, hf_name, cfg)


def qwen35_gguf_dequant_apply_for_load(w, hf_name: str, cfg: dict):
    import torch

    dense = w.float() if isinstance(w, torch.Tensor) else torch.from_numpy(w).float()
    hidden = _qwen35_hidden_size(cfg)
    key_dim, value_dim, _ = _linear_attn_dims(cfg)

    def _reshape_dequant_rows_to_hidden_major(rows, out_features: int):
        if rows.ndim == 2 and rows.shape[0] == out_features and rows.shape[1] == hidden:
            return rows.t().contiguous()
        return rows

    if hf_name.endswith(".in_proj_z.weight"):
        dense = _reshape_dequant_rows_to_hidden_major(dense, value_dim)
        return apply_gguf_to_hf_weight(dense, hf_name, cfg)
    if hf_name.endswith((".in_proj_a.weight", ".in_proj_b.weight")):
        nvh = int(cfg["num_value_heads"])
        dense = _reshape_dequant_rows_to_hidden_major(dense, nvh)
        return apply_gguf_to_hf_weight(dense, hf_name, cfg)
    if hf_name.endswith(".in_proj_qkv.weight"):
        key_dim, value_dim, _ = _linear_attn_dims(cfg)
        lead = key_dim * 2 + value_dim
        hidden = _qwen35_hidden_size(cfg)
        if dense.ndim == 2 and dense.shape[0] == lead and dense.shape[1] == hidden:
            dense = dense.t().contiguous()
        return apply_gguf_to_hf_weight(dense, hf_name, cfg)
    return _qwen35_on_the_fly_linear_attn_qweight(dense, hf_name, cfg)


def qwen35_col_linear_weight_to_loader_layout(hf_name: str, w):
    if not hf_name.endswith((".mlp.gate.weight", ".mlp.shared_expert.gate.weight")):
        return w
    if w.ndim != 2:
        return w
    out, inp = int(w.shape[0]), int(w.shape[1])
    if out > inp:
        return w.t().contiguous()
    return w


def _is_rmsnorm_like(hf_name: str) -> bool:
    if "linear_attn.norm" in hf_name or "ssm_norm" in hf_name:
        return False
    if ".norm." in hf_name or hf_name.endswith("_norm.weight") or "layernorm" in hf_name:
        return True
    return False


def apply_f32_transforms(w, gguf_name: str):
    hf_name = qwen35moe_gguf_to_hf(gguf_name)
    if hf_name is None:
        return w
    return apply_f32_transforms_hf(w, hf_name)


def apply_f32_transforms_hf(w, hf_name: str):
    if _is_rmsnorm_like(hf_name):
        return w - 1.0
    if hf_name.endswith(".A_log"):
        return alog_transform(w)
    if hf_name.endswith(".conv1d.weight"):
        return conv1d_reorder(w)
    return w


def _linear_attn_dims(cfg: dict) -> tuple[int, int, int]:
    k_heads = int(cfg["k_heads"])
    v_per_k = int(cfg["num_v_per_k"])
    head_v = int(cfg["head_v_dim"])
    head_k = int(cfg.get("head_k_dim", head_v))
    key_dim = k_heads * head_k
    value_dim = v_per_k * k_heads * head_v
    conv_dim = 2 * key_dim + value_dim
    return key_dim, value_dim, conv_dim


def _needs_v_head_reorder(hf_name: str, cfg: dict) -> bool:
    if "linear_attn" not in hf_name:
        return False
    vpk = int(cfg["num_v_per_k"])
    if vpk <= 1:
        return False
    v_heads = int(cfg.get("num_value_heads", int(cfg["k_heads"]) * vpk))
    return v_heads > int(cfg["k_heads"])


def _v_reorder_dim0_segment(w, key_dim: int, value_dim: int, v_per_k: int, k_heads: int, head_dim: int):
    import torch

    if w.shape[0] != key_dim * 2 + value_dim:
        return w
    q = w[:key_dim]
    k = w[key_dim : 2 * key_dim]
    v = w[2 * key_dim : 2 * key_dim + value_dim]
    v_g = v_head_tiled_to_grouped(
        v.reshape(v_per_k, k_heads, head_dim), v_per_k, k_heads, head_dim
    )
    return torch.cat([q, k, v_g], dim=0)


def _v_reorder_dim1_segment(w, key_dim: int, value_dim: int, v_per_k: int, k_heads: int, head_dim: int):
    import torch

    if w.ndim != 2 or w.shape[1] != key_dim * 2 + value_dim:
        return w
    q = w[:, :key_dim]
    k = w[:, key_dim : 2 * key_dim]
    v = w[:, 2 * key_dim : 2 * key_dim + value_dim]
    v_t = v.reshape(w.shape[0], v_per_k, k_heads, head_dim).permute(0, 2, 1, 3)
    v_g = v_t.reshape(w.shape[0], value_dim)
    return torch.cat([q, k, v_g], dim=1)


def apply_gguf_to_hf_weight(w, hf_name: str, cfg: dict):
    import torch

    w = apply_f32_transforms_hf(w, hf_name)
    if not _needs_v_head_reorder(hf_name, cfg):
        if hf_name.endswith(".conv1d.weight") and w.ndim == 2:
            return w.unsqueeze(1).contiguous()
        return qwen35_col_linear_weight_to_loader_layout(hf_name, w)

    key_dim, value_dim, conv_dim = _linear_attn_dims(cfg)
    vpk, kh, hvd = int(cfg["num_v_per_k"]), int(cfg["k_heads"]), int(cfg["head_v_dim"])

    if hf_name.endswith(".in_proj_qkv.weight"):
        lead = key_dim * 2 + value_dim
        if w.ndim == 1 and w.numel() == lead:
            return _v_reorder_dim0_segment(w, key_dim, value_dim, vpk, kh, hvd)
        if w.ndim == 2 and w.shape[0] == lead:
            return _v_reorder_dim0_segment(w, key_dim, value_dim, vpk, kh, hvd)
        if w.ndim == 2 and w.shape[1] == lead:
            return _v_reorder_dim1_segment(w, key_dim, value_dim, vpk, kh, hvd)
    if hf_name.endswith(".in_proj_z.weight") and w.ndim >= 1:
        if w.ndim == 2 and w.shape[0] == value_dim:
            return w.reshape(vpk, kh, hvd, w.shape[1]).permute(1, 0, 2, 3).reshape(w.shape)
        if w.ndim == 2 and w.shape[1] == value_dim:
            return w.reshape(w.shape[0], vpk, kh, hvd).permute(0, 2, 1, 3).reshape(w.shape)
        flat = w.reshape(-1)
        if flat.numel() == value_dim:
            return v_head_tiled_to_grouped(
                flat.reshape(vpk, kh, hvd), vpk, kh, hvd
            ).reshape(w.shape)
    if hf_name.endswith((".in_proj_a.weight", ".in_proj_b.weight")):
        lead = kh * vpk
        if w.ndim == 2 and w.shape[0] == lead:
            return w.reshape(vpk, kh, 1, w.shape[1]).permute(1, 0, 2, 3).reshape(w.shape)
        if w.ndim == 2 and w.shape[1] == lead:
            return w.reshape(w.shape[0], vpk, kh, 1).permute(0, 2, 1, 3).reshape(w.shape)
        flat = w.reshape(-1)
        if flat.numel() == lead:
            return (
                v_head_tiled_to_grouped(flat.reshape(vpk, kh, 1), vpk, kh, 1)
                .reshape(w.shape)
            )
    if hf_name.endswith(".conv1d.weight") and w.ndim == 2 and w.shape[0] == conv_dim:
        qk = w[: 2 * key_dim]
        v = w[2 * key_dim :]
        kernel = v.shape[1]
        v_g = (
            v.reshape(vpk, kh, hvd, kernel)
            .permute(1, 0, 2, 3)
            .reshape(value_dim, kernel)
        )
        w2 = torch.cat([qk, v_g], dim=0)
        return w2.unsqueeze(1).contiguous()
    if hf_name.endswith(".out_proj.weight") and w.ndim == 2:
        in_ch = w.shape[1]
        if in_ch == value_dim:
            perm = get_out_proj_activation_perm(vpk, kh, hvd)
            return w[:, perm].contiguous()
    if hf_name.endswith((".A_log", ".dt_bias")):
        flat = w.reshape(-1)
        if flat.numel() == kh * vpk:
            return (
                v_head_tiled_to_grouped(flat.reshape(vpk, kh, 1), vpk, kh, 1)
                .reshape(w.shape)
            )
    if hf_name.endswith(".conv1d.weight") and w.ndim == 2:
        return w.unsqueeze(1).contiguous()
    return qwen35_col_linear_weight_to_loader_layout(hf_name, w)
