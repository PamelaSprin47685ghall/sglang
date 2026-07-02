from __future__ import annotations

import re
from typing import Optional


_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_TOP_LEVEL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "model.lm_head.weight",
    "nextn.eh_proj.weight": "model.mtp.eh_proj.weight",
    "nextn.eh_norm.weight": "model.mtp.eh_norm.weight",
}

_BLOCK_SUFFIXES = {
    "attn_norm.weight": "attn_norm.weight",
    "post_attention_norm.weight": "ffn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "attn_q.weight": "attn.q_proj.weight",
    "attn_k.weight": "attn.k_proj.weight",
    "attn_v.weight": "attn.v_proj.weight",
    "attn_output.weight": "attn.output.weight",
    "attn_q_norm.weight": "attn.q_norm.weight",
    "attn_k_norm.weight": "attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a.weight": "linear_attn.ssm_a.weight",
    "ssm_dt.bias": "linear_attn.ssm_dt.bias",
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


def is_qwen35moe_expert(gguf_name: str) -> bool:
    match = _BLOCK_RE.match(gguf_name)
    return bool(match and match.group(2) in _EXPERT_SUFFIXES)


def qwen35moe_gguf_to_hf(gguf_name: str) -> Optional[str]:
    if gguf_name in _TOP_LEVEL:
        return _TOP_LEVEL[gguf_name]

    match = _BLOCK_RE.match(gguf_name)
    if not match:
        return None

    layer, suffix = match.groups()
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
    for layer in range(num_hidden_layers):
        for suffix, hf_suffix in _BLOCK_SUFFIXES.items():
            gguf_name = f"blk.{layer}.{suffix}"
            out[gguf_name] = f"model.layers.{layer}.{hf_suffix}"
        for exp in range(num_experts):
            out[f"blk.{layer}.ffn_gate_exps.{exp}.weight"] = f"model.layers.{layer}.mlp.experts.{exp}.gate_proj.weight"
            out[f"blk.{layer}.ffn_up_exps.{exp}.weight"] = f"model.layers.{layer}.mlp.experts.{exp}.up_proj.weight"
            out[f"blk.{layer}.ffn_down_exps.{exp}.weight"] = f"model.layers.{layer}.mlp.experts.{exp}.down_proj.weight"
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
    if w.ndim != 2:
        raise ValueError(f"conv1d GGUF tensor must be 2D, got {tuple(w.shape)}")
    kernel_size, out_ch = w.shape
    qk_ch = k_heads * head_dim
    v_ch = num_v_per_k * k_heads * head_dim
    if out_ch != qk_ch + v_ch:
        raise ValueError(
            f"out_ch {out_ch} != qk_ch {qk_ch} + v_ch {v_ch}"
        )
    v_start = qk_ch
    v_end = qk_ch + v_ch
    reordered = w.clone()
    v = reordered[:, v_start:v_end]
    v = v.reshape(kernel_size, num_v_per_k, k_heads, head_dim)
    v = v.flip(1)
    reordered[:, v_start:v_end] = v.reshape(kernel_size, v_ch)
    return reordered.unsqueeze(1)


def v_head_tiled_to_grouped(w, v_per_k: int, k_heads: int, head_dim: int):
    return w.reshape(v_per_k, k_heads, head_dim).permute(1, 0, 2).flatten()


def whole_row_q6k_permute(w, perm, block_bytes: int = 160):
    del block_bytes
    return w.index_select(0, perm)


def get_out_proj_activation_perm(num_v_per_k: int, k_heads: int, head_dim: int):
    import torch

    grouped = (
        torch.arange(num_v_per_k * k_heads * head_dim)
        .reshape(num_v_per_k, k_heads, head_dim)
        .permute(1, 0, 2)
        .flatten()
    )
    return torch.argsort(grouped)


def _is_rmsnorm_like(hf_name: str) -> bool:
    if "linear_attn.norm" in hf_name or "ssm_norm" in hf_name:
        return False
    if ".norm." in hf_name or hf_name.endswith("_norm.weight"):
        return True
    return False


def apply_f32_transforms(w, gguf_name: str):
    hf_name = qwen35moe_gguf_to_hf(gguf_name)
    if hf_name is None:
        return w
    if _is_rmsnorm_like(hf_name):
        return w - 1.0
    if hf_name.endswith(".ssm_a.weight"):
        return alog_transform(w)
    if hf_name.endswith(".conv1d.weight"):
        return conv1d_reorder(w)
    return w
