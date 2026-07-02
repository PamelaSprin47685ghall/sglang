"""TDD: _map_shared_expert_gguf_checkpoint_name fuses shared_expert GGUF names.

GGUF stores shared_expert ``gate_proj``/``up_proj`` as separate tensors, but
the SGLang model (``Qwen2MoeMLP``) exposes a single fused ``gate_up_proj``.
The function ``_map_shared_expert_gguf_checkpoint_name`` post-processes the
GGUF-mapped checkpoint name so it matches ``params_dict``.
"""

from sglang.srt.model_loader.gguf_qwen35moe import (
    _map_shared_expert_gguf_checkpoint_name,
)

# ── gate_proj / up_proj unchanged at map stage (shard load in load_weights) ──


def test_gate_proj_weight_unchanged_at_map():
    n = "model.layers.3.mlp.shared_expert.gate_proj.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_up_proj_qweight_unchanged_at_map():
    n = "model.layers.3.mlp.shared_expert.up_proj.qweight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_gate_proj_qweight_type_unchanged_at_map():
    n = "model.layers.3.mlp.shared_expert.gate_proj.qweight_type"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


# ── gate.weight → shared_expert_gate.weight ───────────────────────────────


def test_gate_weight_to_shared_expert_gate():
    """shared_expert.gate.weight → shared_expert_gate.weight (routing gate)."""
    n = "model.layers.3.mlp.shared_expert.gate.weight"
    expected = "model.layers.3.mlp.shared_expert_gate.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == expected


def test_gate_bias_to_shared_expert_gate_bias():
    """shared_expert.gate.bias → shared_expert_gate.bias."""
    n = "model.layers.3.mlp.shared_expert.gate.bias"
    expected = "model.layers.3.mlp.shared_expert_gate.bias"
    assert _map_shared_expert_gguf_checkpoint_name(n) == expected


# ── down_proj unchanged ───────────────────────────────────────────────────


def test_down_proj_unchanged():
    """shared_expert.down_proj.weight must NOT be modified."""
    n = "model.layers.3.mlp.shared_expert.down_proj.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_down_proj_qweight_unchanged():
    """shared_expert.down_proj.qweight unchanged."""
    n = "model.layers.3.mlp.shared_expert.down_proj.qweight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


# ── non-shared_expert names unchanged ─────────────────────────────────────


def test_dense_mlp_gate_proj_unchanged():
    """Non-shared_expert gate_proj (dense MLP) must NOT be affected."""
    n = "model.layers.3.mlp.gate_proj.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_dense_mlp_up_proj_unchanged():
    n = "model.layers.3.mlp.up_proj.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_attention_qkv_unchanged():
    n = "model.layers.3.self_attn.q_proj.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_embed_tokens_unchanged():
    n = "model.embed_tokens.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_lm_head_unchanged():
    n = "lm_head.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_norm_unchanged():
    n = "model.norm.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


# ── already-fused / already-correct names unchanged ───────────────────────


def test_already_fused_gate_up_proj_unchanged():
    """If name already contains gate_up_proj (should not normally happen from
    GGUF mapper, but safety net), it must not be corrupted."""
    n = "model.layers.3.mlp.shared_expert.gate_up_proj.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_already_correct_shared_expert_gate_unchanged():
    """shared_expert_gate.weight (already correct) unchanged."""
    n = "model.layers.3.mlp.shared_expert_gate.weight"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_already_correct_shared_expert_gate_bias_unchanged():
    n = "model.layers.3.mlp.shared_expert_gate.bias"
    assert _map_shared_expert_gguf_checkpoint_name(n) == n


# ── all layers variant ────────────────────────────────────────────────────


def test_all_layers_gate_proj_unchanged():
    for layer in (0, 1, 10, 39, 41):
        n = f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight"
        assert _map_shared_expert_gguf_checkpoint_name(n) == n


def test_all_layers_gate_fused():
    for layer in (0, 1, 10, 39, 41):
        n = f"model.layers.{layer}.mlp.shared_expert.gate.weight"
        expected = f"model.layers.{layer}.mlp.shared_expert_gate.weight"
        assert _map_shared_expert_gguf_checkpoint_name(n) == expected
