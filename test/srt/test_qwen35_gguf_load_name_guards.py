"""stacked_params_mapping string-rewrite guard must NOT corrupt shared_expert names.

``stacked_params_mapping`` joins ``gate_proj`` + ``up_proj`` → ``gate_up_proj``
(FusedMoE convention).  The existing guard ``"mlp.experts" in name`` protects
per-expert paths (e.g. ``mlp.experts.0.gate_proj.weight``) but *not*
``mlp.shared_expert.gate_proj.weight`` — ``shared_expert`` does not contain
the substring ``mlp.experts``.  Without an extended guard, the rewrite
produces ``mlp.shared_expert.gate_up_proj``, which does not exist in
``params_dict`` (shared_expert has independent ``gate_proj``/``down_proj``/
``up_proj`` attributes, NOT a fused ``gate_up_proj``).

This file unit-tests the guard logic in isolation.
"""


def _should_skip_stacked_rewrite(name: str) -> bool:
    """Return True when stacked_params_mapping rewrite must be skipped.

    Mirrors the guard condition used in
    ``Qwen3_5MoeForCausalLM.load_weights`` and
    ``Qwen3_5MoeForConditionalGeneration.load_weights``.
    """
    if "mlp.experts" in name:
        return True
    if "shared_expert" in name and "gate_up_proj" not in name:
        return True
    return False


_STACKED_PARAMS_MAPPING = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]


def _simulate_stacked_rewrite(name: str) -> str:
    """Apply stacked_params_mapping string-rewrite *with* the skip guard.

    Mirrors load_weights: first matching shard replaces once, then break
    (avoids k_proj matching inside qkv_proj, up_proj inside gate_up_proj).
    """
    for param_name, weight_name, _shard_id in _STACKED_PARAMS_MAPPING:
        if weight_name not in name:
            continue
        if _should_skip_stacked_rewrite(name):
            return name
        return name.replace(weight_name, param_name, 1)
    return name


# ── guard recognition tests ──────────────────────────────────────────────


def test_skip_mlp_experts_gate_proj():
    """Per-expert gate_proj is protected by existing mlp.experts guard."""
    n = "model.layers.3.mlp.experts.0.gate_proj.weight"
    assert _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == n


def test_skip_shared_expert_gate_proj():
    """shared_expert.gate_proj must NOT be rewritten to gate_up_proj."""
    n = "model.layers.3.mlp.shared_expert.gate_proj.weight"
    assert _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == n


def test_skip_shared_expert_down_proj():
    n = "model.layers.3.mlp.shared_expert.down_proj.weight"
    assert _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == n


def test_skip_shared_expert_up_proj():
    n = "model.layers.3.mlp.shared_expert.up_proj.weight"
    assert _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == n


# ── non-shared paths still rewrite (status quo) ──────────────────────────


def test_rewrite_dense_gate_proj():
    """Non-expert, non-shared gate_proj is still fused to gate_up_proj."""
    n = "model.layers.3.mlp.gate_proj.weight"
    assert not _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == "model.layers.3.mlp.gate_up_proj.weight"


def test_rewrite_dense_up_proj():
    n = "model.layers.3.mlp.up_proj.weight"
    assert not _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == "model.layers.3.mlp.gate_up_proj.weight"


def test_rewrite_q_proj():
    n = "model.layers.3.self_attn.q_proj.weight"
    assert not _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == "model.layers.3.self_attn.qkv_proj.weight"


# ── edge cases ────────────────────────────────────────────────────────────


def test_skip_mlp_experts_gate_proj_fused_qweight():
    """w13_qweight also contains mlp.experts → protection works."""
    n = "model.layers.3.mlp.experts.w13_qweight"
    assert _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == n


def test_shared_expert_gate_weight():
    """shared_expert.gate.weight (routing) must also be left alone."""
    n = "model.layers.3.mlp.shared_expert.gate.weight"
    assert _should_skip_stacked_rewrite(n)
    assert _simulate_stacked_rewrite(n) == n
