"""TDD: Qwen3.5 GGUF 名字映射与 loader hook 静态测试。"""
import pytest

from sglang.srt.model_loader.gguf_qwen35moe import (
    qwen35moe_gguf_to_hf,
    is_qwen35moe_expert,
    make_qwen35moe_gguf_map,
)

KNOWN_MAP = [
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output_norm.weight", "model.norm.weight"),
    ("output.weight", "model.lm_head.weight"),
    ("blk.0.attn_norm.weight", "model.layers.0.attn_norm.weight"),
    ("blk.0.ffn_norm.weight", "model.layers.0.ffn_norm.weight"),
    ("blk.0.attn_q.weight", "model.layers.0.attn.q_proj.weight"),
    ("blk.0.attn_k.weight", "model.layers.0.attn.k_proj.weight"),
    ("blk.0.attn_v.weight", "model.layers.0.attn.v_proj.weight"),
    ("blk.0.attn_output.weight", "model.layers.0.attn.output.weight"),
    ("blk.0.attn_q_norm.weight", "model.layers.0.attn.q_norm.weight"),
    ("blk.0.attn_k_norm.weight", "model.layers.0.attn.k_norm.weight"),
    ("blk.0.attn_qkv.weight", "model.layers.0.linear_attn.in_proj_qkv.weight"),
    ("blk.0.attn_gate.weight", "model.layers.0.linear_attn.in_proj_z.weight"),
    ("blk.0.ssm_alpha.weight", "model.layers.0.linear_attn.in_proj_a.weight"),
    ("blk.0.ssm_beta.weight", "model.layers.0.linear_attn.in_proj_b.weight"),
    ("blk.0.ssm_a", "model.layers.0.linear_attn.ssm_a.weight"),
    ("blk.0.ssm_a.weight", "model.layers.0.linear_attn.ssm_a.weight"),
    ("blk.0.ssm_dt.bias", "model.layers.0.linear_attn.ssm_dt.bias"),
    ("blk.0.ssm_conv1d.weight", "model.layers.0.linear_attn.conv1d.weight"),
    ("blk.0.ssm_norm.weight", "model.layers.0.linear_attn.norm.weight"),
    ("blk.0.linear_attn_norm.weight", "model.layers.0.linear_attn.norm.weight"),
    ("blk.0.ssm_out.weight", "model.layers.0.linear_attn.out_proj.weight"),
    ("blk.0.ffn_gate_inp.weight", "model.layers.0.mlp.gate.weight"),
    ("blk.0.ffn_gate_exps.weight", "model.layers.0.mlp.experts.gate_proj.weight"),
    ("blk.0.ffn_up_exps.weight", "model.layers.0.mlp.experts.up_proj.weight"),
    ("blk.0.ffn_down_exps.weight", "model.layers.0.mlp.experts.down_proj.weight"),
    ("blk.40.nextn.eh_proj.weight", "model.mtp.eh_proj.weight"),
    ("blk.40.nextn.enorm.weight", "model.mtp.enorm.weight"),
    ("blk.40.nextn.hnorm.weight", "model.mtp.hnorm.weight"),
    ("blk.40.nextn.shared_head_norm.weight", "model.mtp.shared_head_norm.weight"),
    ("blk.0.ffn_gate_inp_shexp.weight", "model.layers.0.mlp.shared_expert.gate.weight"),
    ("blk.0.ffn_gate_shexp.weight", "model.layers.0.mlp.shared_expert.gate_proj.weight"),
    ("blk.0.ffn_down_shexp.weight", "model.layers.0.mlp.shared_expert.down_proj.weight"),
    ("blk.0.ffn_up_shexp.weight", "model.layers.0.mlp.shared_expert.up_proj.weight"),
    ("blk.39.attn_q.weight", "model.layers.39.attn.q_proj.weight"),
    ("blk.39.attn_qkv.weight", "model.layers.39.linear_attn.in_proj_qkv.weight"),
    ("nextn.eh_proj.weight", "model.mtp.eh_proj.weight"),
    ("nextn.eh_norm.weight", "model.mtp.eh_norm.weight"),
]

EXPERT_GGUF = [
    "blk.0.ffn_gate_exps.weight",
    "blk.0.ffn_down_exps.weight",
    "blk.0.ffn_up_exps.weight",
    "blk.0.ffn_gate_exps.weight",
    "blk.0.ffn_down_exps.weight",
    "blk.0.ffn_up_exps.weight",
    "blk.39.ffn_gate_exps.weight",
    "blk.0.ffn_gate_exps.0.weight",
    "blk.0.ffn_down_exps.0.weight",
    "blk.0.ffn_up_exps.0.weight",
]

BAD_GGUUFS = [
    "nonexistent.tensor.weight",
    "",
    "model.visual.patch_embed.weight",
    "blk.0.mtp.some_proj.weight",
]


def test_all_known_names_map_non_none():
    for gg, _ in KNOWN_MAP:
        result = qwen35moe_gguf_to_hf(gg)
        assert result is not None, f"Unmapped: {gg}"


def test_all_known_names_match_expected():
    for gg, expected in KNOWN_MAP:
        result = qwen35moe_gguf_to_hf(gg)
        assert result == expected, f"{gg} → {result} != {expected}"


def test_non_expert_names_do_not_expert():
    for gg, _ in KNOWN_MAP:
        if "ffn_" in gg and "_exps" in gg:
            continue
        assert not is_qwen35moe_expert(gg), f"False positive expert: {gg}"


def test_expert_names_detected():
    for gg in EXPERT_GGUF:
        assert is_qwen35moe_expert(gg), f"Missing expert detection: {gg}"


def test_unknown_names_return_none():
    for gg in BAD_GGUUFS:
        assert qwen35moe_gguf_to_hf(gg) is None, f"Should have been None: {gg}"


def test_no_language_model_prefix_in_output():
    for gg, _ in KNOWN_MAP:
        result = qwen35moe_gguf_to_hf(gg)
        if result is not None:
            assert "language_model" not in result, f"Leaked prefix: {gg} → {result}"


def test_make_qwen35moe_gguf_map_covers_non_expert_layers():
    m = make_qwen35moe_gguf_map(
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=1,
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_key_value_heads=2,
        num_attention_heads=8,
    )
    for gg, expected in KNOWN_MAP:
        if any(gg.startswith(f"blk.{layer}.") for layer in (0, 1)):
            assert m.get(gg) == expected, f"{gg} → {m.get(gg)} != {expected}"


def test_make_qwen35moe_gguf_map_experts_are_flat_per_expert():
    m = make_qwen35moe_gguf_map(
        num_hidden_layers=2,
        num_experts=3,
        num_experts_per_tok=1,
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_key_value_heads=2,
        num_attention_heads=8,
    )
    assert m["blk.0.ffn_gate_exps.0.weight"] == "model.layers.0.mlp.experts.0.gate_proj.weight"
    assert m["blk.0.ffn_gate_exps.2.weight"] == "model.layers.0.mlp.experts.2.gate_proj.weight"
    assert m["blk.1.ffn_down_exps.1.weight"] == "model.layers.1.mlp.experts.1.down_proj.weight"
    assert m["blk.0.ffn_gate_exps.weight"] == "model.layers.0.mlp.experts.gate_proj.weight"
    assert m["blk.0.ffn_up_exps.weight"] == "model.layers.0.mlp.experts.up_proj.weight"
    assert m["blk.0.ffn_down_exps.weight"] == "model.layers.0.mlp.experts.down_proj.weight"
    assert m["blk.2.nextn.eh_proj.weight"] == "model.mtp.eh_proj.weight"
    assert m["blk.2.nextn.enorm.weight"] == "model.mtp.enorm.weight"
    assert m["blk.2.nextn.hnorm.weight"] == "model.mtp.hnorm.weight"
    assert m["blk.2.nextn.shared_head_norm.weight"] == "model.mtp.shared_head_norm.weight"


# ---- loader hook 测试 (依赖 sglang) ----

sglang = pytest.importorskip("sglang", reason="sglang not installed")

from sglang.srt.model_loader.gguf_qwen35moe_hook import (  # noqa: E402
    install_gguf_qwen35moe,
    uninstall_gguf_qwen35moe,
)


def _fake_load_config():
    return type("LC", (), {"model_loader_extra_config": None})()


@pytest.fixture
def patched_loader():
    install_gguf_qwen35moe()
    yield
    uninstall_gguf_qwen35moe()


def test_loader_hook_installs_and_uninstalls(patched_loader):
    from sglang.srt.model_loader.loader import GGUFModelLoader

    assert GGUFModelLoader._get_gguf_weights_map.__name__ == "_patched"


def test_loader_hook_dispatches_non_qwen_to_original():
    from sglang.srt.model_loader.loader import GGUFModelLoader

    install_gguf_qwen35moe()
    try:
        cfg = type("C", (), {
            "model_type": "llama",
            "num_hidden_layers": 2,
            "text_config": None,
            "architectures": ["LlamaForCausalLM"],
        })()
        model_config = type("MC", (), {"hf_config": cfg})()
        model_config.model_path = "/tmp/non-qwen-gguf"
        with pytest.raises((RuntimeError, ValueError, AttributeError)):
            GGUFModelLoader(_fake_load_config())._get_gguf_weights_map(model_config)
    finally:
        uninstall_gguf_qwen35moe()


def test_loader_hook_dispatches_qwen35moe_to_custom():
    from sglang.srt.model_loader.loader import GGUFModelLoader

    install_gguf_qwen35moe()
    try:
        cfg = type("C", (), {
            "model_type": "qwen3_5_moe",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "vocab_size": 100,
            "intermediate_size": 128,
            "num_experts": 4,
            "num_experts_per_tok": 1,
            "num_key_value_heads": 2,
            "num_attention_heads": 8,
            "text_config": None,
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        })()
        model_config = type("MC", (), {"hf_config": cfg})()
        model_config.model_path = "/tmp/qwen35moe-gguf"
        m = GGUFModelLoader(_fake_load_config())._get_gguf_weights_map(model_config)
        assert m["blk.0.attn_q.weight"] == "model.layers.0.attn.q_proj.weight"
        assert m["blk.1.ffn_gate_exps.3.weight"] == "model.layers.1.mlp.experts.3.gate_proj.weight"
    finally:
        uninstall_gguf_qwen35moe()


def test_loader_hook_logs_activation(caplog):
    from sglang.srt.model_loader.loader import GGUFModelLoader

    install_gguf_qwen35moe()
    try:
        cfg = type("C", (), {
            "model_type": "qwen3_5_moe",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "vocab_size": 100,
            "intermediate_size": 128,
            "num_experts": 4,
            "num_experts_per_tok": 1,
            "num_key_value_heads": 2,
            "num_attention_heads": 8,
            "text_config": None,
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        })()
        model_config = type("MC", (), {"hf_config": cfg})()
        model_config.model_path = "/tmp/qwen35moe-gguf"
        with caplog.at_level("INFO"):
            GGUFModelLoader(_fake_load_config())._get_gguf_weights_map(model_config)
        assert "Activating qwen35moe GGUF name-map hook." in caplog.text
    finally:
        uninstall_gguf_qwen35moe()
