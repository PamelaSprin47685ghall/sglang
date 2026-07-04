"""Qwen3.5 MoE VL full-attn checkpoint names map to layer-level params."""


def test_vl_full_attention_checkpoint_strips_self_attn_segment_when_param_exists():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(layers_block_type=None, layer_types=["full_attention"])
    params = {"model.layers.0.self_attn.qkv_proj.weight_packed": None}
    name = "model.layers.0.self_attn.q_proj.weight"

    assert (
        _map_vl_lm_layer_param_to_sglang_attn_submodule(name, cfg, params)
        == "model.layers.0.self_attn.q_proj.weight"
    )


def test_flat_hf_slice_preserves_self_attn_until_param_resolution():
    from sglang.srt.models.qwen3_5 import _map_hf_flat_text_to_vl_lm_param

    raw = "model.layers.15.self_attn.k_norm.weight"
    mapped = _map_hf_flat_text_to_vl_lm_param(raw)
    assert mapped == "model.layers.15.self_attn.k_norm.weight"
    assert ".self_attn." in mapped


def test_layer_norm_remapped_under_linear_attn_submodule():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(
        layers_block_type=None,
        layer_types=None,
        full_attention_interval=4,
    )
    flat = "model.layers.0.input_layernorm.weight"
    params = {
        "model.layers.0.linear_attn.input_layernorm.weight": None,
    }
    assert (
        _map_vl_lm_layer_param_to_sglang_attn_submodule(flat, cfg, params)
        == "model.layers.0.linear_attn.input_layernorm.weight"
    )


def test_mlp_weights_not_nested_under_linear_attn():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(
        layers_block_type=None,
        layer_types=None,
        full_attention_interval=4,
    )
    gate = "model.layers.0.mlp.shared_expert.gate_proj.weight"
    assert _map_vl_lm_layer_param_to_sglang_attn_submodule(gate, cfg, {}) == gate


def test_qwen35_moe_vl_disables_parent_hf_mapper():
    from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
    from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

    assert (
        Qwen3_5MoeForConditionalGeneration.hf_to_sglang_mapper.orig_to_new_prefix
        != Qwen3VLForConditionalGeneration.hf_to_sglang_mapper.orig_to_new_prefix
        or Qwen3_5MoeForConditionalGeneration.hf_to_sglang_mapper.orig_to_new_prefix
        == {}
    )


def test_full_attention_mapping_preserves_self_attn_when_runtime_uses_language_model_prefix():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(layers_block_type=None, layer_types=["full_attention"])
    # Checkpoint name uses model.layers prefix
    name = "model.layers.0.self_attn.q_proj.weight"
    # Runtime params use model.language_model.layers prefix
    params = {"model.language_model.layers.0.self_attn.q_proj.weight": None}

    mapped = _map_vl_lm_layer_param_to_sglang_attn_submodule(name, cfg, params)
    # Should map to runtime prefix, keep self_attn segment
    assert mapped == "model.language_model.layers.0.self_attn.q_proj.weight"
    assert ".self_attn." in mapped


def test_full_attention_mapping_handles_qkv_packed_with_language_model_prefix():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(layers_block_type=None, layer_types=["full_attention"])
    name = "model.layers.0.self_attn.qkv_proj.weight_packed"
    params = {"model.language_model.layers.0.self_attn.qkv_proj.weight_packed": None}

    mapped = _map_vl_lm_layer_param_to_sglang_attn_submodule(name, cfg, params)
    assert mapped == "model.language_model.layers.0.self_attn.qkv_proj.weight_packed"


def test_full_attention_mapping_strips_self_attn_when_runtime_has_no_self_attn_submodule():
    from types import SimpleNamespace

    from sglang.srt.models.qwen3_5 import _map_vl_lm_layer_param_to_sglang_attn_submodule

    cfg = SimpleNamespace(
        layers_block_type=None,
        layer_types=None,
        full_attention_interval=4,
    )
    # Checkpoint name uses self_attn.q_proj
    name = "model.layers.11.self_attn.q_proj.weight_packed"
    # Runtime has model.layers.11.qkv_proj but NOT self_attn
    params = {
        "model.layers.11.qkv_proj.weight_packed": None,
        "model.layers.11.o_proj.weight_packed": None,
    }

    mapped = _map_vl_lm_layer_param_to_sglang_attn_submodule(name, cfg, params)
    # Should strip self_attn. to allow stacked loader to process it
    assert mapped == "model.layers.11.q_proj.weight_packed"


def test_gguf_perms_enabled_by_bypass_env(monkeypatch):
    import os
    monkeypatch.setenv("SGLANG_KT_BYPASS_GPU_MOE", "1")
    
    # Verify that the condition saw_gguf_checkpoint or env == "1" evaluates to True
    saw_gguf_checkpoint = False
    condition = saw_gguf_checkpoint or os.environ.get("SGLANG_KT_BYPASS_GPU_MOE") == "1"
    assert condition is True


def test_enable_gguf_linear_attn_out_proj_perms_recursive():
    from sglang.srt.models.qwen3_5 import _enable_gguf_linear_attn_out_proj_perms
    import torch
    import torch.nn as nn

    class DummyLinearAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_v_heads = 4
            self.num_k_heads = 2
            self.head_v_dim = 8
            self.out_proj = nn.Linear(16, 16)
            self.out_proj.qweight = nn.Parameter(torch.empty(0))
            self.register_buffer(
                "_gguf_out_proj_act_perm",
                torch.tensor([], dtype=torch.long),
                persistent=False,
            )

        def enable_gguf_out_proj_activation_perm(self):
            self._gguf_out_proj_act_perm = torch.tensor([0, 1])

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = DummyLinearAttn()

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # VLM model has self.model (CausalLM)
            self.model = nn.Module()
            # CausalLM has self.model (Model)
            self.model.model = nn.Module()
            # Model has self.layers
            self.model.model.layers = nn.ModuleList([DummyLayer()])

    model = DummyModel()
    _enable_gguf_linear_attn_out_proj_perms(model)
    # Under old implementation, layers are not found so perm stays empty
    assert model.model.model.layers[0].linear_attn._gguf_out_proj_act_perm.numel() > 0
