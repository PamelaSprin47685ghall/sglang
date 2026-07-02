from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional


_original_get_gguf_weights_map: Optional[Callable] = None
logger = logging.getLogger(__name__)


def _extract_qwen35moe_params(model_config: Any) -> Optional[Dict[str, Any]]:
    from sglang.srt.model_loader.loader import _get_gguf_num_hidden_layers

    cfg = getattr(model_config, "hf_config", model_config)
    text_cfg = getattr(cfg, "text_config", None) or cfg

    vocab_size = getattr(text_cfg, "vocab_size", None)
    hidden_size = getattr(text_cfg, "hidden_size", None)
    num_layers = _get_gguf_num_hidden_layers(cfg)
    num_experts = getattr(text_cfg, "num_experts", None) or getattr(
        cfg, "num_experts", None
    )
    num_experts_per_tok = getattr(text_cfg, "num_experts_per_tok", None) or getattr(
        cfg, "num_experts_per_tok", None
    )
    intermediate_size = getattr(text_cfg, "intermediate_size", None) or getattr(
        text_cfg, "moe_intermediate_size", None
    )
    num_key_value_heads = getattr(text_cfg, "num_key_value_heads", None)
    num_attention_heads = getattr(text_cfg, "num_attention_heads", None)
    num_expert_shared = getattr(text_cfg, "num_experts_per_tok", None)

    if hidden_size is None or num_layers is None:
        return None

    return {
        "num_hidden_layers": num_layers,
        "num_experts": num_experts or 1,
        "num_experts_per_tok": num_experts_per_tok or 1,
        "vocab_size": vocab_size or 1,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size or hidden_size * 4,
        "num_key_value_heads": num_key_value_heads or 1,
        "num_attention_heads": num_attention_heads or 1,
        "num_expert_shared": num_expert_shared or 1,
    }


def _is_qwen35moe_config(model_config: Any) -> bool:
    cfg = getattr(model_config, "hf_config", model_config)
    model_type = getattr(cfg, "model_type", "")
    architectures = getattr(cfg, "architectures", [])
    return (
        model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}
        or "Qwen3_5MoeForConditionalGeneration" in architectures
        or "Qwen3_5MoeForCausalLM" in architectures
    )


def install_gguf_qwen35moe() -> None:
    global _original_get_gguf_weights_map

    if _original_get_gguf_weights_map is not None:
        return

    from sglang.srt.model_loader.loader import GGUFModelLoader

    _original_get_gguf_weights_map = GGUFModelLoader._get_gguf_weights_map

    def _patched(self, model_config):
        if _is_qwen35moe_config(model_config):
            params = _extract_qwen35moe_params(model_config)
            if params is not None:
                logger.info("Activating qwen35moe GGUF name-map hook.")
                from sglang.srt.model_loader.gguf_qwen35moe import make_qwen35moe_gguf_map

                return make_qwen35moe_gguf_map(**params)
        return _original_get_gguf_weights_map(self, model_config)

    GGUFModelLoader._get_gguf_weights_map = _patched


def uninstall_gguf_qwen35moe() -> None:
    global _original_get_gguf_weights_map

    if _original_get_gguf_weights_map is None:
        return

    from sglang.srt.model_loader.loader import GGUFModelLoader

    GGUFModelLoader._get_gguf_weights_map = _original_get_gguf_weights_map
    _original_get_gguf_weights_map = None
