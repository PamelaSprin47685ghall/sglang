from types import SimpleNamespace

from sglang.srt.models.qwen3_5 import _qwen35_decoder_layer_type


def test_from_layer_types_full_attention_maps_to_attention():
    cfg = SimpleNamespace(
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4,
    )
    assert _qwen35_decoder_layer_type(cfg, 3) == "attention"
    assert _qwen35_decoder_layer_type(cfg, 0) == "linear_attention"


def test_from_full_attention_interval_when_no_layer_types():
    cfg = SimpleNamespace(full_attention_interval=4)
    assert _qwen35_decoder_layer_type(cfg, 3) == "attention"
    assert _qwen35_decoder_layer_type(cfg, 0) == "linear_attention"