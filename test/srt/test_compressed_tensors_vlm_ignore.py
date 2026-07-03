"""Regression: VLM runtime prefixes must match HF ignore entries after WeightsMapper."""

from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
    should_ignore_layer,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper


def test_vlm_runtime_prefix_matches_mapped_ignore_entry():
    runtime = "model.language_model.layers.0.linear_attn.in_proj_a"
    hf_ignore = ["model.layers.0.linear_attn.in_proj_a"]
    mapped = Qwen3VLForConditionalGeneration.hf_to_sglang_mapper.apply_list(
        hf_ignore
    )
    assert mapped == ["language_model.model.layers.0.linear_attn.in_proj_a"]
    assert check_equal_or_regex_match(runtime, mapped)


def test_should_ignore_in_proj_b_after_mapper():
    runtime = "model.language_model.layers.7.linear_attn.in_proj_b"
    ignore = Qwen3VLForConditionalGeneration.hf_to_sglang_mapper.apply_list(
        ["model.layers.7.linear_attn.in_proj_b"]
    )
    assert should_ignore_layer(runtime, ignore=ignore)


def test_hf_style_ignore_without_mapper_still_matches_runtime():
    runtime = "model.language_model.layers.2.linear_attn.in_proj_a"
    assert should_ignore_layer(
        runtime,
        ignore=["model.language_model.layers.2.linear_attn.in_proj_a"],
    )