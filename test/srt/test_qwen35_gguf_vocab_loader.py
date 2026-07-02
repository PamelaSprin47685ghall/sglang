"""GGUF qweight loads via VocabParallelEmbedding.weight_loader."""
from unittest.mock import patch

import torch
from torch.nn.parameter import UninitializedParameter

from sglang.srt.layers.quantization.gguf import GGUFConfig, GGUFUninitializedParameter
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


def test_gguf_quant_config_registers_embed_qweight():
    with patch(
        "sglang.srt.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
        return_value=0,
    ), patch(
        "sglang.srt.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        emb = VocabParallelEmbedding(128, 64, quant_config=GGUFConfig())
    assert hasattr(emb, "qweight")
    assert hasattr(emb, "qweight_type")


def test_parallel_lm_head_has_tp_rank_after_init():
    with patch(
        "sglang.srt.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
        return_value=0,
    ), patch(
        "sglang.srt.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        head = ParallelLMHead(128, 64)
    assert head.tp_rank == 0
    assert head.tp_size == 1


def test_gguf_qweight_materializes_uninitialized_param():
    emb = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    emb.tp_size = 1
    emb.tp_rank = 0
    emb.org_vocab_size = 4
    emb.use_presharded_weights = False
    param = GGUFUninitializedParameter(requires_grad=False)
    param.output_dim = 0
    param.is_gguf_weight = True
    w = torch.randn(4, 8)
    emb.weight_loader(param, w)
    assert param.data.shape == (4, 8)