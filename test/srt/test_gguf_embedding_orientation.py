"""GGUF embedding: quant rows are vocab-indexed; F16 logical layout may be [hidden, vocab]."""
import torch

from sglang.srt.layers.quantization.gguf import apply_gguf_embedding
from gguf import GGMLQuantizationType as WeightType


def test_unquantized_transposes_hidden_major_table():
    hidden, vocab = 4, 6
    table = torch.arange(hidden * vocab, dtype=torch.float32).reshape(hidden, vocab)
    ids = torch.tensor([0, 2, 5], dtype=torch.long)
    expected = torch.stack([table[:, i] for i in ids.tolist()])
    out = apply_gguf_embedding(
        ids, table, int(WeightType.F32), hidden_size=hidden, dtype=torch.float32
    )
    assert out.shape == (3, hidden)
    assert torch.equal(out, expected)


def test_unquantized_vocab_major_table_unchanged():
    hidden, vocab = 8, 4
    table = torch.randn(vocab, hidden)
    ids = torch.tensor([1], dtype=torch.long)
    out = apply_gguf_embedding(
        ids, table, int(WeightType.F32), hidden_size=hidden, dtype=torch.float32
    )
    assert torch.allclose(out[0], table[1])