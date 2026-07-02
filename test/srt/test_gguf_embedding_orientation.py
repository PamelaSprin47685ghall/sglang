"""GGUF token_embd is stored [hidden, vocab]; embedding lookup must use vocab dim."""
import torch

from sglang.srt.layers.quantization.gguf import apply_gguf_embedding
from gguf import GGMLQuantizationType as WeightType


def test_unquantized_embedding_uses_transposed_table():
    hidden, vocab = 4, 6
    # GGUF layout: rows=hidden, cols=vocab
    table = torch.arange(hidden * vocab, dtype=torch.float32).reshape(hidden, vocab)
    ids = torch.tensor([0, 2, 5], dtype=torch.long)
    expected = torch.stack([table[:, i] for i in ids.tolist()])
    out = apply_gguf_embedding(
        ids, table, int(WeightType.F32), hidden_size=hidden, dtype=torch.float32
    )
    assert out.shape == (3, hidden)
    assert torch.equal(out, expected)


def test_gguf_layout_vocab_axis_is_dim1_when_rows_are_hidden():
    hidden = 8
    vocab = 4
    table = torch.randn(hidden, vocab)
    assert table.shape[0] == hidden
    ids = torch.tensor([1], dtype=torch.long)
    out = apply_gguf_embedding(
        ids, table, int(WeightType.F32), hidden_size=hidden, dtype=torch.float32
    )
    assert torch.allclose(out[0], table[:, 1])