from sglang.srt.layers.quantization.marlin_utils import (
    marlin_pad_n_for_repack,
    marlin_slice_n_output,
)


def test_marlin_pad_n_gdn_in_proj():
    assert marlin_pad_n_for_repack(32) == 64
    assert marlin_pad_n_for_repack(2048) == 2048


def test_marlin_slice_n_output():
    import torch

    x = torch.randn(2, 64)
    y = marlin_slice_n_output(x, 32)
    assert y.shape == (2, 32)