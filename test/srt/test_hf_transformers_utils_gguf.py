import pytest
from transformers import GPT2Config

from sglang.srt.utils import hf_transformers_utils as hf_utils


def test_get_config_retries_without_gguf_file_for_unsupported_arch(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_APPLY_CONFIG_BACKUP", "none")
    gguf_file = tmp_path / "dummy.gguf"
    gguf_file.write_bytes(b"GGUF")

    calls = []

    def fake_from_pretrained(model, trust_remote_code, revision=None, **kwargs):
        calls.append((str(model), dict(kwargs)))
        if "gguf_file" in kwargs:
            raise ValueError("GGUF model with architecture qwen35moe is not supported yet.")
        return GPT2Config()

    monkeypatch.setattr(hf_utils, "_ensure_llama_flash_attention2_compat", lambda: None)
    monkeypatch.setattr(hf_utils.AutoConfig, "from_pretrained", fake_from_pretrained)

    config = hf_utils.get_config(
        str(gguf_file),
        trust_remote_code=True,
        revision=None,
        model_override_args={},
    )

    assert isinstance(config, GPT2Config)
    assert len(calls) == 2
    assert calls[0][0] == str(tmp_path)
    assert calls[0][1]["gguf_file"] == str(gguf_file)
    assert "gguf_file" not in calls[1][1]


def test_get_config_preserves_non_gguf_value_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_APPLY_CONFIG_BACKUP", "none")
    model_dir = tmp_path / "plain-model"
    model_dir.mkdir()

    def fake_from_pretrained(model, trust_remote_code, revision=None, **kwargs):
        raise ValueError("some other config error")

    monkeypatch.setattr(hf_utils, "_ensure_llama_flash_attention2_compat", lambda: None)
    monkeypatch.setattr(hf_utils.AutoConfig, "from_pretrained", fake_from_pretrained)

    with pytest.raises(ValueError, match="some other config error"):
        hf_utils.get_config(
            str(model_dir),
            trust_remote_code=True,
            revision=None,
            model_override_args={},
        )
