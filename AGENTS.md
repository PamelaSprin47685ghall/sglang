# AGENTS — sglang Ornith (qwen35moe GGUF)

父索引：`../ktransformers/AGENTS.md`（工作区）或上游 ktransformers 仓 `AGENTS.md` §3。

## 本仓成就

- **GGUF 路由**：`srt/model_loader/loader.py` — `.gguf` auto → `GGUFModelLoader`
- **qwen35moe 映射/变换**：`srt/model_loader/gguf_qwen35moe.py` + `gguf_qwen35moe_hook.py`
- **VL 加载**：`srt/models/qwen3_5.py` — `_checkpoint_name_to_model_param`、out_proj GGUF perm、MTP 跳过
- **MoE**：跳过 expert GGUF yield（`weight_utils.py`）；kt CPU 专家并行
- **测试**：`python/test/srt/test_qwen35*.py`

```bash
cd python && pip install -e . && pytest test/srt/test_qwen35*.py -q
```

远程：`PamelaSprin47685ghall/sglang`