# Qwen3-VL TP Runtime Prototype

这是一个面向单机研究的 runtime 原型目录，目标不是部署，而是把
Qwen3-VL 的 text decoder 执行路径拆开、看清、可控。

当前目录主要聚焦于：

- 构造真实 layer forward 所需的 bundle
- 复用 attention / mlp / decoder layer 的 forward 函数
- 支持连续多层 layer range forward
- 用 `gloo` 跑最小 TP 原型

目录说明：

- `hexgen_core/`: 参考 HexGen 的 runtime 骨架，承接并行分组、heterogeneous pipeline、transport、generation utils 等公共层
- `models/`: 模型相关实现，当前先放 Qwen3-VL text decoder 的 forward / trace 能力
- `core/config.py`: 默认路径和常量
- `core/dist.py`: 最小分布式初始化和 CPU 通信辅助
- `core/inputs.py`: 构造 Qwen3-VL 输入、加载模型和 processor
- `core/ops.py`: dtype、mask、RoPE、RMSNorm、eager attention 等基础算子
- `core/forward.py`: 运行时内核，主命名为 `forward_*` / `trace_*`
- `core/capture.py`: 从真实模型抓取 full layer / layer range bundle
- `core/stage.py`: stage 级别的统一入口，屏蔽 text stage / 后续 stage 类型差异
- `core/transport.py`: stage handoff 通信，支持多 tensor payload、按 tensor 保留 dtype，以及空通信占位
- `core/pipeline.py`: 多段 text pipeline 的分段解析、manifest 和 rank 运行入口
- `core/hybrid.py`: stage 内 TP + stage 间 PP 的最小混合并行骨架
- `core/`: 兼容层；当前保留已有导入路径，并逐步把实现转发到 `hexgen_core/` / `models/`
- `cli/full_layer.py`: 单层 decoder layer 的 prepare / tp 入口
- `cli/layer_range.py`: 多层 layer range 的 prepare / tp 入口
- `cli/text_hybrid.py`: text stage 的最小 PP+TP 混合原型
- `cli/text_stage.py`: 早期 text stage（含 DeepStack 注入）的 prepare / tp 入口
- `cli/two_stage_text.py`: 两段连续 text stage 的最小 pipeline 原型
- `cli/text_pipeline.py`: 多段连续 text stage 的通用 pipeline 原型

当前设计参考了两条思路：

- Jupiter：把 runtime 组织和模型前向解耦
- vLLM：把 TP primitive 拆成 column / row / qkv / merged-column 语义
- HexGen：把 runtime 内核和入口脚本分层组织

最小用法：

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/full_layer.py prepare \
  --layer-idx 11
```

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29571 WORLD_SIZE=2 RANK=0 \
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/full_layer.py tp \
  --bundle-path /mnt/ssd/code/Qwen3_vl/qwen3vl_full_layer_case.pt \
  --device cuda \
  --comm-dtype float32
```

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/layer_range.py prepare \
  --start-idx 11 \
  --end-idx 12
```

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29581 WORLD_SIZE=2 RANK=0 \
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/layer_range.py tp \
  --bundle-path /mnt/ssd/code/Qwen3_vl/qwen3vl_layer_range_case.pt \
  --device cuda \
  --comm-dtype float32
```

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/text_stage.py prepare \
  --start-idx 0 \
  --end-idx 2
```

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29591 WORLD_SIZE=2 RANK=0 \
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/text_stage.py tp \
  --bundle-path /mnt/ssd/code/Qwen3_vl/qwen3vl_text_stage_case.pt \
  --device cuda \
  --comm-dtype float32 \
  --trace-layers
```

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/two_stage_text.py prepare \
  --start-idx 0 \
  --split-idx 5 \
  --end-idx 11
```

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29601 WORLD_SIZE=2 RANK=0 \
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/two_stage_text.py pp \
  --stage0-bundle-path /mnt/ssd/code/Qwen3_vl/qwen3vl_text_stage0_case.pt \
  --stage1-bundle-path /mnt/ssd/code/Qwen3_vl/qwen3vl_text_stage1_case.pt \
  --device cuda \
  --comm-dtype float32
```

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/text_pipeline.py prepare \
  --ranges 0:5 6:11 12:17 \
  --bundle-dir /mnt/ssd/code/Qwen3_vl/qwen3vl_text_pipeline \
  --manifest-path /mnt/ssd/code/Qwen3_vl/qwen3vl_text_pipeline_manifest.pt
```

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29611 WORLD_SIZE=3 RANK=0 \
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/text_pipeline.py pp \
  --manifest-path /mnt/ssd/code/Qwen3_vl/qwen3vl_text_pipeline_manifest.pt \
  --device cuda \
  --comm-dtype float32
```

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/text_hybrid.py prepare \
  --ranges 0:5 6:11 \
  --tp 2 2 \
  --bundle-dir /mnt/ssd/code/Qwen3_vl/qwen3vl_text_hybrid \
  --manifest-path /mnt/ssd/code/Qwen3_vl/qwen3vl_text_hybrid_manifest.pt
```

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29621 WORLD_SIZE=4 RANK=0 \
/mnt/ssd/miniconda3/envs/vlm/bin/python \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/cli/text_hybrid.py run \
  --manifest-path /mnt/ssd/code/Qwen3_vl/qwen3vl_text_hybrid_manifest.pt \
  --device cuda \
  --comm-dtype float32
```

当前 hybrid / TP 原型默认采用：

- `tp_attn_math=orig`
- `tp_mlp_math=float32`

这是当前实验里 stage0 对齐效果最好的默认组合；如果需要做数值对照，可以在各个 `tp` / `run` 入口上显式覆盖 `--tp-attn-math` 和 `--tp-mlp-math`。
