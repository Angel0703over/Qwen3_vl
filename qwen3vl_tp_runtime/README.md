# qwen3vl_tp_runtime

这是面向 Qwen3-VL 的分布式推理运行时原型，主路径支持直接执行 `pp` / `tp` / `hybrid`。

## 主路径

- 主入口：`scripts/runtime.py`
- 推荐模式：`backend=pp|tp|hybrid`
- 主路径在启动时直接从 `model_path` 构建每个 stage/rank 的 `StageState`
- 推荐并行参数：`--pp N` 表示 pipeline stage 数，自动按模型层数平均切分；`--tp N` 表示统一 tensor parallel 度数。
- 高级覆盖参数：`--stage-ranges` 和 `--tp-degrees` 仍然保留，用于手工切层或异构 hybrid，例如 `--pp 2 --tp-degrees 2 1`。

常用写法：

```bash
# 纯 PP：2 个 stage，默认平均切层，例如 36 层会切成 0:17 / 18:35
--backend pp --pp 2

# 纯 TP：单 stage，全模型 TP=2
--backend tp --tp 2

# 均匀 PP+TP hybrid：2 个 PP stage，每个 stage 都是 TP=2
--backend hybrid --pp 2 --tp 2

# 异构 PP+TP hybrid：2 个 PP stage，stage0 TP=2，stage1 TP=1
--backend hybrid --pp 2 --tp-degrees 2 1
```

## 当前状态

这个 runtime 最初有两个明确目标：

- `TP` 不能再是“每张卡先加载完整权重，再在计算时按 rank 切分”，而要改成“每张卡只加载自己那份 shard”。
- `PP / TP / hybrid` 主运行路径不能依赖预先准备好的 `replay bundle` 或 manifest replay 产物，而要在启动时直接从 `model_path` 构建每个 stage/rank 的 `StageState`。

当前已经完成：

- 主 `pp / tp / hybrid` 路径已经是直接构建优先，会直接从 `model_path` 构建 `StageState`。
- `backend=tp` 已有独立入口 `hexgen_core/modules/tensor_parallel.py`：先在 TP 模块内校验单 stage TP layout，再复用共享的 `StageState` 执行引擎。
- TP 主路径公开名已收口为 `TensorParallelRunner`、`run_tensor_parallel_rank`、`run_stage_state_tp`；无引用的旧 text-specific TP wrapper 已删除。
- text `TP` 主路径不再让每张卡加载完整 decoder projection 权重再计算时切分。`tp_degree > 1` 的 direct stage 会先广播无权重 scaffold，然后每个 rank 从 `model_path` materialize 自己的本地 shard。
- `backend=tp` text generate 已通过真实 Jetson 冒烟验证：两个 rank 都是 `weight_load.tp_weight_sharded=true`，分别为 `tp_shard_rank=0/2` 和 `1/2`，projection 形状是 shard 后大小，`loaded_weight_tensor_bytes` 完全一致。
- `backend=hybrid` text generate 已通过真实 Jetson 冒烟验证：stage0 使用 rank-local TP shard，stage1 只加载自己的 PP stage 权重。
- `pp / hybrid` multimodal direct runtime 已在只依赖运行时构建的主路径通过真实 Jetson 冒烟验证。所有 rank 生成一致的 `generated_token_ids=[87140, 15946, 3837, 101177]`，summary 能证明 stage-local frontend/weight scope 和 hybrid TP shard-local materialization。
- multimodal startup transport 已经保持 stage-local 且足够薄：只携带 runtime shared metadata/tensor、本地 stage handoff、本地 stage visual 和 frame metadata；root/full/replay payload 会被拒绝。
- 当前里程碑可以认为已完成：从 `model_path` 直接启动 PP/TP/hybrid、rank-local text decoder shard，以及 `pp / hybrid` multimodal stage-only/shard-only 冒烟验证。runtime summary 已包含 PP stage 权重范围、TP projection shape proof，以及同一 TP stage 内权重字节数一致性证据。

仍然保留的尾部工作：

- embedding 和 `lm_head` 目前仍按当前执行语义复制。vocab/embedding parallelism 是后续优化，不属于当前已完成里程碑。
- 启动时间和峰值显存基线暂时不作为硬验收；当前验收重点是输出 token 和 `weight_load` shard 证据。
- schema、legacy compatibility、仅调试用 transport 仍可以继续清理，但 replay/capture 路径已经不再是主运行入口。

## 调试路径

下面这些路径只用于 replay、capture 和回归调试：

- `--manifest-path` replay run
- `--compare-direct`
- `--trace-layers`
- `--dump-layer`

这些路径需要显式传 `--allow-debug-paths`。
其中 `--compare-direct / --trace-layers / --dump-layer` 目前只支持 `backend=tp|hybrid` 的非 generate run。

## 后续路线

- 有序后续任务记录在 `ROADMAP.md`。
- 除非重新对齐目标，否则后续工作默认按 `ROADMAP.md` 的顺序推进。

## 基线

- 固定回归命令记录在 `BASELINE.md`。
- 修改 runtime 前后，默认使用这些 case id 作为 smoke/regression 集合。

## 目录结构

- `hexgen_core/`
  核心分布式运行时：process group、transport、schema，以及独立的 `pp` / `tp` / `hybrid` runner 模块。
- `models/qwen3vl/`
  Qwen3-VL 相关的模型适配代码。
- `models/qwen3vl/execution/`
  attention、decoder、stage forward/trace 等底层 tensor 执行逻辑。
- `models/qwen3vl/processing/`
  输入构造、processor/tokenizer 加载、model-path helper。
- `models/qwen3vl/weights/`
  权重 index、load plan、shard slicing，以及从权重 materialize `StageState` 的逻辑。
- `models/qwen3vl/runtime_builder.py`
  direct runtime builder：把 `model_path` 转换成 stage/rank `StageState` 和 manifest。
- `models/qwen3vl/runtime_text.py`
  text-only prompt metadata、runtime-only scaffold restore、启动 session helper。
- `models/qwen3vl/runtime_text_stage.py`
  text scaffold compaction、runtime input rebuild、本地 stage materialization helper。
- `scripts/`
  面向用户的 runtime 入口和辅助脚本。
- `scripts/helpers/`
  稳定 shell wrapper，例如 `run-runtime.sh`、`generate.sh`、`run-pp-mm-generate.sh`、`run-tp-mm-generate.sh` 和 `run-hybrid-mm-generate.sh`。
- `scripts/runtime_cli.py`
  runtime CLI 默认值、参数校验和 debug-path gating helper。
- `scripts/runtime_summary.py`
  runtime 输出 JSON summary 和 generated text 解码 helper。

## 命名约定

- 主路径中，从 `model_path` 构建出来的本地 stage/rank 执行对象统一叫 `StageState`。
- `replay bundle` / `bundle` 只保留给 debug、capture 和 manifest replay 产物。
- helper 名称优先短、明确，不把整条调用链都编码进函数名。
- 使用 `text_prompt_meta`，避免反复写 `runtime_only_text_generate_prompt_metadata`。
- 公开名称要描述清楚职责，但避免过长。
