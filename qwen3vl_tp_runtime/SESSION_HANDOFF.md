# qwen3vl_tp_runtime 对话迁移手册

> 目的：把当前对话里形成的项目状态、架构判断、命名约定、测试结果、运行命令、协作习惯和后续风险，尽可能完整地迁移到新的对话框。
>
> 建议新对话里的 Codex 先读本文件，再读 `README.md`、`ROADMAP.md`、`BASELINE.md`。如果时间很紧，至少读本文件的「新对话启动提示」和「当前架构状态」两节。

## 新对话启动提示

可以在新对话开始时直接贴下面这段：

```text
请先阅读 qwen3vl_tp_runtime/SESSION_HANDOFF.md。
这个项目是 Qwen3-VL 的 correctness-first 分布式推理 runtime 原型。
当前主线目标已经从旧 replay bundle 路径迁移到从 model_path 直接构建 StageState。
架构顺序必须保持：PP 和 TP 是并列基础后端，HYBRID 是依赖 PP/TP 的组合层。
主路径术语统一叫 StageState；bundle 只留给 replay/debug/capture。
最近一次重要重构：backend=tp 已有独立 TensorParallelManifest / build_direct_tp_manifest / TensorParallelRunner，不再借用 TextHybridManifest 或 TextHybridRunner；TP 的空壳 GenerateWorker / DecodeWorker 已删除；旧 TP debug/replay 入口已搬到顶层 qwen3vl_tp_runtime/debug/，hexgen_core/modules/ 只保留三种并行后端。
近期性能前置工作：HYBRID runtime-only broadcast 已收口到 `hybrid_runtime_inputs_v1` schema；纯 TP multimodal 已改成 rank0/input-owner startup contract；`baseline_runs/20260429-longterm-profile/` 已冻结 `tp-mm-generate`、`hybrid-text-generate`、`hybrid-mm-generate --pp 2 --tp-degrees 2 1` 的真实 Jetson profile；ROADMAP step 14 已完成 TP collective profiling 收口，TP collective event 已新增 `phase/layer_idx/module/reason` 和 device/CPU/gloo 子阶段 profiling；`baseline_runs/20260430-step14-substage-profile/` 已确认 pure TP 主要卡在 gloo；`baseline_runs/20260430-comm-dtype-profile/` 已完成 opt-in `--comm-dtype bfloat16/float16` 实验，两者 generated ids/text 不变并把 pure TP collective bytes 减半；`baseline_runs/20260430-runtime-input-local-profile/` 已完成 pure TP runtime input broadcast 减量，generate-time `stage_input_broadcast` 从每 rank `4` 个 event 降到 `0`；`baseline_runs/20260430-bfloat16-default-candidate/` 已完成 bfloat16 默认值候选回归；`baseline_runs/20260430-bfloat16-default/` 已完成 bfloat16 默认值落地回归，覆盖 `tp-text`、`tp-mm`、`hybrid-mm` 和 `tp-mm` 16-token 长 decode，不显式传 `--comm-dtype` 时 TP collective event 已显示 `comm_dtype=torch.bfloat16`，generated ids/text 均保持一致。
请继续用中文和我配合，优先读代码、直接修改、跑测试、同步到两台 Jetson。
```

## 当前日期和环境

- 当前日期：2026-04-30
- 当前工作目录：`/mnt/ssd/code/Qwen3_vl`
- 主要项目目录：`/mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime`
- Python 环境：`/mnt/ssd/miniconda3/envs/vlm/bin/python`
- Torchrun：`/mnt/ssd/miniconda3/envs/vlm/bin/torchrun`
- 模型路径：`/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct`
- 视频帧路径：`/mnt/ssd/code/Qwen3_vl/frames`
- baseline 输出目录习惯：`/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d)`
- 常用同步目标：
  - `10.126.126.3`
  - `10.126.126.4`
- 常用同步命令：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```

## 协作风格迁移

这部分不是隐藏指令的逐字复制，而是当前对话中形成的可见协作习惯总结，方便新对话延续同一种配合方式。

- 默认使用中文回答。
- 用户喜欢直接推进，不喜欢只给空泛计划。能读代码、能改代码、能跑测试时，应该直接做。
- 用户重视架构顺序和命名语义，不只是代码能跑。尤其强调：先有独立 PP 和独立 TP，最后才是 HYBRID 组合。
- 用户会提出“为什么这个文件没变化”“为什么只有这俩有，TP 呢”“有没有重复代码”等架构一致性问题。回答时要解释依赖方向和设计取舍。
- 用户偏好 vLLM 风格命名，但不想为了类而堆空壳。类名要服务职责，不要制造“看起来很工程化但没方法”的空架子。
- 用户已经确定主路径对象名为 `StageState`，不想继续混用旧的 `bundle` 语义。
- 当用户说“把这个完成”“同步一下”“继续下一步”时，通常希望直接修改代码、跑相关测试、同步 Jetson。
- 当用户问状态或汇报稿时，需要给能直接对老师/同事说的版本，分清“已经落地”和“后续计划”。
- 语气建议：温暖、清楚、直接，像一起推进项目的搭档。可以说“我们”，但不要过度客套。
- 不要让用户重复上下文；先查本地文件和日志。
- 不要做破坏性 git 操作。不要回滚用户改动。
- 手动文件编辑使用 `apply_patch`。
- 搜索优先 `rg`，文件列表优先 `rg --files`。
- 代码变更后尽量跑最相关的测试；如果改了分布式入口或公共导出，至少跑 CLI/export/loader 相关测试。
- 同步到 Jetson 前最好先本地验证。

## 项目目标

这个项目不是简单调用 HuggingFace Qwen3-VL，而是把 Qwen3-VL 拆成一个可验证、可切分、可并行化的分布式推理 runtime 原型。

当前核心目标已经演进为：

- 主路径不再依赖提前 capture 出来的 `replay bundle` 或 manifest replay 产物。
- 启动时直接从 `model_path` 构建每个 stage/rank 的 `StageState`。
- 纯 PP、纯 TP、PP+TP HYBRID 都要成为主路径。
- TP 不能是每张卡加载完整权重再计算时切分，而要让每张卡只 materialize 自己那份 rank-local shard。
- 多模态 runtime 要收口到 stage-only / shard-only 形态：
  - stage0 才负责视觉 frontend。
  - non-stage0 不跑视觉 frontend，只消费 startup contract / stage handoff。
  - 每个 PP stage 只加载自己的 decoder 层。
  - TP rank 只加载自己的 projection shard。
  - startup transport 只携带薄 metadata/tensor，不携带 full/root/replay payload。

## 当前架构状态

### 总体依赖方向

当前必须保持这个依赖方向：

```text
PP 基础后端      TP 基础后端
     \          /
      \        /
       HYBRID 组合后端
```

含义：

- `pipeline_parallel.py` 是纯 PP 后端。
- `tensor_parallel.py` 是纯 TP 后端。
- `hybrid_parallel.py` 是 PP+TP 组合后端。
- `hexgen_core/modules/` 目录只放 `pipeline_parallel.py`、`tensor_parallel.py`、`hybrid_parallel.py` 和 `__init__.py`。
- debug/replay helper 放在顶层 `qwen3vl_tp_runtime/debug/`，不放在 `hexgen_core/modules/` 下。
- HYBRID 可以调用 TP 的底层执行函数和公共 helper。
- TP 不能调用 HYBRID，也不能借用 `TextHybridRunner`。
- PP 和 TP 是并列基础层，PP 不应该为了少几行重复代码反向依赖 TP。

最近已经完成的关键重构：

- `backend=tp` 不再借 `TextHybridManifest` 或 `TextHybridRunner`。
- 新增独立 `TensorParallelManifest`。
- 新增 `build_direct_tp_manifest`。
- `runtime_cli._load_tp_manifest_for_args()` 现在调用 `build_direct_tp_manifest`。
- `tensor_parallel.py` 中 grep `TextHybridManifest|hybrid_parallel|TextHybridRunner|run_text_hybrid` 应该没有结果。
- `hybrid_parallel.py` 可以 import `tensor_parallel.py` 的 helper，因为 HYBRID 依赖 TP 是合理方向。

### 主路径入口

主 CLI：

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python qwen3vl_tp_runtime/scripts/runtime.py
```

分布式一般使用：

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/torchrun \
  --nnodes ... --nproc-per-node 1 --node-rank ... \
  --master-addr ... --master-port ... \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/scripts/runtime.py \
  ...
```

推荐用户参数：

- 纯 PP：`--backend pp --pp 2`
- 纯 TP：`--backend tp --tp 2`
- 均匀 HYBRID：`--backend hybrid --pp 2 --tp 2`
- 异构 HYBRID：`--backend hybrid --pp 2 --tp-degrees 2 1`

高级覆盖仍然保留：

- `--stage-ranges 0:17 18:35`
- `--tp-degrees 2 1`

但推荐路径已经是 `--pp` / `--tp` 风格，类似 vLLM 参数表达。

### 关键文件职责

- `qwen3vl_tp_runtime/scripts/runtime.py`
  - 统一 CLI 主入口。
  - `backend=pp` 走 PP runner。
  - `backend=tp` 走 `run_tensor_parallel_rank`。
  - `backend=hybrid` 走 `TextHybridRunner`。
  - 因为 torchrun 直接执行脚本，保留绝对 import 是合理的。

- `qwen3vl_tp_runtime/scripts/runtime_cli.py`
  - CLI 默认值、参数校验、debug path gating。
  - `ParallelConfig` 负责把 `--pp` / `--tp` 解析成 `stage_ranges` / `tp_degrees`。
  - `_load_pipeline_manifest_for_args()` 构建 direct PP manifest。
  - `_load_tp_manifest_for_args()` 构建 direct TP manifest。
  - `_load_hybrid_manifest_for_args()` 构建 direct HYBRID manifest。

- `qwen3vl_tp_runtime/scripts/runtime_replay.py`
  - debug-only manifest replay loader。
  - `load_debug_pipeline_manifest`
  - `load_debug_tp_manifest`
  - `load_debug_hybrid_manifest`
  - `--manifest-path` 只应该通过这里进入 replay。

- `qwen3vl_tp_runtime/scripts/runtime_summary.py`
  - 输出 JSON summary。
  - TP 和 HYBRID 当前共用 `_summarize_hybrid_run()`，因为 summary 字段结构类似。
  - 这不是执行依赖，只是输出格式复用。

- `qwen3vl_tp_runtime/hexgen_core/schema.py`
  - `StageState: TypeAlias = dict[str, Any]`
  - `StageSpec`
  - `TextPipelineManifest`
  - `TensorParallelManifest`
  - `TextHybridManifest`
  - `HybridRankContext`
  - `StageHandoffPayload`
  - 这里仍有 replay 兼容属性：`bundle_path` / `bundle_dir`，但主路径通过 `replay` 字段区分 direct/replay。

- `qwen3vl_tp_runtime/hexgen_core/modules/pipeline_parallel.py`
  - 纯 PP 后端。
  - 主 direct loader：`load_stage_state_by_index`、`load_stage_state_for_rank`
  - 主 runner：`TextPipelineRunner`、`TextGeneratePipelineRunner`
  - replay/capture prepare 函数保留在 `LEGACY_REPLAY_EXPORTS`。

- `qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py`
  - 纯 TP 后端。
  - `TensorParallelManifest`
  - `TensorParallelRunner`
  - `load_tp_manifest`
  - `load_stage_state_for_tp_rank`
  - `run_tensor_parallel_rank`
  - `run_stage_state_tp`
  - direct `__all__` 只保留主路径 TP 符号；旧 captured-bundle replay runner 不在这个文件里。
  - HYBRID 现在复用这里的一些 helper。

- `qwen3vl_tp_runtime/debug/tp_debug.py`
  - debug-only TP compare / layer trace / outlier dump helper。
  - `TpDebugConfig`
  - `build_stage_traces`

- `qwen3vl_tp_runtime/debug/tensor_parallel_replay.py`
  - 旧 captured-bundle TP replay 入口。
  - `DEBUG_REPLAY_EXPORTS`
  - `load_text_stage_bundle`
  - `run_text_tensor_parallel_stage`
  - `TextTensorParallelRunner`
  - `run_text_tensor_parallel_rank`

- `qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
  - PP+TP HYBRID 组合后端。
  - 可以调用 `tensor_parallel.py` 中的 TP 执行函数和 generate helper。
  - 不应该再复制和 TP 完全一样的基础 helper。
  - 自己仍保留 HYBRID-specific 逻辑：rank layout、stage group、handoff、non-first stage 输入、runtime-only stage type 判断。

- `qwen3vl_tp_runtime/hexgen_core/transport.py`
  - 当前 transport 封装名为 `StageCommunicator`。
  - 旧 `StageHandoffTransport` 保留为 compat subclass。

- `qwen3vl_tp_runtime/hexgen_core/stage.py`
  - StageState view / handoff payload / stage run helper。
  - `StageStateView` 和 `as_stage_state_view` 是主语义。
  - `StageBundleView` 等只应作为 legacy replay compatibility。

- `qwen3vl_tp_runtime/models/qwen3vl/runtime_builder.py`
  - direct runtime builder。
  - `DirectStageStateBuilder`
  - `StageStateLoader`
  - `build_direct_stage_state`
  - `materialize_text_stage_state`
  - `build_direct_pipeline_manifest`
  - `build_direct_tp_manifest`
  - `build_direct_hybrid_manifest`
  - 主路径实现已经以 `StageState` 命名；`build_direct_stage_bundle` / `DirectStageBundleBuilder` 只作为 legacy alias 保留。
  - 文件内仍有 `layer_bundle`、`bundle_path` 等名字，语义是 layer weight 集合或 replay/file-backed reference，不是主路径 stage runtime object。

- `qwen3vl_tp_runtime/models/qwen3vl/runtime_text_stage.py`
  - text scaffold compaction、runtime input rebuild、本地 stage materialization helper。
  - 当前文件内还有 `layer_bundle` 等历史变量名，因为 weight loader 层仍以 layer bundle 表达参数集合。语义上属于 StageState 内部层参数，不是主路径 stage bundle。

- `qwen3vl_tp_runtime/models/qwen3vl/runtime_text.py`
  - text prompt metadata、runtime-only scaffold restore、启动 session helper。

- `qwen3vl_tp_runtime/models/qwen3vl/weights/`
  - 模型权重 index / load plan / shard slicing。
  - TP shard-local weight materialize 的关键底座。

## 命名约定

已经确定的术语：

- 主路径本地执行对象：`StageState`
- replay/capture/debug 旧产物：`bundle` / `replay bundle`
- 本地 rank materialize：`materialize_text_stage_state`
- direct builder：`DirectStageStateBuilder`
- stage state loader：`StageStateLoader`
- pure TP manifest：`TensorParallelManifest`
- pure TP runner：`TensorParallelRunner`
- pure TP rank入口：`run_tensor_parallel_rank`
- TP stage 执行函数：`run_stage_state_tp`
- HYBRID runner：`TextHybridRunner`
- transport 封装：`StageCommunicator`

注意：

- `bundle` 不应该再出现在主路径用户概念中。
- 但 `bundle` 可以继续出现在 replay/debug/capture/file-backed reference、schema legacy compat 或底层 layer weight 兼容逻辑里。
- `build_stage_bundle` / `DirectStageBundleBuilder` / `build_direct_stage_bundle` 是 legacy alias，不进入 direct `__all__`。
- 用户不喜欢带 `Runtime` 的名字，所以之前选定的是 `StageState`，不是 `StageRuntimeState`。
- 用户喜欢 vLLM 风格类名，但不希望为了类名创建大量没有方法的空类。

## 已完成的重要里程碑

### 1. 多模态 PP generate 已通过

用户曾在 Jetson 上跑过：

- `backend=pp`
- `modality=multimodal`
- `mode=generate`
- `stage-ranges 0:17 18:35`
- `max-new-tokens 4`

当前 baseline 记录：

- logs:
  - `baseline_runs/20260427/pp-mm-generate-rank0.log`
  - `baseline_runs/20260427/pp-mm-generate-rank1.log`
- `generated_token_ids`: `[87140, 15946, 3837, 101177]`
- `generated_text`: `视频中，一名`
- stage0:
  - frontend active
  - loaded layers `0..17`
  - top-level weight: `embed_tokens_weight`
- stage1:
  - frontend consume-only
  - loaded layers `18..35`
  - top-level weights: `final_norm_weight`, `lm_head_weight`

### 2. 多模态 HYBRID generate 已作为里程碑通过

用户早期跑 HYBRID 时曾看到一个 rank1 报错：

```text
RuntimeError: cached_attention_tp: attention mask 与 key/query 长度不匹配
hidden_states_shape=(1, 627, 2560)
query_shape=(1, 16, 627, 128)
key_shape=(1, 4, 1254, 128)
attention_mask_shape=(1, 1, 1, 628)
past_key_shape=(1, 4, 627, 128)
```

随后用户指出 rank0 log 已经生成，并决定“没事已经可以算已完成”。后续文档和 baseline 以 `baseline_runs/20260427/hybrid-mm-generate-rank0.log`、`rank1.log`、`rank2.log` 的 smoke 作为当前验收记录。

当前 baseline 记录：

- stage0 rank0/rank1:
  - `tp_weight_sharded=true`
  - `tp_shard_rank=0/2` and `1/2`
  - `loaded_weight_tensor_bytes=2594763776`
  - `tp_stage_loaded_weight_tensor_bytes_equal=true`
- stage1 rank2:
  - consume-only
  - `tp_weight_sharded=false`
  - loaded layers `18..35`
  - top-level weights `final_norm_weight`, `lm_head_weight`
  - `loaded_weight_tensor_bytes=4411426816`
- all ranks:
  - `generated_token_ids=[87140, 15946, 3837, 101177]`
  - `generated_text=视频中，一名`

### 3. 纯 TP text generate 已通过

用户后来跑了纯 TP text generate，并说“可以，通过了”。

当前 baseline 记录：

- `tp-text-generate`
- `rank0`: `tp_weight_sharded=true`, `tp_shard_rank=0`, `tp_shard_world_size=2`
- `rank1`: `tp_weight_sharded=true`, `tp_shard_rank=1`, `tp_shard_world_size=2`
- rank0/rank1 `loaded_weight_tensor_bytes=5189532672`
- `generated_token_ids=[104455, 9909, 9286, 16488]`
- `generated_text=人工智能（Artificial`

### 4. StageState 术语迁移完成

已完成内容：

- 文档主路径改为 `StageState`。
- `StageState` 类型别名已加入 schema。
- 新函数名 wrapper 已加入。
- 主运行路径变量从 `stage_bundle` 收口为 `stage_state`。
- `bundle` 只保留给 replay/debug/capture/file-backed reference。

### 5. TP 独立后端完成

用户明确指出：

```text
顺序应该是这样，先有tp和pp，这俩是并列的，然后最后有的混合吧！
```

随后完成：

- `TensorParallelManifest` 独立存在于 `schema.py`。
- `build_direct_tp_manifest` 独立存在于 `runtime_builder.py`。
- `runtime_cli._load_tp_manifest_for_args()` 走 `build_direct_tp_manifest`。
- `tensor_parallel.py` 不再 import `hybrid_parallel`。
- `tensor_parallel.py` 不再依赖 `TextHybridRunner`。
- `hybrid_parallel.py` 仍可调用 `run_stage_state_tp`。
- 当前依赖方向变成 `HYBRID -> TP`，不是 `TP -> HYBRID`。

### 6. HYBRID 复用 TP helper，删除重复代码

用户指出：

```text
因为混合并行是依赖tp并行和pp并行的，所以混合并行能调用函数的可以直接调用了，不用在新创文件
```

因此没有新建 `generate_common.py`，而是让 `hybrid_parallel.py` 直接从 `tensor_parallel.py` import 可复用 helper。

后续又把这组 HYBRID 复用的 TP helper 从 `_` 私有名规范成无下划线 module-level helper；它们是 backend-level 复用点，但不进入 `tensor_parallel.py.__all__`，也不进入 `hexgen_core.modules.__all__`。

已删除 HYBRID 本地重复定义并改为复用 TP：

- `build_generate_phase_state`
- `strip_runtime_layer_cache`
- `build_generate_cache_map`
- `is_runtime_only_generate_state`
- `infer_runtime_tensor_device`
- `infer_runtime_tensor_dtype`
- `infer_runtime_token_dtype`
- `build_runtime_only_stage_input_template`
- `token_tensor_to_list`
- `broadcast_token_id`

仍保留在 HYBRID：

- `_build_runtime_only_text_generate_phase_state`

原因：

- HYBRID 需要根据当前 stage 是否 last stage 判断 `text_prefill_last` / `text` / `text_decode_last` / `text_decode`。
- 纯 TP 永远是单 stage / last stage，语义不完全相同。

## 当前测试记录

本轮完整 baseline 已冻结到 `baseline_runs/20260428/`：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py --case-id pp-text-generate \
  baseline_runs/20260428/pp-text-generate-rank0.log \
  baseline_runs/20260428/pp-text-generate-rank1.log

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py --case-id pp-mm-generate \
  baseline_runs/20260428/pp-mm-generate-rank0.log \
  baseline_runs/20260428/pp-mm-generate-rank1.log

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py --case-id tp-text-generate \
  baseline_runs/20260428/tp-text-generate-rank0.log \
  baseline_runs/20260428/tp-text-generate-rank1.log

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py --case-id tp-mm-generate \
  baseline_runs/20260428/tp-mm-generate-rank0.log \
  baseline_runs/20260428/tp-mm-generate-rank1.log

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py --case-id hybrid-text-generate \
  baseline_runs/20260428/hybrid-text-generate-rank0.log \
  baseline_runs/20260428/hybrid-text-generate-rank1.log \
  baseline_runs/20260428/hybrid-text-generate-rank2.log

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py --case-id hybrid-mm-generate \
  baseline_runs/20260428/hybrid-mm-generate-rank0.log \
  baseline_runs/20260428/hybrid-mm-generate-rank1.log \
  baseline_runs/20260428/hybrid-mm-generate-rank2.log
```

全部 PASS，汇总保存为：

- `baseline_runs/20260428/check-baseline-logs.txt`

本轮 baseline 期间修复：

- `qwen3vl_tp_runtime/models/qwen3vl/live/inputs.py` 补 `MmVisualState` import。
- `qwen3vl_tp_runtime/scripts/live/live_multimodal_runtime.py` 的 generate summary 补 `generated_token_ids` / `generated_text`。
- `qwen3vl_tp_runtime/models/qwen3vl/runtime_builder.py` 修复 pure TP multimodal full-stage prefill handoff 越界。

新增测试：

- `test/test_check_baseline_logs.py`
- `test/test_runtime_builder_handoffs.py`

runtime core 最小回归矩阵已固化为：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
```

权重加载相关改动额外加：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --include-weight-loader
```

默认矩阵会跑核心 direct loader / CLI / summary / compat 单测，并检查 `baseline_runs/20260428` 下 6 个 distributed case 的 frozen rank logs。

本轮 runtime smoke wrapper 固化后已运行并通过：

```bash
bash -n \
  qwen3vl_tp_runtime/scripts/helpers/run-runtime.sh \
  qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh \
  qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh \
  qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh

NODE_RANK=0 DRY_RUN=1 bash qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh --save-dtype bfloat16
NODE_RANK=1 DRY_RUN=1 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh --save-dtype bfloat16
NODE_RANK=2 DRY_RUN=1 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh --save-dtype bfloat16
```

后续已补充 PP/TP wrapper 的多 Jetson 鲁棒性验证：

```bash
OUT=/tmp/qwen-smoke-dryrun NNODES=4 NODE_RANK=3 DRY_RUN=1 \
  bash qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh
OUT=/tmp/qwen-smoke-dryrun NNODES=4 NODE_RANK=3 DRY_RUN=1 \
  bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh

# 以下错误配置均应在 wrapper 层 exit 2。
OUT=/tmp/qwen-smoke-dryrun NNODES=1 NODE_RANK=0 DRY_RUN=1 \
  bash qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh
OUT=/tmp/qwen-smoke-dryrun NNODES=4 PP=2 NODE_RANK=0 DRY_RUN=1 \
  bash qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh
OUT=/tmp/qwen-smoke-dryrun NNODES=4 TP=2 NODE_RANK=0 DRY_RUN=1 \
  bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
OUT=/tmp/qwen-smoke-dryrun NNODES=4 NODE_RANK=4 DRY_RUN=1 \
  bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
```

本轮 StageState / bundle 命名收紧后已运行并通过：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python -m py_compile \
  qwen3vl_tp_runtime/hexgen_core/stage.py \
  qwen3vl_tp_runtime/hexgen_core/modules/pipeline_parallel.py \
  qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py \
  qwen3vl_tp_runtime/debug/tp_debug.py \
  qwen3vl_tp_runtime/models/qwen3vl/runtime_text.py \
  qwen3vl_tp_runtime/models/qwen3vl/runtime_builder.py

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_schema_direct_manifest.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tp_debug.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_pipeline_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_qwen3vl_exports.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_cli_modes.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_stage_handoff.py
```

本轮 HYBRID 调用 TP helper 规范后已运行并通过：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python -m py_compile \
  qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py \
  qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py \
  test/test_compat_package_exports.py

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_cli_modes.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_summary.py
```

本轮 TP debug/replay 搬家后已运行并通过：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python -m py_compile \
  qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py \
  qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py \
  qwen3vl_tp_runtime/hexgen_core/modules/__init__.py \
  qwen3vl_tp_runtime/debug/__init__.py \
  qwen3vl_tp_runtime/debug/tp_debug.py \
  qwen3vl_tp_runtime/debug/tensor_parallel_replay.py \
  test/test_compat_package_exports.py

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tp_debug.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_summary.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_cli_modes.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_qwen3vl_exports.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_schema_direct_manifest.py
```

最近几轮已运行并通过的测试：

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python -m py_compile \
  qwen3vl_tp_runtime/hexgen_core/schema.py \
  qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py \
  qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py \
  qwen3vl_tp_runtime/hexgen_core/__init__.py \
  qwen3vl_tp_runtime/hexgen_core/modules/__init__.py \
  qwen3vl_tp_runtime/models/qwen3vl/runtime_builder.py \
  qwen3vl_tp_runtime/models/qwen3vl/__init__.py \
  qwen3vl_tp_runtime/models/__init__.py \
  qwen3vl_tp_runtime/scripts/runtime.py \
  qwen3vl_tp_runtime/scripts/runtime_cli.py \
  qwen3vl_tp_runtime/scripts/runtime_replay.py \
  test/test_tensor_parallel_direct.py \
  test/test_runtime_cli_modes.py \
  test/test_schema_direct_manifest.py \
  test/test_compat_package_exports.py \
  test/test_qwen3vl_exports.py
```

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_cli_modes.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_schema_direct_manifest.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_qwen3vl_exports.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_pipeline_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py
```

测试结果：

- `test/test_tensor_parallel_direct.py`: OK
- `test/test_runtime_cli_modes.py`: OK
- `test/test_schema_direct_manifest.py`: OK
- `test/test_compat_package_exports.py`: OK
- `test/test_qwen3vl_exports.py`: OK
- `test/test_pipeline_direct_loader.py`: OK
- `test/test_hybrid_direct_loader.py`: OK
- `test/test_model_weight_loader.py`: OK

已知 warning：

- `test_model_weight_loader.py` 在当前 Jetson 环境可能打印 CUDA/NvRmMem warning：

```text
NvRmMemInitNvmap failed with No such file or directory
Memory Manager Not supported
CUDA initialization: CUDA unknown error
```

但测试最终是 OK。这个 warning 当前不作为失败。

最近两次代码同步都已完成：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```

## 当前运行命令模板

优先使用 smoke wrapper，减少分布式多 rank 手动复制出错：

```bash
# PP multimodal generate, 2 nodes.
NODE_RANK=0 MASTER_ADDR=10.126.126.2 bash qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.2 bash qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh

# TP multimodal generate, 2 nodes.
NODE_RANK=0 MASTER_ADDR=10.126.126.2 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.2 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh

# HYBRID multimodal generate, 3 nodes.
NODE_RANK=0 MASTER_ADDR=10.126.126.2 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.2 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
NODE_RANK=2 MASTER_ADDR=10.126.126.2 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
```

wrapper 默认 `OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d)`，会打印最终命令并写 `case-rankN.log`。可用 `MASTER_PORT`、`MODEL_PATH`、`FRAME_DIR`、`MM_PROMPT`、`MAX_NEW_TOKENS` 覆盖默认值；`DRY_RUN=1` 只打印命令。

### 纯 TP text generate，2 节点

rank0：

```bash
OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d); mkdir -p "$OUT"
/mnt/ssd/miniconda3/envs/vlm/bin/torchrun \
  --nnodes 2 --nproc-per-node 1 --node-rank 0 \
  --master-addr 10.126.126.2 --master-port 29580 \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/scripts/runtime.py \
  --backend tp --modality text --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --tp 2 \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 2>&1 | tee "$OUT/tp-text-generate-rank0.log"
```

rank1：

```bash
OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d); mkdir -p "$OUT"
/mnt/ssd/miniconda3/envs/vlm/bin/torchrun \
  --nnodes 2 --nproc-per-node 1 --node-rank 1 \
  --master-addr 10.126.126.2 --master-port 29580 \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/scripts/runtime.py \
  --backend tp --modality text --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --tp 2 \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 2>&1 | tee "$OUT/tp-text-generate-rank1.log"
```

### PP multimodal generate，2 节点

rank0：

```bash
OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d); mkdir -p "$OUT"
HEXGEN_STARTUP_LOG=1 /mnt/ssd/miniconda3/envs/vlm/bin/torchrun \
  --nnodes 2 --nproc-per-node 1 --node-rank 0 \
  --master-addr 10.126.126.2 --master-port 29572 \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/scripts/runtime.py \
  --backend pp --modality multimodal --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --frame-dir /mnt/ssd/code/Qwen3_vl/frames \
  --num-frames 8 \
  --pp 2 \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 2>&1 | tee "$OUT/pp-mm-generate-rank0.log"
```

rank1：

```bash
OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d); mkdir -p "$OUT"
HEXGEN_STARTUP_LOG=1 /mnt/ssd/miniconda3/envs/vlm/bin/torchrun \
  --nnodes 2 --nproc-per-node 1 --node-rank 1 \
  --master-addr 10.126.126.2 --master-port 29572 \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/scripts/runtime.py \
  --backend pp --modality multimodal --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --frame-dir /mnt/ssd/code/Qwen3_vl/frames \
  --num-frames 8 \
  --pp 2 \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 2>&1 | tee "$OUT/pp-mm-generate-rank1.log"
```

### HYBRID multimodal generate，3 节点

当前 baseline 是 2 个 PP stage，stage0 TP=2，stage1 TP=1：

```text
rank0 -> stage0 local_rank0 / tp_degree2
rank1 -> stage0 local_rank1 / tp_degree2
rank2 -> stage1 local_rank0 / tp_degree1
```

rank0：

```bash
OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d); mkdir -p "$OUT"
HEXGEN_STARTUP_LOG=1 /mnt/ssd/miniconda3/envs/vlm/bin/torchrun \
  --nnodes 3 --nproc-per-node 1 --node-rank 0 \
  --master-addr 10.126.126.2 --master-port 29573 \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/scripts/runtime.py \
  --backend hybrid --modality multimodal --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --frame-dir /mnt/ssd/code/Qwen3_vl/frames \
  --num-frames 8 \
  --pp 2 --tp-degrees 2 1 \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 2>&1 | tee "$OUT/hybrid-mm-generate-rank0.log"
```

rank1：

```bash
OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d); mkdir -p "$OUT"
HEXGEN_STARTUP_LOG=1 /mnt/ssd/miniconda3/envs/vlm/bin/torchrun \
  --nnodes 3 --nproc-per-node 1 --node-rank 1 \
  --master-addr 10.126.126.2 --master-port 29573 \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/scripts/runtime.py \
  --backend hybrid --modality multimodal --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --frame-dir /mnt/ssd/code/Qwen3_vl/frames \
  --num-frames 8 \
  --pp 2 --tp-degrees 2 1 \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 2>&1 | tee "$OUT/hybrid-mm-generate-rank1.log"
```

rank2：

```bash
OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/$(date -u +%Y%m%d); mkdir -p "$OUT"
HEXGEN_STARTUP_LOG=1 /mnt/ssd/miniconda3/envs/vlm/bin/torchrun \
  --nnodes 3 --nproc-per-node 1 --node-rank 2 \
  --master-addr 10.126.126.2 --master-port 29573 \
  /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/scripts/runtime.py \
  --backend hybrid --modality multimodal --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --frame-dir /mnt/ssd/code/Qwen3_vl/frames \
  --num-frames 8 \
  --pp 2 --tp-degrees 2 1 \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 2>&1 | tee "$OUT/hybrid-mm-generate-rank2.log"
```

## 当前已知冗余和保留理由

### StageState / bundle 命名边界已收紧

当前规则：

- direct 主路径、summary 主字段、README 主路径说明统一使用 `StageState`。
- replay/capture/debug/file-backed reference 可以继续使用 `bundle`。
- schema 里的 `bundle_path` / `bundle_dir` 只作为 legacy replay 兼容属性保留；direct manifest 不序列化这些字段。
- execution / weight loader 里的 `layer_bundle` 或 `bundle` 表示层参数集合，不是 stage runtime object。
- `build_stage_bundle` / `DirectStageBundleBuilder` / `build_direct_stage_bundle` 只作为 legacy alias 保留，不进入 direct public `__all__`。

本轮清理：

- `pipeline_parallel.py` docstring 改成 direct `StageState` 主路径。
- `debug.tp_debug.build_stage_traces()` 参数从 `bundle` 改为 `stage_state`。
- `runtime_builder.py` 内部 file-backed reference / trace 临时变量和 helper 从 `stage_bundle` 收口为 `stage_state`。
- 删除未使用的 `_restore_text_prompt_bundle`。

### TP debug/replay 兼容路径已搬家

主路径文件：`hexgen_core/modules/tensor_parallel.py`

当前状态：

- `tensor_parallel.py.__all__` 只保留 direct TP 主路径。
- `tensor_parallel.py` 不再导出 `DEBUG_REPLAY_EXPORTS`。
- `tensor_parallel.py` 不再包含旧 captured-bundle replay runner。

新位置：

- `qwen3vl_tp_runtime/debug/tp_debug.py`
  - `TpDebugConfig`
  - `build_stage_traces`
- `qwen3vl_tp_runtime/debug/tensor_parallel_replay.py`
  - `DEBUG_REPLAY_EXPORTS`
  - `load_text_stage_bundle`
  - `run_text_tensor_parallel_stage`
  - `TextTensorParallelRunner`
  - `run_text_tensor_parallel_rank`

兼容路径：

- `hexgen_core.modules` 仍通过 lazy compat export 暴露旧 replay 名字，但实际实现来自顶层 `qwen3vl_tp_runtime.debug.tensor_parallel_replay`。
- 新代码优先直接 import 顶层 `qwen3vl_tp_runtime.debug.*`。

### TP 的空壳 GenerateWorker / DecodeWorker 已删除

文件：`hexgen_core/modules/tensor_parallel.py`

当前：

- `StageRunner` 有实际逻辑。
- `TensorParallelRunner(StageRunner)` 是纯 TP 公开 runner。
- `GenerateWorker` / `DecodeWorker` 不再出现在 TP 主路径和 `tensor_parallel.py.__all__` 中。

处理原因：

- 用户选择优先删除空壳类，只保留有实际职责的 `StageRunner / TensorParallelRunner`。
- PP/HYBRID 仍保留自己的 `GenerateWorker / DecodeWorker`，本次只清理纯 TP。

### TensorParallelManifest 有固定 layout 字段

文件：`hexgen_core/schema.py`

纯 TP 里这些字段固定或为空：

- `stage_rank_groups`
- `pp_rank_groups`
- `send_list`
- `recv_list`
- `send_empty_list`
- `recv_empty_list`

保留理由：

- summary 和 loader 结构复用更方便。
- 兼容旧 manifest dict 中带 layout 字段的情况。
- 让 TP/HYBRID 输出字段结构更接近。

后续可选清理：

- 如果想更纯，可以让 TP summary 不依赖这些字段，然后简化 `TensorParallelManifest`。

### PP 与 TP 仍有部分 generate helper 重复

`pipeline_parallel.py` 和 `tensor_parallel.py` 仍各自有：

- generate phase state 构造
- runtime layer cache 剥离
- generate cache map 构造
- runtime-only generate 判断
- token broadcast

保留理由：

- PP 和 TP 是并列基础层。
- 用户明确希望 HYBRID 依赖 TP/PP，但没有要求 PP 依赖 TP。
- 不能为了少几行重复，让 PP 反向依赖 TP。

后续可选清理：

- 如果用户同意，也可以新建基础公共文件；但用户刚刚明确说“不用新创文件”，所以当前不做。

### HYBRID runtime-only phase builder 没有抽到 TP

文件：`hybrid_parallel.py`

保留：

- `_build_runtime_only_text_generate_phase_state`

原因：

- HYBRID 需要知道当前 stage 是否 last stage。
- TP 永远单 stage last。
- 两者类似但不完全等价，硬抽容易引入 stage_type 错误。

## 重要设计决策时间线

### 早期问题：multimodal startup contract 缺少 stage0 handoff

用户遇到过：

```text
RuntimeError: multimodal startup contract 缺少 stage0 handoff activation。
```

这个问题推动了 multimodal startup contract 的设计收口：

- stage0 负责构建 frontend/handoff。
- non-stage0 不能自己跑视觉前端。
- startup contract 必须被 seed / broadcast / consume。

### PP multimodal generate 过线

用户提供 rank0/rank1 日志，PP 2 stage 运行成功。

关键事实：

- stage0 active frontend
- stage1 consume-only
- generated ids 一致
- stage scope 正确

### HYBRID multimodal generate 过线

用户提供 rank1 曾经的 attention mask mismatch 日志，但随后指出 rank0 已生成，并决定该目标已完成。

后来文档基线按 `hybrid-mm-generate` 已通过记录。

### 用户确认两个总目标是否完成

用户问：

```text
改成在启动时直接从 model_path 构建每个 stage/rank 的运行参数
要改成每张卡只拿自己那份权重
现在这两个总目标已经完成了吗
```

当前回答应是：

- 对主路径 PP/TP/HYBRID：已经完成。
- 对 text decoder projection TP shard：已经完成。
- 对 embedding/lm_head vocab parallelism：未完成，当前仍复制。
- 对 replay/capture debug 路径：仍保留旧 bundle，不属于主路径验收。

### StageState 命名确定

用户先问：

```text
我们的bundle可以换一个名字吗，因为之前旧的离线 replay bundle容易混淆
```

讨论后确定：

```text
StageState确定是这个了
```

此后所有主路径文档和代码都应以 `StageState` 为准。

### TP 和 PP 的架构顺序确认

用户指出：

```text
backend=tp还是要单独的一个文件的，因为我做的不止是混合模式，还有单独的pp，单独的tp呢
```

后来进一步明确：

```text
顺序应该是这样，先有tp和pp，这俩是并列的，然后最后有的混合吧！
```

这成为当前最重要架构约束。

### vLLM 风格命名讨论

用户提出可以学习 vLLM 风格命名，列过一张表：

| 当前想抽的类 | vLLM 风格名字 | 位置 |
| --- | --- | --- |
| `ParallelPlan` | `ParallelConfig` / `ParallelismConfig` | `scripts/runtime_cli.py` 或 `hexgen_core/parallel_config.py` |
| `MmStartupContract` | `MultimodalMetadata` / `MultimodalStartupMetadata` | `models/qwen3vl/runtime_mm_stage.py` 或新文件 |
| `MmStartupTransportCodec` | `MultimodalMetadataCodec` | 同上 |
| `TextScaffoldTransportCodec` | `TextMetadataCodec` / `TextScaffoldCodec` | `runtime_text_stage.py` |
| `ReferenceStateBuilder` | `ReferenceStateBuilder` / `ReferenceModelRunner` | `runtime_builder.py` 拆出去 |
| `StageStateMaterializer` | `StageStateLoader` | `models/qwen3vl/weights/` 或 `runtime_builder.py` |
| `PP/HYBRID generate phase` | `GenerateWorker` / `DecodeWorker` / `StageRunner` | `hexgen_core/modules/` |
| `TP direct runner` | `StageRunner` / `TensorParallelRunner` | `hexgen_core/modules/tensor_parallel.py` |
| `transport 封装` | `StageCommunicator` | `hexgen_core/transport.py` |

当前已完成：

- `ParallelConfig`
- `StageCommunicator`
- `StageStateLoader`
- PP/HYBRID 的 `StageRunner` / `GenerateWorker` / `DecodeWorker`
- TP 的 `StageRunner` / `TensorParallelRunner`

但注意用户后来对空类提出疑问，所以后续不要继续机械加类。

## 仍未完成或后续可做

### 1. TensorParallelManifest 简化

当前为了 summary/兼容保留了一些固定 layout 字段。

可选方案：

- 简化为只保存 `tp_degree`、`stage_ranges`、`stages`、`runtime_config`。
- 在 summary 时动态生成固定 layout 字段。

但这会影响测试和兼容 loader，不是优先任务。

### 3. HYBRID 调用 TP helper 已规范

`hybrid_parallel.py` 现在从 `tensor_parallel.py` import 无下划线的 backend-level helper。

这是按用户偏好做的：不新建公共文件，让 HYBRID 直接复用 TP。

当前状态：

- helper 不进入 `tensor_parallel.py.__all__`。
- helper 不进入 package-level `DIRECT_RUNTIME_EXPORTS`。
- `test/test_compat_package_exports.py` 覆盖 HYBRID 引用的是 TP 同一份实现。

### 4. embedding / lm_head vocab parallelism

当前仍复制：

- `embed_tokens_weight`
- `lm_head_weight`

这是已知后续优化，不属于当前里程碑。

### 5. 更完整的 distributed regression

本地单测已过，但每次改到 runtime core 后，最好在 Jetson 上跑：

- `pp-mm-generate`
- `tp-mm-generate`
- `hybrid-mm-generate`

优先用：

- `scripts/helpers/run-pp-mm-generate.sh`
- `scripts/helpers/run-tp-mm-generate.sh`
- `scripts/helpers/run-hybrid-mm-generate.sh`

PP/TP wrapper 当前已按多 Jetson 场景收紧：

- `NNODES` 默认仍是 2，但允许设置为更大的整数。
- pure PP 默认 `PP=NNODES`，并要求 `PP == NNODES`。
- pure TP 默认 `TP=NNODES`，并要求 `TP == NNODES`。
- 每台机器只需要设置自己的 `NODE_RANK=0..NNODES-1`。

看：

- generated ids
- generated text
- `weight_load.tp_weight_sharded`
- `tp_shard_rank/world_size`
- loaded weight bytes
- stage scope
- startup contract payload 是否仍薄

## 当前主路径导出面

### `hexgen_core.modules.tensor_parallel.__all__`

应包含：

- `TensorParallelManifest`
- `StageRunner`
- `TensorParallelRunner`
- `load_tp_manifest`
- `load_stage_state_for_tp_rank`
- `run_tensor_parallel_rank`
- `run_stage_state_tp`

旧 captured-bundle TP replay 在：

- `qwen3vl_tp_runtime.debug.tensor_parallel_replay.DEBUG_REPLAY_EXPORTS`
- `hexgen_core.modules.LEGACY_REPLAY_EXPORTS` 中保留 lazy compat 名字

### `hexgen_core.modules.pipeline_parallel.__all__`

应只包含 direct runner/loader。

prepare/capture replay 入口在：

- `LEGACY_REPLAY_EXPORTS`

### `hexgen_core.modules.hybrid_parallel.__all__`

应只包含 direct runner/loader/layout 相关。

prepare/capture replay 入口在：

- `LEGACY_REPLAY_EXPORTS`

### package-level exports

已收口目标：

- `__all__` 只代表 direct 主路径。
- legacy replay/capture 入口进入 `LEGACY_*_EXPORTS`。
- debug-only 入口进入顶层 `qwen3vl_tp_runtime.debug.*`，必要时由 compat package lazy 转发。

相关测试：

- `test/test_compat_package_exports.py`
- `test/test_qwen3vl_exports.py`

## 代码检查建议

### 检查 TP 是否反向依赖 HYBRID

```bash
rg -n "TextHybridManifest|hybrid_parallel|TextHybridRunner|run_text_hybrid" \
  qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py
```

期望：无结果。

### 检查 HYBRID 是否复用 TP helper

```bash
rg -n "from \\.tensor_parallel import" \
  qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py
```

应看到：

- `run_stage_state_tp`
- `build_generate_phase_state`
- `strip_runtime_layer_cache`
- `build_generate_cache_map`
- `is_runtime_only_generate_state`
- `infer_runtime_tensor_device`
- `infer_runtime_tensor_dtype`
- `infer_runtime_token_dtype`
- `build_runtime_only_stage_input_template`
- `token_tensor_to_list`
- `broadcast_token_id`

### 检查主路径是否还出现 stage_bundle

```bash
rg -n "stage_bundle" qwen3vl_tp_runtime \
  -g'*.py' \
  -g'*.md'
```

解释：

- 出现在 capture/replay/debug/file-backed reference 可接受。
- 出现在 schema legacy compat、legacy alias 或 layer weight 集合里可接受。
- 出现在 direct 主路径 public API、用户文档主路径、runtime summary 主字段中要谨慎。

### 检查 direct/debug exports

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py
```

## 文档状态

### `README.md`

已中文化。

记录了：

- 主路径入口
- `--pp` / `--tp`
- 当前状态
- debug path
- 目录结构
- 命名约定

### `ROADMAP.md`

记录了：

- 两个主目标
- multimodal 过线标准
- StageState 命名迁移
- direct/replay schema 分离
- TP/PP/HYBRID 当前状态

### `BASELINE.md`

记录了：

- 固定 case
- baseline 规则
- 验收字段
- 已确认 shard-only smoke
- 已确认 multimodal direct smoke
- 各种运行命令

如果新对话要继续汇报项目状态，优先综合这三个文件和本 handoff。

## 用户偏好和上下文细节

- 用户在 Jetson 集群上跑真实命令，会把 rank 日志贴回来。
- 用户常用主机名：
  - `jetson1`
  - `jetson2`
- 用户关注真实运行，不满足于 unit tests。
- 用户会打开多个文件查看：
  - `ROADMAP.md`
  - `README.md`
  - `BASELINE.md`
  - `tensor_parallel.py`
  - `hybrid_parallel.py`
  - `pipeline_parallel.py`
  - `runtime_builder.py`
  - `stage.py`
  - `runtime_text_stage.py`
- 用户不喜欢旧离线 replay bundle 和新主路径 bundle 混淆。
- 用户已经接受 `StageState` 作为主路径名。
- 用户希望命令像 vLLM 那样简单：
  - `--pp 2`
  - `--tp 2`
  - hybrid 两个都指定。
- 用户有时要求“不要设置环境变量，完整命令给我”，这种情况下要给完整命令，不要只给变量模板。
- 用户希望代码结构“函数名和功能一致、简洁”，类名可以添加，但要真的有意义。
- 用户希望 import 尽量相对路径；但 `scripts/runtime.py` 因直接执行保留绝对 import 是可以解释的。

## P3 性能 / 显存 / transport profile 状态

ROADMAP step 9 和 step 11 已落地。

新增或更新：

- `qwen3vl_tp_runtime/scripts/collect_runtime_perf.py`
- `test/test_collect_runtime_perf.py`
- `qwen3vl_tp_runtime/hexgen_core/schema.py`
- `qwen3vl_tp_runtime/hexgen_core/distributed.py`
- `qwen3vl_tp_runtime/hexgen_core/transport.py`
- `qwen3vl_tp_runtime/scripts/runtime_summary.py`
- `test/test_runtime_summary.py`
- `runtime_metrics` 写入 runtime JSON summary
- `baseline_runs/20260428/runtime-perf-records.json`
- `baseline_runs/20260428/runtime-perf-table.md`

`runtime_metrics` 字段：

- `runtime_metrics.timing.runtime_total_seconds`
- `runtime_metrics.startup.events`
- `runtime_metrics.startup.totals_by_kind.prepare_session_seconds`
- `runtime_metrics.startup.totals_by_kind.startup_contract_transport_seconds`
- `runtime_metrics.startup.totals_by_kind.materialize_stage_seconds`
- `runtime_metrics.startup.totals_by_kind.post_load_barrier_seconds`
- `runtime_metrics.memory.cpu_max_rss_bytes`
- `runtime_metrics.memory.peak_allocated_bytes`
- `runtime_metrics.memory.peak_reserved_bytes`
- `runtime_metrics.transport.events`
- `runtime_metrics.transport.totals_by_kind.startup_contract`
- `runtime_metrics.transport.totals_by_kind.scaffold`
- `runtime_metrics.transport.totals_by_kind.stage_handoff`
- `runtime_metrics.transport.totals_by_kind.tp_collective`
- `runtime_metrics.transport.totals_by_channel.*`

transport/payload profile 当前覆盖：

- object/tensor send / recv / broadcast
- `StageCommunicator` 的 PP/HYBRID stage handoff
- TP helper 的 `all_reduce_cpu` / `all_gather_cpu` / `broadcast_cpu`
- `PayloadSummary` 中的 tensor shape / dtype / numel / bytes

`collect_runtime_perf.py` 额外汇总：

- startup contract bytes / tensor bytes / object bytes
- scaffold bytes / tensor bytes / object bytes
- stage handoff bytes / seconds
- TP collective bytes / seconds
- transport event count

收集命令：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/collect_runtime_perf.py \
  --baseline-dir baseline_runs/20260428 \
  --output-json baseline_runs/20260428/runtime-perf-records.json \
  --output-md baseline_runs/20260428/runtime-perf-table.md
```

当前 `baseline_runs/20260428` 是旧 correctness log：能解析已有 startup timer、`/usr/bin/time -p real` 和 loaded weight bytes，但没有 CUDA peak memory 和新 transport profile 字段；所以当前 perf 表里的 CUDA peak 与 payload bytes 为空。新代码重跑任意 runtime case 后会在 JSON summary 中直接出现 peak allocated/reserved 与 `runtime_metrics.transport`。

step 11 profiling 重跑记录：

- 输出目录：`baseline_runs/20260428-step11-profile/`
- 已重跑：`pp-text-generate`、`pp-mm-generate`、`tp-text-generate`、`tp-mm-generate`
- 拓扑：Jetson2 rank0 + Jetson3 rank1，`MASTER_ADDR=10.126.126.3`
- checker：`baseline_runs/20260428-step11-profile/check-baseline-logs.txt`，四个 case 全部 PASS
- perf 表：`baseline_runs/20260428-step11-profile/runtime-perf-table.md`
- machine-readable：`baseline_runs/20260428-step11-profile/runtime-perf-records.json`
- 已能看到 PP startup/handoff payload bytes 和 TP collective seconds/bytes
- HYBRID 当轮只保留 PP/TP profiling：当时受环境限制不能稳定纳入第 3 个 rank；后续 3-rank HYBRID profile 已由 `baseline_runs/20260429-longterm-profile/` 补齐并冻结。

step 12 startup contract 第一轮减量：

- 目标：向 vLLM-style PP contract 靠拢，主路径只传后续 stage 必须消费的中间量，不传 reference/debug 用的 stage output。
- 已改代码：
  - `runtime_builder.py`
  - `test/test_pipeline_direct_loader.py`
- 行为变化：
  - `pack_mm_startup_transport()` / `DirectStageStateBuilder.export_mm_startup_transport()` 支持 `include_stage_output`。
  - `mode=generate` 且 `include_runtime_reference=false` 时，multimodal startup contract 只导出 `stage_input`，不导出 `stage_output`。
  - reference/debug 或非 generate 路径仍保留 `stage_output`。
  - restore/seed runtime config 可以消费 stage-input-only handoff。
- 本地验证：
  - `test/test_pipeline_direct_loader.py`
  - `test/test_hybrid_direct_loader.py`
  - `python -m py_compile runtime_builder.py pipeline_parallel.py hybrid_parallel.py`
  - `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`
- Jetson2/3 实测目录：`baseline_runs/20260429-startup-contract-opt/`
  - `pp-mm-generate-startup-opt` PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`
  - `hybrid-mm-generate-startup-opt` PASS，2 节点 `--pp 2 --tp-degrees 1 1` 变体，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`
  - startup contract 从 `7,563,328` bytes / 13 tensors 降到 `4,353,088` bytes / 12 tensors
  - 新 payload keys 不再包含 `stage_handoffs.1.stage_output`
- 注意：
  - 本轮是 2 节点 HYBRID startup contract 路径验证；完整 3-rank HYBRID `--pp 2 --tp-degrees 2 1` 后续已由 `baseline_runs/20260429-longterm-profile/` 验证并冻结。

step 12 startup contract 第二轮减量，已把这一步彻底收口：

- 目标：主路径 multimodal generate startup contract 不再传可本地重建的 derived shared tensor。
- 已改代码：
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_mm_stage.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_builder.py`
  - `test/test_pipeline_direct_loader.py`
  - `test/test_runtime_mm.py`
- 行为变化：
  - `compact_mm_runtime_shared(..., include_derived=False)` 可导出 compact shared metadata。
  - `pack_mm_startup_transport()` / `DirectStageStateBuilder.export_mm_startup_transport()` 支持 `include_derived_shared`。
  - `mode=generate` 且 `include_runtime_reference=false` 时，startup contract 不导出 `shared.attention_mask` / `shared.cos` / `shared.sin`。
  - reference/debug 或非 generate 路径仍保留 derived shared tensor。
  - `build_mm_stage_state()` 会用 compact shared + `stage_input` 本地重建 dense `attention_mask` 和 RoPE `cos/sin`。
- 本地验证：
  - `test/test_runtime_mm.py`
  - `test/test_pipeline_direct_loader.py`
  - `test/test_hybrid_direct_loader.py`
  - `test/test_model_weight_loader.py`
  - `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`
- Jetson2/3 实测目录：`baseline_runs/20260429-startup-contract-derived-opt/`
  - `pp-mm-generate-derived-opt` PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`
  - `hybrid-mm-generate-derived-opt` PASS，2 节点 `--pp 2 --tp-degrees 1 1` 变体，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`
  - startup contract 从第一轮后的 `4,353,088` bytes / 12 tensors 降到 `3,245,806` bytes / 9 tensors
  - 新 payload keys 不再包含 `shared.attention_mask` / `shared.cos` / `shared.sin`
- 注意：
  - 当前 compact contract 仍保留 `shared.attention_mask_2d` / `shared.position_ids` / `shared.rope_deltas` / multimodal grid metadata。
  - 完整 3-rank HYBRID `--pp 2 --tp-degrees 2 1` 后续已由 `baseline_runs/20260429-longterm-profile/` 验证并冻结。

step 13 scaffold broadcast 减量第一轮，已完成低风险 alias 去重：

- 目标：先保留 HYBRID stage leader broadcast 机制，只减少明确重复的 scaffold tensor。
- 已改代码：
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_text_stage.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_builder.py`
  - `test/test_model_weight_loader.py`
- 行为变化：
  - `pack_text_scaffold_transport()` 会识别 tensor alias，重复 tensor 只发送一次，其他位置用 tensor ref 指向首个 payload key。
  - multimodal runtime-only generate 且 `include_text_weights=False` 时，如果 `stage_input` 和 `layer_input` 是同一个 alias，weightless scaffold 会移除 `layer_input`。
  - restore 后 alias ref 仍能还原出 `layer_input` 指向同一份 runtime input；builder compact 路径则直接避免传 `layer_input`。
- 本轮 before/after 记录：
  - before: 3 tensors，216 bytes，keys=`scaffold.stage_input` / `scaffold.layer_input` / `scaffold.prefill_attention_mask_2d`
  - after: 2 tensors，120 bytes，keys=`scaffold.stage_input` / `scaffold.prefill_attention_mask_2d`
- 本地验证：
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
- 注意：
  - 本轮是 HYBRID scaffold alias 去重的单元 slice；完整 3-rank HYBRID 真实 profile 后续已由 `baseline_runs/20260429-longterm-profile/` 冻结。
  - 后续 step 13 已继续完成 `prefill_attention_mask_2d` 和 top-level derived tensor 本地重建。

step 13 scaffold broadcast 减量第二轮，已完成 rank-local 可推导字段移除：

- 目标：HYBRID stage leader 不再把 rank/stage-local scalar metadata 放进 scaffold broadcast。
- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_text_stage.py`
  - `test/test_hybrid_direct_loader.py`
  - `test/test_model_weight_loader.py`
- 行为变化：
  - leader 发送 scaffold 前移除 `stage_idx` / `start_idx` / `end_idx` / `save_dtype` / `hidden_size` / `batch_size`。
  - follower/leader 收到 scaffold 后从 `StageSpec` 和本地 compute dtype 恢复 `stage_idx` / `start_idx` / `end_idx` / `save_dtype`。
  - `materialize_text_stage_state()` 会从本地 text config 恢复 `hidden_size`，并从已有 runtime tensor shape 恢复 `batch_size`。
  - `compute_dtype_arg=auto` 且 scaffold 不带 `save_dtype` 时，会按本地可推导来源解析：manifest/stage save_dtype、scaffold floating tensor dtype、最后才读本地模型权重 dtype reference。
- 本轮 before/after 记录：
  - before: object meta 153 bytes，0 tensors，0 tensor bytes，meta keys=`batch_size` / `end_idx` / `hidden_size` / `layers` / `save_dtype` / `stage_idx` / `start_idx`
  - after: object meta 94 bytes，0 tensors，0 tensor bytes，meta keys=`layers` / `rank_local_fields_local_rebuild`
- 本地验证：
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
- 注意：
  - 这一步主要减少 `stage_scaffold_meta` object bytes，不改变 tensor payload keys。
  - 后续已完成 `prefill_attention_mask_2d` / top-level derived tensor 本地重建。
  - 完整 3-rank HYBRID `--pp 2 --tp-degrees 2 1` 真实 profile 后续已由 `baseline_runs/20260429-longterm-profile/` 冻结。

step 13 scaffold broadcast 减量第三轮，已完成 prefill runtime tensor 本地重建：

- 目标：HYBRID multimodal scaffold 不再重复发送 startup shared 已能本地重建的 top-level prefill runtime tensor。
- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_text.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_text_stage.py`
  - `test/test_hybrid_direct_loader.py`
  - `test/test_model_weight_loader.py`
- 行为变化：
  - leader 发送 multimodal runtime-only scaffold 前移除 `prefill_attention_mask_2d` / `prefill_attention_mask` / `prefill_position_ids` / `prefill_cos` / `prefill_sin`。
  - `_restore_text_prompt_stage_state()` 能从 `_mm_startup_shared.attention_mask_2d` 或 shared `input_ids` 恢复 `prefill_attention_mask_2d`。
  - `materialize_text_stage_state()` 会用 `_mm_startup_shared` + local `stage_input` 调 `build_mm_stage_state()`，本地重建 dense attention mask、position ids、RoPE cos/sin。
  - 纯 TP local scaffold 没有走 HYBRID broadcast strip，当前不改变纯 TP multimodal 的本地构建语义。
- 本轮 before/after 记录：
  - before: 7 tensors，308 tensor bytes，object meta 485 bytes，keys=`scaffold.stage_input` / `scaffold.rope_deltas` / `scaffold.prefill_attention_mask_2d` / `scaffold.prefill_attention_mask` / `scaffold.prefill_position_ids` / `scaffold.prefill_cos` / `scaffold.prefill_sin`
  - after: 2 tensors，56 tensor bytes，object meta 251 bytes，keys=`scaffold.stage_input` / `scaffold.rope_deltas`
- 本地验证：
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
- 注意：
  - 完整 3-rank HYBRID `--pp 2 --tp-degrees 2 1` 真实 profile 后续已由 `baseline_runs/20260429-longterm-profile/` 冻结。

step 13 scaffold broadcast 减量第四轮，已完成中期收口：

- 目标：HYBRID multimodal scaffold 不再广播只用于 frontend 来源记录的 `num_frames` / `frame_paths`。
- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_text.py`
  - `test/test_hybrid_direct_loader.py`
  - `test/test_model_weight_loader.py`
  - `qwen3vl_tp_runtime/ROADMAP.md`
  - `qwen3vl_tp_runtime/BASELINE.md`
  - `qwen3vl_tp_runtime/SESSION_HANDOFF.md`
- 行为变化：
  - leader 发送 multimodal runtime-only scaffold 前移除 `num_frames` / `frame_paths`。
  - follower/materialize 侧从 `_mm_num_frames` / `_mm_frame_paths` 恢复兼容字段。
  - 这一步只恢复 metadata，不重新读取图片/视频，也不重新激活视觉 frontend。
  - `text_scaffold` / `stage_scaffold` 当前只保留为 transport/profile label 差异；实际 codec 统一走 generic dict+tensor scaffold transport。
- 本轮 before/after 记录：
  - before: object meta 347 bytes，2 tensors，56 tensor bytes，meta keys=`frame_paths` / `layers` / `mm_prefill_runtime_tensors_local_rebuild` / `modality` / `num_frames` / `rank_local_fields_local_rebuild` / `rope_deltas` / `runtime_only_generate` / `stage_input`
  - after: object meta 324 bytes，2 tensors，56 tensor bytes，meta keys=`layers` / `mm_frontend_metadata_local_rebuild` / `mm_prefill_runtime_tensors_local_rebuild` / `modality` / `rank_local_fields_local_rebuild` / `rope_deltas` / `runtime_only_generate` / `stage_input`
  - tensor payload keys 不变：`scaffold.stage_input` / `scaffold.rope_deltas`
- 本地验证：
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
- 当前结论：
  - step 13 中期阶段已经收口；后续长期目标也已继续推进到 `runtime_inputs` broadcast、正式 schema、纯 TP multimodal input-owner 和真实 Jetson profile 冻结。

step 13 长期目标第一轮，已把 runtime-only 主路径切到 `runtime_inputs` broadcast：

- 目标：HYBRID `runtime-only generate` stage-group broadcast 不再发送 scaffold-like StageState root。
- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
  - `qwen3vl_tp_runtime/hexgen_core/distributed.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_builder.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/runtime_text_stage.py`
  - `test/test_hybrid_direct_loader.py`
  - `qwen3vl_tp_runtime/ROADMAP.md`
  - `qwen3vl_tp_runtime/BASELINE.md`
  - `qwen3vl_tp_runtime/SESSION_HANDOFF.md`
- 行为变化：
  - `include_runtime_reference=false` 时，HYBRID stage-group broadcast 使用 `runtime_inputs` root。
  - 新协议：`hybrid_runtime_inputs_v1`。
  - follower 收到 runtime input dict 后，用本地 `StageSpec` / runtime config / startup shared metadata 恢复最小 StageState scaffold，再 `materialize_text_stage_state()` 加载本 rank 权重 shard。
  - `distributed._classify_transport_kind()` 把 `runtime_inputs_*` label 仍归入 `scaffold` kind，保证 perf 表继续能汇总这类启动同步 bytes。
  - reference/debug/file-backed path 仍保留旧 `text_scaffold` / `stage_scaffold` transport。
- 本轮 before/after 记录：
  - multimodal before: root=`scaffold`，object meta 433 bytes，2 tensors / 56 bytes，keys=`scaffold.stage_input` / `scaffold.rope_deltas`
  - multimodal after: root=`runtime_inputs`，object meta 270 bytes，2 tensors / 56 bytes，keys=`runtime_inputs.stage_input` / `runtime_inputs.rope_deltas`
  - text before: root=`scaffold`，object meta 271 bytes，0 tensors / 0 bytes
  - text after: root=`runtime_inputs`，object meta 190 bytes，0 tensors / 0 bytes
- 本地验证：
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_summary.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
  - `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`
- 后续进展：
  - 长期第二轮已让 leader 不再先构造 scaffold，而是直接从 text prompt metadata / multimodal startup contract 组出 `runtime_inputs`。

step 13 长期目标第二轮，已让 leader 直接构造 runtime input dict：

- 目标：HYBRID `runtime-only generate` stage leader 不再先调用 `build_direct_stage_state(..., include_text_weights=False)` 构造 scaffold-like StageState，再从里面抽 `runtime_inputs`。
- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
  - `test/test_hybrid_direct_loader.py`
  - `qwen3vl_tp_runtime/ROADMAP.md`
  - `qwen3vl_tp_runtime/BASELINE.md`
  - `qwen3vl_tp_runtime/SESSION_HANDOFF.md`
- 行为变化：
  - text runtime input 直接来自 `_runtime_only_input_ids` / `_runtime_only_attention_mask`。
  - multimodal runtime input 直接来自 `_mm_startup_shared`、`_mm_startup_stage_handoffs[stage_idx].stage_input` 和可选 `_mm_startup_stage_visuals[stage_idx]`。
  - `include_runtime_reference=false` 的 stage-group broadcast 分支里，leader 不再调用 `build_direct_stage_state`。
  - follower 收到 `runtime_inputs` 后只恢复 materialize 所需的最小 StageState scaffold；rank-local 权重仍由本 rank 从 `model_path` materialize。
  - reference/debug/file-backed path 仍走旧 `text_scaffold` / `stage_scaffold` 兼容路径。
- 本轮 before/after 记录：
  - text before: object meta 190 bytes，0 tensors / 0 bytes，keys=空
  - text after: object meta 249 bytes，1 tensor / 24 bytes，keys=`runtime_inputs.input_ids`
  - multimodal before: object meta 270 bytes，2 tensors / 56 bytes，keys=`runtime_inputs.stage_input` / `runtime_inputs.rope_deltas`
  - multimodal after: object meta 441 bytes，4 tensors / 104 bytes，keys=`runtime_inputs.shared.input_ids` / `runtime_inputs.shared.attention_mask_2d` / `runtime_inputs.shared.rope_deltas` / `runtime_inputs.stage_handoff.stage_input`
- 注意：
  - bytes 增加是预期结果，因为新的 payload 是自洽的 request/runtime input dict，不再依赖 scaffold 派生字段或隐式本地 runtime_config。
- 本地验证：
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_distributed_serialization.py`
  - `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`
- 已同步：
  - `bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed`
  - `bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed`

step 13 长期目标第三轮，已把 runtime input 协议固化成 schema：

- 目标：把 `hybrid_runtime_inputs_v1` 从 HYBRID helper 内部约定提升为正式可验证 schema。
- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/schema.py`
  - `qwen3vl_tp_runtime/hexgen_core/__init__.py`
  - `qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
  - `qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`
  - `test/test_hybrid_runtime_input_schema.py`
  - `test/test_hybrid_direct_loader.py`
  - `test/test_compat_package_exports.py`
  - `qwen3vl_tp_runtime/ROADMAP.md`
  - `qwen3vl_tp_runtime/BASELINE.md`
  - `qwen3vl_tp_runtime/SESSION_HANDOFF.md`
- 新增 schema：
  - `HYBRID_RUNTIME_INPUT_PROTOCOL = "hybrid_runtime_inputs_v1"`
  - `HybridRuntimeInputSchema`
- schema 规则：
  - text top-level 只允许 `protocol` / `modality` / `mode` / `runtime_only_generate` / `input_ids` / `attention_mask_2d` / `runtime_only_prompt_local_rebuild`。
  - multimodal top-level 只允许 `protocol` / `modality` / `mode` / `runtime_only_generate` / `shared` / `stage_handoff` / `stage_visuals`。
  - multimodal `shared` 必需 `input_ids` / `attention_mask_2d` / `rope_deltas`，可选 `position_ids` / `mm_token_type_ids` / `image_grid_thw` / `video_grid_thw`。
  - multimodal `stage_handoff` 只允许 `stage_input`。
  - multimodal `stage_visuals` 只允许 `visual_pos_masks` / `deepstack_by_layer`。
  - 禁止 weights/bias、layers、StageState/scaffold rank-local 字段、frontend paths、derived attention tensors、root/replay/full payload 进入 runtime input broadcast。
- enforcement：
  - leader build runtime input 时校验。
  - broadcast restore 后、resolve compute dtype 前校验。
  - `_restore_stage_scaffold_from_runtime_inputs()` 内再次校验。
- payload 影响：
  - 本步不改变 tensor keys/count/bytes，只增加协议 guard。
  - text 仍为第二刀 after：`runtime_inputs.input_ids`，1 tensor / 24 bytes。
  - multimodal 仍为第二刀 after：4 tensors / 104 bytes。
- 本地验证：
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_runtime_input_schema.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`

step 13 长期目标第四轮，已完成纯 TP multimodal input-owner 优化：

- 目标：`backend=tp` multimodal generate 不再让每个 TP rank 都 active multimodal frontend / file-backed prefill reference。
- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py`
  - `test/test_tensor_parallel_direct.py`
  - `qwen3vl_tp_runtime/ROADMAP.md`
  - `qwen3vl_tp_runtime/BASELINE.md`
  - `qwen3vl_tp_runtime/SESSION_HANDOFF.md`
- 行为变化：
  - rank0/input owner 用 `DirectStageStateBuilder(... include_text_weights=False, mm_activate_frontend=True)` 准备 thin multimodal startup contract。
  - contract 通过 TP group broadcast 同步，label 是 `tp_multimodal_startup_contract_meta/tensors stage_idx=0`。
  - 所有 TP rank，包括 rank0，都会 seed 本地 startup contract，然后最终以 `mm_activate_frontend=False` 构建 consume-only scaffold。
  - 每个 TP rank 仍本地 materialize 自己的 decoder projection / MLP projection shard；权重不通过 startup contract 或 broadcast 传输。
- 本轮 before/after 记录：
  - before：没有 input-owner startup broadcast，0 tensors / 0 tensor bytes；代价是每个 TP rank 各自 active frontend。
  - after object meta：82 bytes，meta keys=`frame_paths` / `num_frames`。
  - after tensor count / bytes：9 tensors / 239 bytes。
  - after tensor keys：`shared.input_ids` / `shared.attention_mask_2d` / `shared.position_ids` / `shared.rope_deltas` / `shared.mm_token_type_ids` / `shared.image_grid_thw` / `shared.video_grid_thw` / `stage_handoffs.0.stage_input` / `stage_visuals.0.visual_pos_masks`。
  - after 不包含：weights、layers、`stage_output`、dense `shared.attention_mask`、`shared.cos`、`shared.sin`、root/full/replay payload。
- 已运行：
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
  - `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`
- 真实 Jetson profile 已确认：
  - `tp-mm-generate` rank0 有 `prepare multimodal input-owner startup contract`，rank0/rank1 最终 direct builder 都是 `multimodal_frontend_mode=consume-only startup_contract_ready=True`，生成 token/text 不变。

step 13 长期目标第五步，已重跑真实 Jetson profile 并冻结 baseline：

- 输出目录：
  - `baseline_runs/20260429-longterm-profile/`
- 拓扑：
  - `tp-mm-generate`: Jetson2 rank0，Jetson3 rank1，`--backend tp --tp 2`。
  - `hybrid-text-generate`: Jetson2 rank0/rank1，Jetson3 rank2，`--backend hybrid --pp 2 --tp-degrees 2 1`。
  - `hybrid-mm-generate`: Jetson2 rank0/rank1，Jetson3 rank2，`--backend hybrid --pp 2 --tp-degrees 2 1`。
  - `MASTER_ADDR=10.126.126.3`。
- correctness：
  - `tp-mm-generate`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text=`视频中，一名`。
  - `hybrid-text-generate`: PASS，generated ids `[104455, 9909, 9286, 16488]`，text=`人工智能（Artificial`。
  - `hybrid-mm-generate`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text=`视频中，一名`。
- 关键 payload：
  - `tp-mm-generate`: `tp_multimodal_startup_contract_meta` 422 object bytes；`tp_multimodal_startup_contract_tensors` 12 tensors / 12,093,371 tensor bytes；无 scaffold broadcast。
  - `hybrid-text-generate`: stage0 `runtime_inputs_meta` 249 object bytes；`runtime_inputs_tensors` 1 tensor / 128 tensor bytes，key=`runtime_inputs.input_ids`。
  - `hybrid-mm-generate`: stage1 startup contract 422 object bytes + 9 tensors / 3,245,384 tensor bytes；stage0 runtime input broadcast 938 object bytes + 11 tensors / 12,093,371 tensor bytes。
- 关键 weight bytes：
  - `tp-mm-generate` rank0/rank1：`5,189,532,672`，`tp_weight_sharded=true`。
  - `hybrid-text-generate` stage0 rank0/rank1：`2,594,763,776`，`tp_stage_loaded_weight_tensor_bytes_equal=true`；stage1 rank2：`4,411,426,816`。
  - `hybrid-mm-generate` stage0 rank0/rank1：`2,594,763,776`，`tp_stage_loaded_weight_tensor_bytes_equal=true`；stage1 rank2：`4,411,426,816`。
- 生成文件：
  - `baseline_runs/20260429-longterm-profile/check-longterm-baseline.txt`
  - `baseline_runs/20260429-longterm-profile/runtime-perf-records.json`
  - `baseline_runs/20260429-longterm-profile/runtime-perf-table.md`
  - `baseline_runs/20260429-longterm-profile/README.md`
- 运行命令要点：
  - `tp-mm-generate` 用 `run-tp-mm-generate.sh`，`MASTER_PORT=29636`。
  - `hybrid-text-generate` 用手动 `RANK/WORLD_SIZE/LOCAL_RANK` 启动，因为 Jetson2 承担 rank0/rank1，`MASTER_PORT=29637`。
  - `hybrid-mm-generate` 用 `run-hybrid-mm-generate.sh`，Jetson2 启动 node-rank 0/1，Jetson3 启动 node-rank 2，`MASTER_PORT=29638`。

step 11 已运行并通过的本地检查：

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python -m py_compile \
  qwen3vl_tp_runtime/hexgen_core/schema.py \
  qwen3vl_tp_runtime/hexgen_core/distributed.py \
  qwen3vl_tp_runtime/hexgen_core/transport.py \
  qwen3vl_tp_runtime/scripts/runtime_summary.py \
  qwen3vl_tp_runtime/scripts/collect_runtime_perf.py \
  test/test_runtime_summary.py \
  test/test_collect_runtime_perf.py

PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_summary.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_collect_runtime_perf.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python qwen3vl_tp_runtime/scripts/collect_runtime_perf.py \
  --baseline-dir baseline_runs/20260428 \
  --output-json baseline_runs/20260428/runtime-perf-records.json \
  --output-md baseline_runs/20260428/runtime-perf-table.md

bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --include-weight-loader --skip-baseline-checks
```

step 14 TP collective profiling 第一刀，已完成“补观测，不改语义”：

- vLLM 对照：
  - vLLM 把 TP collective 收口到 `tensor_model_parallel_all_reduce` / `tensor_model_parallel_all_gather` 等 wrapper。
  - `ColumnParallelLinear` 只在 `gather_output=true` 且 `tp_size>1` 时 all-gather。
  - `RowParallelLinear` 只在 `reduce_results=true` 且 `tp_size>1` 时 all-reduce。
  - 我们当前是 CPU/gloo correctness-first runtime，不照搬 CUDA/NCCL/custom-op communicator，只照搬 wrapper + semantic attribution + profiler-first 思路。
- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/distributed.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/execution/attention.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/execution/mlp.py`
  - `qwen3vl_tp_runtime/models/qwen3vl/execution/stages.py`
  - `qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py`
  - `qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
  - `qwen3vl_tp_runtime/scripts/collect_runtime_perf.py`
  - `test/test_runtime_summary.py`
  - `test/test_collect_runtime_perf.py`
- 行为变化：
  - `all_reduce_cpu` / `all_gather_cpu` / `broadcast_cpu` 支持 `profile_context`。
  - TP collective event 现在会带 `phase` / `layer_idx` / `module` / `reason`。
  - attention sharded output reduce 标为 `module=attention reason=row_parallel_reduce`。
  - MLP sharded down projection reduce 标为 `module=mlp reason=row_parallel_reduce`。
  - non-sharded fallback all-gather 标为 `reason=column_parallel_gather`，leader broadcast 标为 `reason=full_weight_leader_broadcast`。
  - runtime stage input broadcast 标为 `module=runtime_input reason=stage_input_broadcast`。
  - `collect_runtime_perf.py` 新增 `payload.tp_collective_breakdown`，按 `phase/module/reason/operation` 聚合 event count、seconds、bytes。
- before/after：
  - 本轮只增加 event metadata，不改变 generated 逻辑、不改变 tensor payload 计算方式。
  - TP collective payload key 仍是 single tensor profile 的 `tensor`。
  - true Jetson generated ids/text 已由 `baseline_runs/20260429-step14-profile/` 确认不变。
- 真实 Jetson profile：
  - 输出目录：`baseline_runs/20260429-step14-profile/`。
  - `tp-text-generate`: PASS，generated ids `[104455, 9909, 9286, 16488]`，text `人工智能（Artificial`。
  - `tp-mm-generate`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`。
  - `hybrid-mm-generate`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`。
  - `runtime-perf-records.json` / `runtime-perf-table.md` 已生成。
- 关键 breakdown：
  - `tp-text-generate`: prefill attention/MLP row all-reduce 各 36 次，各约 `5.63 MiB`；decode attention/MLP 各 108 次但 bytes 小。
  - `tp-mm-generate`: prefill attention/MLP row all-reduce 各 36 次，各约 `220.43 MiB`，各约 `21s`。
  - `hybrid-mm-generate` stage0：prefill attention/MLP row all-reduce 各 18 次，各约 `110.21 MiB`。
  - `hybrid-mm-generate` stage1：TP degree 是 1，但当前 generic path 仍记录 prefill MLP all-gather 约 `418.82 MiB`，是下一刀最明确的低风险候选。
- step 14 第二刀 `tp_degree=1` collective bypass，已完成：
  - 对照 vLLM `world_size == 1` wrapper bypass。
  - `all_reduce_cpu` / `all_gather_cpu` / `broadcast_cpu` 在 group world size 为 1 时直接返回本地 tensor，不调用 `dist.*`，不记录 TP collective event。
  - 新增 `test/test_distributed_single_rank_bypass.py`。
  - 真实 Jetson 复测目录：`baseline_runs/20260429-step14-tp1-bypass/`。
  - `hybrid-mm-generate --pp 2 --tp-degrees 2 1`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`。
  - rank2 TP collective bytes：before `648.46 MiB` -> after `0 B`。
  - rank2 TP collective event：before `220` 个、payload key `tensor` -> after `0` 个、payload keys 为空。
  - rank0/rank1 TP collective bytes 仍为 `227.64 MiB`，说明真实 TP group 路径未被旁路。
- 本地验证：
  - `python -m py_compile distributed.py attention.py mlp.py stages.py tensor_parallel.py hybrid_parallel.py collect_runtime_perf.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_summary.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_collect_runtime_perf.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_distributed_single_rank_bypass.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
  - `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_distributed_serialization.py`
  - `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`
- 下一步：
  - 已完成 vLLM ColumnParallel / RowParallel 对照检查：
    - pure TP / HYBRID stage0 的 sharded 主路径没有 `column_parallel_gather`。
    - 当前大头是每层两个真实 RowParallel all-reduce：attention `o_proj` 后一次、MLP `down_proj` 后一次。
    - `tp-mm-generate` prefill collective shape 是 `(1, 627, 2560)`，默认 `comm_dtype=torch.float32`，单次约 `6.12 MiB`，每层 attention/MLP 各 36 次。
    - attention reduce 和 MLP reduce 不能低风险合并，因为 MLP 输入依赖 attention reduce 后的 residual。
    - decode 小 collective 数量多但 bytes 小，优先级低于 prefill 大 all-reduce。
  - 下一刀先补 `all_reduce_cpu` / `broadcast_cpu` 子阶段 profiling：device->CPU、gloo collective、CPU->device、target dtype、comm dtype、shape。
  - 然后做 opt-in `--comm-dtype bfloat16/float16` profile 实验，不改默认值。
  - pure TP runtime input broadcast 可作为单独低风险候选：如果每个 TP rank 已有本地 `stage_input` 或可本地构建 embeddings，就避免 rank0 广播 dense `stage_input`。
  - 暂缓 reduce-scatter/sequence-parallel/NCCL/custom all-reduce，这些会改变执行结构。

step 14 TP collective substage profiling 收口，已完成：

- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/distributed.py`
  - `qwen3vl_tp_runtime/scripts/collect_runtime_perf.py`
  - `test/test_collect_runtime_perf.py`
  - `test/test_distributed_single_rank_bypass.py`
- 行为变化：
  - `all_reduce_cpu` / `all_gather_cpu` / `broadcast_cpu` 的 TP collective event 增加 `payload_prepare_seconds`、`device_to_cpu_seconds`、`gloo_collective_seconds`、`cpu_to_device_seconds`。
  - event 同时记录 `source_dtype`、`reference_dtype`、`comm_dtype`、`target_dtype`、`source_device`、`target_device`、`world_size`。
  - `collect_runtime_perf.py` 输出 `payload.tp_collective_substage_seconds`，并在 `tp_collective_breakdown` 中聚合子阶段时间。
- 真实 Jetson profile：
  - 输出目录：`baseline_runs/20260430-step14-substage-profile/`。
  - `tp-mm-generate`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`。
  - `hybrid-mm-generate`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`。
  - `runtime-perf-records.json` / `runtime-perf-table.md` 已生成。
- 关键结论：
  - `tp-mm-generate` 的 `44-45s` TP collective 时间主要卡在 gloo collective，不是 device/CPU copy。
  - rank0：total `45.646768s`，gloo `43.010443s` (`94.2%`)，device->CPU `2.108236s` (`4.6%`)，CPU->device `0.521141s` (`1.1%`)。
  - rank1：total `44.913286s`，gloo `42.215497s` (`94.0%`)，device->CPU `2.083570s` (`4.6%`)，CPU->device `0.606738s` (`1.4%`)。
  - representative prefill row-reduce shape `(1, 627, 2560)`，source/target dtype `torch.bfloat16`，comm dtype `torch.float32`，单 event 约 `6.12 MiB`。
  - `hybrid-mm-generate` stage0 同机 TP 的 gloo 占比明显低；rank2 `tp_degree=1` 保持 `0 B / 0s`。
- step 14 可以结束：
  - 已有 semantic breakdown。
  - 已清掉 `tp_degree=1` 伪 collective。
  - 已确认 pure TP 大头是真实 RowParallel all-reduce。
  - 已确认跨 Jetson pure TP 主要瓶颈是 gloo。
- 下一阶段候选：
  - opt-in `--comm-dtype bfloat16/float16` profile 实验。
  - pure TP runtime input broadcast 减量。
  - 更大结构项如 reduce-scatter/sequence-parallel/NCCL/custom all-reduce 暂缓。

opt-in `--comm-dtype bfloat16/float16` profile 实验，已完成：

- 本轮只做 opt-in 实验，不改默认值。
- 输出目录：`baseline_runs/20260430-comm-dtype-profile/`。
- 复测 case：
  - `tp-mm-generate-comm-bfloat16`: `run-tp-mm-generate.sh --comm-dtype bfloat16`
  - `tp-mm-generate-comm-float16`: `run-tp-mm-generate.sh --comm-dtype float16`
- checker：
  - `check-tp-mm-generate-comm-bfloat16.txt`: PASS。
  - `check-tp-mm-generate-comm-float16.txt`: PASS。
- 对照默认 float32 baseline `baseline_runs/20260430-step14-substage-profile/`：
  - default rank0/rank1：generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`，TP collective bytes `449.12 MiB`，TP collective seconds `45.646768s` / `44.913286s`。
  - `--comm-dtype bfloat16` rank0/rank1：generated ids/text 不变，TP collective bytes `224.56 MiB`，TP collective seconds `24.773918s` / `24.117012s`。
  - `--comm-dtype float16` rank0/rank1：generated ids/text 不变，TP collective bytes `224.56 MiB`，TP collective seconds `24.959947s` / `24.327708s`。
- 结论：
  - 两种 opt-in dtype 都把 TP collective bytes 减半。
  - TP collective seconds 从 `44-45s` 降到 `24-25s`。
  - 端到端 runtime 从约 `74.7s` 降到约 `53.6-54.0s`。
  - 本轮 `bfloat16` 略快，先作为 Jetson 推荐 opt-in 候选；默认值暂不改，改默认前要补 `tp-text-generate`、`hybrid-mm-generate` 和更长 decode 回归。

pure TP runtime input broadcast 减量，已完成：

- 已改代码：
  - `qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py`
  - `qwen3vl_tp_runtime/scripts/runtime_summary.py`
  - `test/test_tensor_parallel_direct.py`
- 行为变化：
  - pure TP generate prefill 优先本地构建 runtime input：
    - 有 `input_ids + embed_tokens_weight` 时本地构建 embeddings。
    - 否则使用本地 `stage_input/layer_input`。
    - 本地条件都缺失时才回退到 rank0 `broadcast_cpu`。
  - pure TP generate decode 保留已有 token id broadcast，然后每个 TP rank 用本地 `embed_tokens_weight` 构建 one-token embedding。
  - runtime summary phase stats 增加 `runtime_input_source` 和 `runtime_input_broadcast_skipped`。
- 真实 Jetson profile：
  - 输出目录：`baseline_runs/20260430-runtime-input-local-profile/`。
  - `tp-mm-generate-runtime-input-local`: PASS。
  - generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`。
- before/after：
  - runtime input TP events / rank：`4 -> 0`。
  - runtime input TP bytes / rank：`6,451,200 -> 0`。
  - rank0 runtime input seconds：`1.035093s -> 0s`。
  - rank1 runtime input seconds：`0.563669s -> 0s`。
  - rank0 total TP collective bytes：`449.12 MiB -> 442.97 MiB`。
  - rank1 total TP collective bytes：`449.12 MiB -> 442.97 MiB`。
  - startup contract bytes 仍为每 rank `11.53 MiB`，这是 input-owner startup contract，不属于本轮 generate-time broadcast。
- summary 证据：
  - prefill：`runtime_input_source=local_stage_input`，`runtime_input_broadcast_skipped=true`。
  - decode steps：`runtime_input_source=local_embeddings`，`runtime_input_broadcast_skipped=true`。
- 本地验证：
  - `python -m py_compile qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py qwen3vl_tp_runtime/scripts/runtime_summary.py test/test_tensor_parallel_direct.py`
  - `PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
  - `PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_collect_runtime_perf.py`
  - `PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_summary.py`
  - `PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_distributed_single_rank_bypass.py`
  - `PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
  - `PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_cli_modes.py`

bfloat16 默认值候选回归，已完成：

- 本轮没有改默认值，只验证 `--comm-dtype bfloat16` 是否足够稳定作为默认候选。
- 输出目录：`baseline_runs/20260430-bfloat16-default-candidate/`。
- 覆盖 case：
  - `tp-text-generate-bfloat16`: PASS，generated ids `[104455, 9909, 9286, 16488]`，text `人工智能（Artificial`。
  - `tp-mm-generate-bfloat16-wide`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`。
  - `hybrid-mm-generate-bfloat16-wide`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`。
  - `tp-mm-generate-long-default`: PASS，`MAX_NEW_TOKENS=16`。
  - `tp-mm-generate-long-bfloat16`: PASS，`MAX_NEW_TOKENS=16`。
- 长 decode default vs bfloat16：
  - generated ids 完全一致：`[87140, 15946, 3837, 101177, 105611, 99194, 38035, 113727, 33108, 104362, 38035, 113233, 9370, 104253, 104224, 46944]`。
  - generated text 完全一致：`视频中，一名穿着深色衬衫和浅色裤子的男子站在一个`。
- perf 观察：
  - `tp-text`: TP collective bytes `13.54 MiB -> 6.68 MiB`。
  - current `tp-mm`: TP collective bytes `442.97 MiB -> 221.48 MiB`。
  - `hybrid-mm` stage0: TP collective bytes `227.64 MiB -> 113.82 MiB`。
  - `tp-mm` 16-token long decode: TP collective bytes `451.41 MiB -> 225.70 MiB`，runtime about `84.2s -> 62.9s`。
- 结论：
  - `bfloat16` 可以进入默认值候选状态。
  - 真正改默认值应作为单独下一刀，只改 CLI/config 默认，不改 collective 语义；改完后复跑同一矩阵。

bfloat16 默认值落地回归，已完成：

- 已把 `qwen3vl_tp_runtime/scripts/runtime.py` 里的 `--comm-dtype` 默认值从 `float32` 改为 `bfloat16`。
- 只改 CLI/config 默认值，不改 `all_reduce_cpu` / `all_gather_cpu` / `broadcast_cpu` 语义。
- 输出目录：`baseline_runs/20260430-bfloat16-default/`。
- 所有 case 都不显式传 `--comm-dtype`。
- 覆盖 case：
  - `tp-text-generate-default`: PASS，generated ids `[104455, 9909, 9286, 16488]`，text `人工智能（Artificial`，TP collective event 显示 `comm_dtype=torch.bfloat16`。
  - `tp-mm-generate-default`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`，TP collective bytes `221.48 MiB`。
  - `hybrid-mm-generate-default`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`，stage0 TP collective bytes `113.82 MiB`，stage1 `tp_degree=1` 仍为 `0 B`。
  - `tp-mm-generate-long-default-bfloat16`: PASS，`MAX_NEW_TOKENS=16`，TP collective bytes `225.70 MiB`。
- checker 输出：`baseline_runs/20260430-bfloat16-default/check-*.txt`，全部 PASS。
- perf records/table：`baseline_runs/20260430-bfloat16-default/runtime-perf-records.json`、`runtime-perf-table.md`。
- `ROADMAP.md` step 14 已标记完成，下一步转入 step 15 `Multimodal payload 减量`。

## 继续工作时的默认流程

如果用户让你继续清理或改代码，建议流程：

1. 先用 `rg` 确认调用关系。
2. 只动当前任务相关文件，不顺手大改。
3. 若涉及 PP/TP/HYBRID 公共结构，先确认依赖方向：
   - PP base
   - TP base
   - HYBRID composition
4. 用 `apply_patch` 修改。
5. 跑最小相关测试。
6. 如果改到分布式主路径、导出面或 runtime CLI，跑：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_pipeline_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_cli_modes.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py
```

7. 如果改到 runtime builder / weights，额外跑：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py
```

8. 测试通过后同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```

## 需要避免的事情

- 不要把纯 TP 重新接回 `TextHybridRunner`。
- 不要让 `tensor_parallel.py` import `hybrid_parallel.py`。
- 不要把主路径重新叫 `bundle`。
- 不要为了少量重复，让 PP 依赖 TP，除非用户明确改变架构要求。
- 不要删除 replay/debug compatibility，除非先确认没有测试或外部入口依赖。
- 不要把 `--manifest-path` 重新变成默认主路径。
- 不要在未验证情况下改 multimodal startup contract。
- 不要让 non-stage0 本地跑视觉 frontend。
- 不要把 full/root payload 放进 startup transport。
- 不要把 embedding/lm_head 已复制误说成已经 vocab parallel。
- 不要用 destructive git 命令。

## 当前可以对老师汇报的一句话

这个项目已经从“Qwen3-VL replay 验证器”推进到“从 `model_path` 直接构建 `StageState` 的 correctness-first 分布式推理 runtime 原型”：纯 PP、纯 TP、PP+TP HYBRID 三条主路径都已建立，text generate 和 multimodal generate 的关键 smoke 已通过；TP/PP 是并列基础后端，HYBRID 是组合层；每个 rank/stage 已能只 materialize 自己需要的 StageState 和 text decoder shard，旧 replay bundle 只保留在 debug/capture/file-backed reference 路径中。当前 HYBRID runtime input schema、纯 TP multimodal input-owner、startup / memory / transport payload profiling 都已落地并冻结真实 Jetson profile，剩余重点是按 ROADMAP 从 TP collective profiling 开始做性能优化，以及后续推进 embedding/lm_head vocab parallelism。

## 最后状态备注

- 本文件会随着每轮任务更新；当前工作树状态以实际 `git status --short` 为准。
- 如果新对话接手后第一件事是继续写代码，建议先跑：

```bash
git status --short
```

确认只有预期文件变化。
