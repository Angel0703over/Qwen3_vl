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
最近一次重要重构：backend=tp 已有独立 TensorParallelManifest / build_direct_tp_manifest / TensorParallelRunner，不再借用 TextHybridManifest 或 TextHybridRunner；hybrid_parallel 可以调用 tensor_parallel 的公共 helper，因为 hybrid 依赖 TP，但 TP 不能反向依赖 hybrid。
请继续用中文和我配合，优先读代码、直接修改、跑测试、同步到两台 Jetson。
```

## 当前日期和环境

- 当前日期：2026-04-27
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
  - 旧 debug/replay TP 路径保留在 `DEBUG_REPLAY_EXPORTS`。
  - HYBRID 现在复用这里的一些 helper。

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
  - 这里仍有一些内部变量或函数名带 `bundle`，大多是 file-backed reference / layer weight compatibility。后续可继续清理，但不要误删仍被调用的 reference 逻辑。

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
- 但 `bundle` 可以继续出现在 replay/debug/capture/file-backed reference 或底层 layer weight 兼容逻辑里。
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

已删除 HYBRID 本地重复定义并改为复用 TP：

- `_build_generate_phase_state`
- `_strip_runtime_layer_cache`
- `_build_generate_cache_map`
- `_is_runtime_only_generate_state`
- `_infer_runtime_tensor_device`
- `_infer_runtime_tensor_dtype`
- `_infer_runtime_token_dtype`
- `_build_runtime_only_stage_input_template`
- `_token_tensor_to_list`
- `_broadcast_token_id`

仍保留在 HYBRID：

- `_build_runtime_only_text_generate_phase_state`

原因：

- HYBRID 需要根据当前 stage 是否 last stage 判断 `text_prefill_last` / `text` / `text_decode_last` / `text_decode`。
- 纯 TP 永远是单 stage / last stage，语义不完全相同。

## 当前测试记录

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

### TP 里仍有 debug/replay 兼容路径

文件：`hexgen_core/modules/tensor_parallel.py`

仍保留：

- `load_text_stage_bundle`
- `run_text_tensor_parallel_stage`
- `TextTensorParallelRunner`
- `run_text_tensor_parallel_rank`
- `DEBUG_REPLAY_EXPORTS`

这些不是主路径，是旧 capture/replay 调试入口。当前没有进入 `__all__` 主导出面。

后续可选清理：

- 移到 `tp_debug.py` 或单独 replay/debug 文件。
- 但移动前要更新兼容测试和任何外部调用。

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

- `_build_generate_phase_state`
- `_strip_runtime_layer_cache`
- `_build_generate_cache_map`
- `_is_runtime_only_generate_state`
- `_broadcast_token_id`

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

### 1. debug/replay TP 路径搬家

当前 `tensor_parallel.py` 既有主路径 TP，也有旧 debug/replay TP。

可选方案：

- 把 `TextTensorParallelRunner`、`run_text_tensor_parallel_rank`、`run_text_tensor_parallel_stage` 移到 `tp_debug.py` 或 `tensor_parallel_replay.py`。
- 保持 `tensor_parallel.py.__all__` 主路径纯净。

### 2. TensorParallelManifest 简化

当前为了 summary/兼容保留了一些固定 layout 字段。

可选方案：

- 简化为只保存 `tp_degree`、`stage_ranges`、`stages`、`runtime_config`。
- 在 summary 时动态生成固定 layout 字段。

但这会影响测试和兼容 loader，不是优先任务。

### 3. 统一或明确 private helper import

`hybrid_parallel.py` 当前从 `tensor_parallel.py` import 一些 `_` 开头 helper。

这是按用户偏好做的：不新建公共文件，让 HYBRID 直接复用 TP。

但从 Python 风格看，import 私有函数不太漂亮。

后续可选：

- 把这些 helper 去掉 `_` 前缀并放入 `DIRECT_RUNTIME_EXPORTS`？
- 或保留私有 import，因为它们只是模块内部组合层使用，不对包外公开。

当前不建议急改，先保持测试稳定。

### 4. embedding / lm_head vocab parallelism

当前仍复制：

- `embed_tokens_weight`
- `lm_head_weight`

这是已知后续优化，不属于当前里程碑。

### 5. 更完整的 distributed regression

本地单测已过，但每次改到 runtime core 后，最好在 Jetson 上跑：

- `tp-text-generate`
- `pp-mm-generate`
- `hybrid-mm-generate`

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

旧 debug/replay 在：

- `DEBUG_REPLAY_EXPORTS`

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
- legacy replay/capture/debug 入口进入 `LEGACY_*_EXPORTS` 或 `DEBUG_REPLAY_EXPORTS`。

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
- `_build_generate_phase_state`
- `_strip_runtime_layer_cache`
- `_build_generate_cache_map`
- `_is_runtime_only_generate_state`
- `_infer_runtime_tensor_device`
- `_infer_runtime_tensor_dtype`
- `_infer_runtime_token_dtype`
- `_build_runtime_only_stage_input_template`
- `_token_tensor_to_list`
- `_broadcast_token_id`

### 检查主路径是否还出现 stage_bundle

```bash
rg -n "stage_bundle" qwen3vl_tp_runtime \
  -g'*.py' \
  -g'*.md'
```

解释：

- 出现在 capture/replay/debug/file-backed reference 可接受。
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

这个项目已经从“Qwen3-VL replay 验证器”推进到“从 `model_path` 直接构建 `StageState` 的 correctness-first 分布式推理 runtime 原型”：纯 PP、纯 TP、PP+TP HYBRID 三条主路径都已建立，text generate 和 multimodal generate 的关键 smoke 已通过；TP/PP 是并列基础后端，HYBRID 是组合层；每个 rank/stage 已能只 materialize 自己需要的 StageState 和 text decoder shard，旧 replay bundle 只保留在 debug/capture 路径中。当前剩余重点是进一步清理兼容壳、把 debug TP replay 搬出主 TP 文件，以及后续做 embedding/lm_head vocab parallelism 和更系统的性能/显存基线。

## 最后状态备注

- 写入本文件之前，`git status --short` 没有输出，说明当时工作树是干净的。
- 本文件创建后，工作树会新增 `qwen3vl_tp_runtime/SESSION_HANDOFF.md`。
- 如果新对话接手后第一件事是继续写代码，建议先跑：

```bash
git status --short
```

确认只有预期文件变化。
