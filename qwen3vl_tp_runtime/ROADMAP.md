# qwen3vl_tp_runtime Roadmap

这份 Roadmap 只记录接下来要做的事情。更完整的历史状态、对话迁移信息和已完成细节见：

- `README.md`
- `BASELINE.md`
- `SESSION_HANDOFF.md`

## 当前完成快照

当前主线里程碑已经完成：

- `PP / TP / hybrid` 主路径可以在启动时直接从 `model_path` 构建每个 stage/rank 的 `StageState`。
- 主路径术语已经从旧 `bundle` 收口为 `StageState`；`bundle` 只保留在 replay/debug/capture/file-backed reference 语义里。
- `backend=tp` 已经是独立后端，有自己的 `TensorParallelManifest / build_direct_tp_manifest / TensorParallelRunner / run_tensor_parallel_rank`。
- `backend=tp` 不再依赖 `TextHybridManifest`、`TextHybridRunner` 或 `hybrid_parallel.py`。
- 架构方向已经明确为：`PP` 和 `TP` 是并列基础后端，`HYBRID` 是依赖二者的组合后端。
- direct `tp_degree > 1` stage 已经走 rank-local materialize：每个 TP rank 只加载自己的 text decoder projection shard。
- `pp / hybrid multimodal generate` 已通过 Jetson smoke，stage scope、frontend ownership、startup transport thin contract、TP shard-local materialize 都有 summary/log 证据。
- `--pp N` 和 `--tp N` 已作为推荐 CLI 参数；`--stage-ranges` / `--tp-degrees` 保留为高级覆盖。

当前不要重复做这些大目标，后续主要进入整理、自动化和性能阶段。

## 架构不变量

后续所有改动都必须遵守：

- `pipeline_parallel.py` 是纯 PP 基础后端。
- `tensor_parallel.py` 是纯 TP 基础后端。
- `hybrid_parallel.py` 是 PP+TP 组合后端。
- `hexgen_core/modules/` 目录只放三种并行后端文件和 `__init__.py`：`pipeline_parallel.py`、`tensor_parallel.py`、`hybrid_parallel.py`。
- HYBRID 可以调用 PP/TP 的函数。
- TP 不能反向依赖 HYBRID。
- PP 不应为了复用少量 helper 反向依赖 TP，除非后续明确引入新的公共基础模块。
- 主路径 public API 不再使用 `bundle` 表达 runtime 对象。
- `--manifest-path`、capture、replay、trace、dump 都属于 debug/replay 路径，不是默认主路径。

## P0：当前优先清理

### 1. 已完成：处理 TP 里的空壳 worker 类

背景：

- `tensor_parallel.py` 之前有 `StageRunner / GenerateWorker / DecodeWorker / TensorParallelRunner`。
- `StageRunner` 和 `TensorParallelRunner` 有实际职责。
- `GenerateWorker` / `DecodeWorker` 主要是命名对齐，没有独立职责。

目标：

- 让 TP worker 类名和职责一致。

已采用方案：

- 删除 `GenerateWorker / DecodeWorker` 空壳。
- `TensorParallelRunner` 直接继承 `StageRunner`。
- `tensor_parallel.py.__all__` 不再导出 TP 的 `GenerateWorker / DecodeWorker`。

验收：

- `TensorParallelRunner -> StageRunner` 继承关系清晰。
- TP 主路径不再出现完全空壳 worker 类。
- `test/test_tensor_parallel_direct.py` 通过。
- `test/test_compat_package_exports.py` 已同步导出面预期。

### 2. 已完成：移出 TP 文件里的旧 replay/debug TP 路径

背景：

- `tensor_parallel.py` 之前同时包含主路径 TP 和旧 replay/debug TP。
- 旧路径包括：
  - `load_text_stage_bundle`
  - `run_text_tensor_parallel_stage`
  - `TextTensorParallelRunner`
  - `run_text_tensor_parallel_rank`
  - `DEBUG_REPLAY_EXPORTS`

目标：

- 让 `tensor_parallel.py` 成为纯主路径 TP 文件。
- 旧 replay/debug 仍然可用，但移动到更明确的位置。

已采用方案：

- 新增顶层 debug 包 `qwen3vl_tp_runtime/debug/`。
- TP trace/debug helper 放到 `qwen3vl_tp_runtime/debug/tp_debug.py`。
- 旧 captured-bundle TP replay 入口放到 `qwen3vl_tp_runtime/debug/tensor_parallel_replay.py`。
- `hexgen_core/modules/` 下只保留三种并行后端，不再放 debug/replay 文件。
- `hexgen_core.modules` 继续用 lazy compat export 暴露旧 replay 名字，但实际实现来自顶层 `debug/`。

验收：

- `tensor_parallel.py.__all__` 只保留 direct TP 主路径。
- `tensor_parallel.py` 不再包含 `DEBUG_REPLAY_EXPORTS` 或旧 captured-bundle replay runner。
- 旧 replay/debug import 通过 `qwen3vl_tp_runtime.debug.tensor_parallel_replay.DEBUG_REPLAY_EXPORTS` 或 `hexgen_core.modules` compat 路径访问。
- `test/test_compat_package_exports.py` 覆盖新的位置和兼容路径。
- `test/test_tensor_parallel_direct.py` 通过。

### 3. 已完成：规范 HYBRID 调用 TP helper 的方式

背景：

- HYBRID 依赖 TP，因此 HYBRID 直接复用 TP helper 是正确方向。
- `hybrid_parallel.py` 之前从 `tensor_parallel.py` import 了若干 `_` 开头 helper。

目标：

- 保持依赖方向正确，同时让代码风格更清楚。

已采用方案：

- 采用方案 B：把 HYBRID 复用的 TP generate helper 改成无下划线的 module-level helper。
- 这些 helper 不放入 `tensor_parallel.py.__all__`，也不进入 `hexgen_core.modules.__all__` / `DIRECT_RUNTIME_EXPORTS`。
- `hybrid_parallel.py` 加注释说明：HYBRID 是组合层，可以复用 pure TP backend 的 backend-level helper。

当前 helper 名：

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

验收：

- `tensor_parallel.py` 仍不依赖 HYBRID。
- `hybrid_parallel.py` 不再复制这些 helper。
- `hybrid_parallel.py` 不再 import TP 的 `_` 私有 helper。
- `test/test_compat_package_exports.py` 覆盖这些 helper 只作为 TP module-level 复用点，不进入 package-level public API。
- `test/test_hybrid_direct_loader.py` 和 `test/test_tensor_parallel_direct.py` 通过。

### 4. 已完成：收紧 StageState / bundle 命名残留

背景：

- 主路径已经迁移到 `StageState`。
- 仍有一些 `bundle` 出现在 file-backed reference、capture/replay/debug、layer weight 兼容变量中。

目标：

- 区分“应该保留的 legacy/debug bundle”和“主路径里漏掉的 bundle 命名”。

检查命令：

```bash
rg -n "stage_bundle|bundle" qwen3vl_tp_runtime -g'*.py' -g'*.md'
```

已采用规则：

- capture/replay/debug/file-backed reference：可以保留。
- weight loader / execution 层内部的 `layer_bundle` 或 `bundle`：可以保留，因为表示层参数集合，不是 stage runtime object。
- schema 里的 `bundle_path` / `bundle_dir`：只作为 legacy replay 兼容属性保留；主路径 direct manifest 不序列化这些字段。
- `build_stage_bundle` / `DirectStageBundleBuilder` / `build_direct_stage_bundle`：只作为 legacy alias 保留，不进入 direct public `__all__`。
- direct 主路径 public API / summary / README 主路径说明：不应该继续叫 bundle。

本轮清理：

- `pipeline_parallel.py` 文件说明改为 direct `StageState` 主路径，captured-bundle prepare/replay 是 legacy 入口。
- `debug.tp_debug.build_stage_traces()` 参数从 `bundle` 改为 `stage_state`。
- `runtime_builder.py` 内部 file-backed reference / trace 临时变量和 helper 从 `stage_bundle` 收口为 `stage_state`。
- 删除未使用的 `_restore_text_prompt_bundle` 旧名 helper。
- legacy alias 加说明，明确不属于 direct-runtime export surface。

验收：

- 新增或修改的主路径代码不引入新的 `stage_bundle` 命名。
- `README.md` / `ROADMAP.md` / `BASELINE.md` 术语保持一致。
- `test/test_compat_package_exports.py` 继续确认 legacy bundle API 不在 direct `__all__`。

## P1：自动化回归

### 5. 已完成：增加 baseline log 检查脚本

背景：

- 当前 `BASELINE.md` 已记录应检查字段。
- 真实分布式 smoke 仍主要靠人工读 JSON summary 和日志。

目标：

- 增加一个脚本读取 rank log，自动检查关键字段。

已采用文件：

- `qwen3vl_tp_runtime/scripts/check_baseline_logs.py`
- `test/test_check_baseline_logs.py`

支持：

- 输入一个 case id 和多个 rank log。
- 自动提取最后一个 JSON summary。
- 检查 `generated_token_ids` / `generated_text` 是否一致。
- 对 `tp-text-generate` / `tp-mm-generate` 检查：
  - `tp_weight_sharded=true`
  - `tp_shard_rank`
  - `tp_shard_world_size`
  - `tp_shard_shape_ok`
  - 各 rank `loaded_weight_tensor_bytes` 一致
  - 如果存在 `tp_stage_loaded_weight_tensor_bytes_equal`，该字段必须为 `true`
- 对 `pp-text-generate` / `pp-mm-generate` 检查：
  - stage scope
  - `pp-mm-generate` 额外检查 stage0 frontend active / stage1 consume-only
- 对 `hybrid-text-generate` / `hybrid-mm-generate` 检查：
  - stage0 TP ranks shard-local
  - stage1 stage-local
  - all ranks generated ids 一致

验收：

- 能检查 `baseline_runs/20260428/*-rank*.log`。
- 检查失败时错误信息指向具体 rank 和字段。
- `test/test_check_baseline_logs.py` 通过。

### 6. 已完成：固化 runtime smoke wrapper

背景：

- `BASELINE.md` 已有命令，但分布式多 rank 仍需要手动复制。

目标：

- 给常用 case 增加稳定 shell wrapper，减少手动出错。

已采用方案：

- 保留 `scripts/helpers/run-runtime.sh`。
- 新增：
  - `scripts/helpers/run-pp-mm-generate.sh`
  - `scripts/helpers/run-tp-mm-generate.sh`
  - `scripts/helpers/run-hybrid-mm-generate.sh`

支持：

- 支持传 `NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。
- pure PP/TP 支持 `NNODES>=2`；2 节点只是默认值。
- pure PP 默认 `PP=NNODES`，pure TP 默认 `TP=NNODES`。
- pure PP/TP wrapper 会提前校验当前 degree 必须等于 torchrun world size。
- 默认 `OUT=baseline_runs/$(date -u +%Y%m%d)`。
- 默认使用 `--pp` / `--tp` 参数，不再默认写 `--stage-ranges`。
- 支持 `DRY_RUN=1` 只打印最终命令。
- 支持把额外 runtime 参数直接追加到 wrapper 后面。

验收：

- 每个 wrapper 打印最终命令。
- 用户可以直接在 rank0/rank1/rank2 机器上运行。
- `bash -n` 和 `DRY_RUN=1` 命令展开已通过。

## P2：真实分布式回归补齐

### 7. 已完成：重跑并冻结完整 baseline

本轮冻结日期：

- `20260428`

输出目录：

- `baseline_runs/20260428/`

已重跑并冻结：

- `hf-text-generate`
- `hf-mm-generate`
- `live-mm-generate`
- `pp-text-generate`
- `pp-mm-generate`
- `tp-text-generate`
- `tp-mm-generate`
- `hybrid-text-generate`
- `hybrid-mm-generate`

本轮顺手修复：

- `live-mm-generate` decode/generate 路径缺少 `MmVisualState` import。
- `live-mm-generate` summary 补齐标准 `generated_token_ids` / `generated_text` 字段。
- pure TP multimodal full-stage prefill reference 不再按 layer boundary 构造单 stage handoff，避免 `0:35` full stage 越界。

验收：

- 每个 case 都有 rank log 或 stdout/stderr 文件。
- `BASELINE.md` 更新实际日期、log 路径、generated ids、generated text。
- distributed case 用新的 baseline checker 自动检查，结果保存为 `baseline_runs/20260428/check-baseline-logs.txt`。

### 8. 已完成：每次 runtime core 变更后的最小回归矩阵

如果改到这些文件：

- `tensor_parallel.py`
- `pipeline_parallel.py`
- `hybrid_parallel.py`
- `runtime_builder.py`
- `runtime_text_stage.py`
- `weights/`
- `transport.py`
- `schema.py`

固定入口：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
```

该脚本会跑：

- `test/test_check_baseline_logs.py`
- `test/test_collect_runtime_perf.py`
- `test/test_runtime_builder_handoffs.py`
- `test/test_tensor_parallel_direct.py`
- `test/test_pipeline_direct_loader.py`
- `test/test_hybrid_direct_loader.py`
- `test/test_runtime_cli_modes.py`
- `test/test_runtime_summary.py`
- `test/test_compat_package_exports.py`
- `baseline_runs/20260428` 下 6 个 distributed case 的 baseline checker

如果改到权重加载：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --include-weight-loader
```

如果改到真实 generate 路径，额外在 Jetson 跑：

- `tp-text-generate`
- `pp-mm-generate`
- `hybrid-mm-generate`

可选项：

- `--baseline-dir PATH`：切换冻结 baseline log 目录。
- `--skip-baseline-checks`：只跑本地单测，不检查冻结 rank log。

验收：

- `bash -n qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh` 通过。
- 默认矩阵已通过。

## P3：性能和显存基线

### 9. 已完成：建立启动耗时和显存记录

已落地：

- runtime JSON summary 增加 `runtime_metrics`：
  - `runtime_metrics.timing.runtime_total_seconds`
  - `runtime_metrics.startup.events`
  - `runtime_metrics.startup.totals_by_kind.prepare_session_seconds`
  - `runtime_metrics.startup.totals_by_kind.startup_contract_transport_seconds`
  - `runtime_metrics.startup.totals_by_kind.materialize_stage_seconds`
  - `runtime_metrics.startup.totals_by_kind.post_load_barrier_seconds`
  - `runtime_metrics.memory.cpu_max_rss_bytes`
  - `runtime_metrics.memory.peak_allocated_bytes`
  - `runtime_metrics.memory.peak_reserved_bytes`
- `startup_timer` 现在会记录机器可读事件。
- object/tensor send/recv/broadcast 会记录 transport timing。
- PP/HYBRID post-load barrier 已纳入 timing。
- TP/HYBRID rank-local shard materialize 已纳入 timing。
- HF/live summary 也会带 runtime/memory metrics。

收集入口：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/collect_runtime_perf.py \
  --baseline-dir baseline_runs/20260428 \
  --output-json baseline_runs/20260428/runtime-perf-records.json \
  --output-md baseline_runs/20260428/runtime-perf-table.md
```

当前 20260428 correctness baseline 已生成兼容旧日志的性能表：

- `baseline_runs/20260428/runtime-perf-records.json`
- `baseline_runs/20260428/runtime-perf-table.md`

注意：

- 旧 log 没有 `runtime_metrics.memory.*`，所以当前表里 CUDA peak 显存为空。
- 新代码重跑 baseline 后，summary 会直接写入 peak allocated/reserved。
- 旧 multimodal wrapper log 没有 `/usr/bin/time real`，新代码重跑后用 `runtime_total_seconds` 补齐。

验收：

- `BASELINE.md` 增加性能/显存表。
- 每个 rank 的 timing/memory 可以从 JSON summary 或 `collect_runtime_perf.py` 读出。
- `test/test_collect_runtime_perf.py` 通过。
- `test/test_runtime_summary.py` 通过。

### 10. 性能优化执行原则

性能阶段从现在开始不再是一条大任务，而是按可观测、低风险减量、执行重排、拓扑搜索逐步推进。

通用规则：

- 每个优化前先保留 before 表：
  - `runtime-perf-records.json`
  - `runtime-perf-table.md`
- 每个优化后重跑同一组 case，生成 after 表。
- correctness guard 必须先过：
  - `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`
- 如果改到权重加载：
  - `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --include-weight-loader`
- 如果改到真实 distributed generate 路径，至少重跑对应 smoke：
  - `tp-text-generate`
  - `pp-mm-generate`
  - `hybrid-mm-generate`
- 性能优化不应改变 `generated_token_ids` / `generated_text`。
- 每个优化只动一个主目标，避免把多个变量混在一次对比里。

建议输出目录命名：

```bash
baseline_runs/YYYYMMDD-perf-<topic>-before/
baseline_runs/YYYYMMDD-perf-<topic>-after/
```

### 11. 已完成：先补齐 transport / payload profiling

已落地：

- `runtime_metrics.transport` 会记录机器可读 transport/profile 事件。
- `PayloadSummary` 增加：
  - `tensor_dtypes`
  - `tensor_numels`
  - `tensor_bytes`
  - `total_tensor_bytes`
- object/tensor send/recv/broadcast 会记录：
  - label
  - peer
  - elapsed seconds
  - object bytes
  - tensor count / shapes / dtypes / bytes
- `StageCommunicator` 会记录 PP/HYBRID stage handoff payload。
- TP collective helper 会记录：
  - `all_reduce_cpu`
  - `all_gather_cpu`
  - `broadcast_cpu`
- `collect_runtime_perf.py` 会汇总：
  - startup contract bytes
  - scaffold bytes
  - stage handoff bytes / seconds
  - TP collective bytes / seconds

`runtime_metrics.transport` 结构：

```text
runtime_metrics.transport.events[]
runtime_metrics.transport.totals_by_kind.startup_contract
runtime_metrics.transport.totals_by_kind.scaffold
runtime_metrics.transport.totals_by_kind.stage_handoff
runtime_metrics.transport.totals_by_kind.tp_collective
runtime_metrics.transport.totals_by_channel.*
```

注意：

- `baseline_runs/20260428` 是 frozen correctness baseline，payload bytes 列仍为空。
- `baseline_runs/20260428-step11-profile` 已用新代码重跑 4 个 2 节点 distributed case，payload bytes / TP collective timing 已补齐。
- HYBRID profiling 仍需在 Jetson1 正常终端参与时重跑。

验收：

- `runtime_metrics` 或 rank summary 中能读出 payload bytes / tensor shapes。
- `collect_runtime_perf.py` 能把关键 payload 指标汇总到 JSON。
- correctness baseline 不变。
- `test/test_runtime_summary.py` 通过。
- `test/test_collect_runtime_perf.py` 通过。
- `run-runtime-core-regression.sh` 通过。
- `run-runtime-core-regression.sh --include-weight-loader --skip-baseline-checks` 通过。

### 12. startup contract tensor 减量

目标：

- 减少 multimodal startup contract 传输的 tensor 数量和总 bytes。
- 保持 thin contract 语义，不回退到 full/root/replay payload。

候选方向：

- 去掉 non-stage 必要性不强的 visual/deepstack tensor。
- 只给目标 stage 发送它真正需要的 handoff / visual state。
- 对 metadata 和 tensor payload 做更严格的 stage-local select。

验收：

- `pp-mm-generate` 和 `hybrid-mm-generate` generated ids/text 不变。
- startup contract payload bytes 下降或保持不增。
- non-stage0 仍是 consume-only，不重新激活 frontend。

### 13. scaffold broadcast 减量

目标：

- 减少 HYBRID stage leader 广播给同 stage TP rank 的 scaffold 内容。
- 避免把 rank-local 可以直接加载的权重或不必要 runtime reference 放进 scaffold。

候选方向：

- scaffold 只保留 runtime metadata、shared input、必要 auxiliary tensor。
- TP rank 自己 materialize 的 decoder weights 不经过 scaffold broadcast。
- 检查 `text_scaffold` / `stage_scaffold` 的 tensor keys 和 bytes。

验收：

- `hybrid-text-generate` / `hybrid-mm-generate` correctness 不变。
- scaffold broadcast bytes 下降或保持不增。
- `weight_load.tp_weight_sharded=true` 和 stage-local scope 证据不回退。

### 14. TP collective profiling 和低风险调整

目标：

- 先量化 TP all-reduce / all-gather 的实际耗时，再决定是否优化。

优先记录：

- 每个 decode/prefill step 的 collective 次数。
- collective tensor shape / dtype / elapsed seconds。
- rank 间耗时是否明显不一致。

低风险候选：

- 减少不必要 dtype cast。
- 合并小 collective。
- 避免重复 all-gather。

验收：

- `tp-text-generate` / `tp-mm-generate` correctness 不变。
- collective timing 出现在 summary 或 perf records。
- 优化后 collective 总耗时下降或保持不增。

### 15. buffer reuse / pinned memory 实验

目标：

- 减少反复分配 hidden / handoff / decode step tensor 的开销。
- 评估 Jetson 上 pinned memory 对 CPU transport / CPU->GPU copy 是否有收益。

执行顺序：

1. 先只做 profiling，不改语义。
2. 再做 stage-local buffer reuse。
3. 最后单独实验 pinned memory。

验收：

- correctness 不变。
- CUDA peak reserved / allocated 不上升，最好下降。
- runtime_total_seconds 或 transport seconds 有可解释变化。

### 16. PP handoff overlap

目标：

- 尝试让 PP stage 间 hidden handoff 与本地计算重叠，减少纯等待时间。

前置条件：

- PP handoff payload bytes 和 send/recv timing 已经可观测。
- 当前阻塞点明确，不先盲目改通信结构。

验收：

- `pp-text-generate` / `pp-mm-generate` correctness 不变。
- PP handoff wait 时间下降。
- 没有引入 rank 间死锁风险。

### 17. stage partition 搜索

目标：

- 根据每个 stage 的耗时和显存，搜索更合适的 stage range。
- 支持不同 Jetson 数量和异构性能。

候选方向：

- 先手工比较 `0:17 / 18:35` 附近的切分。
- 后续再写脚本自动枚举 stage ranges。
- 对 HYBRID 同时考虑 stage range 和 `tp_degrees`。

验收：

- 每组 partition 都有 correctness + perf 表。
- 找到比当前 baseline 更好的 total time / peak memory 组合。
- 文档记录最终推荐 partition。

## P4：功能增强

### 20. embedding / lm_head vocab parallelism

当前状态：

- decoder projection / MLP projection 已按 TP shard-local 加载。
- `embed_tokens_weight` 和 `lm_head_weight` 仍按当前执行语义复制。

目标：

- 研究并实现 vocab parallel embedding / lm_head。

风险：

- 会影响 token embedding、logits gather、top-k、sampling。
- 会影响 PP stage0 和 last stage 的权重 scope。

验收：

- TP rank 不再完整复制 embedding/lm_head。
- logits 与当前 reference 对齐。
- `weight_load` 能证明 vocab shard shape。

### 21. KV cache session 化

当前状态：

- KV cache 主要以 correctness-first 的 `cache_by_layer` 字典存在。

目标：

- 设计 stage-local cache manager。
- prefill 后 cache 持久化。
- decode step 复用 cache。
- 后续支持多请求。

验收：

- text generate 和 multimodal generate 不依赖 replay/reference cache。
- cache 生命周期清晰。
- 可以区分 session / request / stage / layer。

### 22. 更接近 serving engine

长期目标：

- request scheduler。
- paged KV cache。
- block manager。
- continuous batching。
- streaming output。
- prefill/decode overlap。

这不是当前 P0/P1 任务，只有在 correctness、baseline、性能记录稳定后再推进。

## Definition Of Done

一个 Roadmap 任务完成时，至少满足：

- 代码改动符合依赖方向。
- 主路径术语仍是 `StageState`。
- debug/replay 路径没有被误认为主路径。
- 相关 unit tests 通过。
- 如果影响分布式 runtime，至少跑一个真实 Jetson smoke。
- 如果影响主路径文档或命令，更新 `README.md` / `BASELINE.md` / `SESSION_HANDOFF.md` 中对应内容。
- 如果需要在 Jetson 上继续测试，完成同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```
