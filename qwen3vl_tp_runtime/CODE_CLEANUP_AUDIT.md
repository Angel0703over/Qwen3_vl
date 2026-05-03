# Step 24 Code Cleanup Audit

目标：让主路径代码、debug/replay 边界和文档入口都更“所见即所得”。本轮先收窄命名和依赖边界，不改变 generated ids/text、payload schema 或 baseline 语义。

## 24A 盘点结论

| 文件 | 行数 | 顶部 import 项 | def/class 数 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `hexgen_core/modules/hybrid_parallel.py` | 2268 | 91 -> 61 | 40 | 已完成 facade 收口；HYBRID 作为 PP+TP 组合层，调用来源更清楚 |
| `hexgen_core/modules/pipeline_parallel.py` | 1724 | 90 -> 58 | 39 | 已完成 facade 收口；legacy replay 仍留在文件内，后续 24E/24F 再评估拆分 |
| `hexgen_core/modules/tensor_parallel.py` | 1116 | 68 -> 57 | 24 | 已完成 facade 收口；已有 `StageRunner/TensorParallelRunner`，不急着类化 |
| `models/qwen3vl/runtime_builder.py` | 3799 | 104 | 35 | 太大，但包含大量 startup contract 和 builder 逻辑，拆分前需要测试保护 |
| `hexgen_core/distributed.py` | 967 | - | 49 | transport/collective/profile 混合；先用 facade 调用，不急拆 |
| `hexgen_core/transport.py` | 512 | - | 19 | 已有 `StageCommunicator`，边界较清楚 |

## 命名归类

| 名称 | 当前含义 | 是否保留 | 原因 / 下一步 |
| --- | --- | --- | --- |
| `StageState` | 主路径运行对象 | 保留 | PP/TP/HYBRID 都围绕它执行 |
| `runtime_inputs` | HYBRID wire protocol key | 保留 | schema 已冻结为 `hybrid_runtime_inputs_v1`，改名会动 payload 语义 |
| `model_input` | 后端内部 helper 名 | 保留 | 对齐 vLLM-style “model input”，不暴露到 wire key |
| `bundle` in `capture/debug/replay` | 历史 replay/capture payload | 保留 | 需要兼容旧 fixture 和 debug 路径 |
| `bundle` in execution main path | 主路径 layer/runtime state | 不保留 | Step 23 已改为 `layer_state/layer_runtime_state`，继续审计残留 |
| `--manifest-path` | replay/debug 入口 | 保留 | 需要 `--allow-debug-paths` 显式启用 |
| `--allow-debug-paths` | debug gate | 保留 | 防止 replay/capture 路径误入主运行 |

## 可动点清单

| 文件 | 问题 | 是否主路径 | 建议动作 | 风险 |
| --- | --- | --- | --- | --- |
| `hexgen_core/modules/hybrid_parallel.py` | 顶部从 PP/TP/distributed/execution/runtime_builder/capture/weights 引入大量函数，主逻辑不易看来源 | 是 | 已收成 `pp_backend` / `tp_backend` / `distributed_backend` / `qwen_execution` / `qwen_runtime_builder` / `qwen_capture` / `qwen_weights` facade | 低，纯调用名改写 |
| `hexgen_core/modules/pipeline_parallel.py` | direct runtime 与 legacy replay prepare helper 同文件；顶部从 distributed/capture/execution/runtime_builder/weights 引入大量函数 | 是/legacy 混合 | 已收成 `distributed_backend` / `qwen_capture` / `qwen_execution` / `qwen_runtime_builder` / `qwen_weights` facade；拆分 replay 留到 24E/24F | 中，外部 compat 可能引用旧导出 |
| `hexgen_core/modules/tensor_parallel.py` | import 区较长，但已有 runner 类 | 是 | 已收成 `distributed_backend` / `qwen_execution` / `qwen_runtime_builder` / `qwen_weights` / `qwen_capture_common` facade | 低 |
| `models/qwen3vl/runtime_builder.py` | startup contract、transport pack/restore、builder、manifest builder 混在一起 | 是 | 先只审计；未来按 `startup_contract.py` / `manifest_builder.py` 拆 | 中到高，payload bytes 验收敏感 |
| `scripts/live/live_multimodal_runtime.py` | “no-bundle” 脚本内部仍叫 `bundle` | debug/live | 可暂留，下一轮改成 `stage_state` 或迁到 debug doc | 低，但会影响测试和文档 |
| `tests/*bundle*` | 大量 legacy fixture 使用 bundle | test/compat | 暂留；只在对应代码改名后同步测试 | 低 |

## 24B backend import 收口

状态：完成。

结果：

| 文件 | before | after | 主要 facade |
| --- | ---: | ---: | --- |
| `hybrid_parallel.py` | 91 | 61 | `pp_backend`、`tp_backend`、`distributed_backend`、`qwen_execution`、`qwen_runtime_builder`、`qwen_capture`、`qwen_weights` |
| `pipeline_parallel.py` | 90 | 58 | `distributed_backend`、`qwen_capture`、`qwen_execution`、`qwen_runtime_builder`、`qwen_weights` |
| `tensor_parallel.py` | 68 | 57 | `distributed_backend`、`qwen_execution`、`qwen_runtime_builder`、`qwen_weights`、`qwen_capture_common` |

已完成动作：

1. `from .pipeline_parallel import prepare_*` -> `from . import pipeline_parallel as pp_backend`。
2. `from .tensor_parallel import ...` -> `from . import tensor_parallel as tp_backend`。
3. `from ..distributed import ...` -> `from .. import distributed as distributed_backend`。
4. `from ...models.qwen3vl.execution import ...` -> `qwen_execution`。
5. `from ...models.qwen3vl.runtime_builder import ...` -> `qwen_runtime_builder`。
6. HYBRID/PP legacy capture helper -> `qwen_capture`，TP replay `load/move_bundle` -> `qwen_capture_common`。
7. Qwen weights helper -> `qwen_weights`。
8. 调用点改成 `*_backend.*` / `qwen_*.*`，测试 patch 目标同步到真实调用点。
9. HYBRID 仍通过 lazy `__getattr__` 保留 TP helper compat，不放进 `__all__`。

验收：

- 三个 backend import 区均明显变短，依赖来源更清楚。
- `rg` 无旧裸调用残留。
- `git diff --check` 通过。
- 本地 focused tests 通过。

已验证：

- `git diff --check`
- `python -m py_compile qwen3vl_tp_runtime/hexgen_core/modules/pipeline_parallel.py qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
- `python -m unittest discover -s test -p 'test_pipeline_direct_loader.py'`
- `python -m unittest discover -s test -p 'test_tensor_parallel_direct.py'`
- `python -m unittest discover -s test -p 'test_hybrid_direct_loader.py'`
- `python -m unittest discover -s test -p 'test_compat_package_exports.py'`
- `python -m unittest discover -s test -p 'test_runtime_cli_modes.py'`
- `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --skip-baseline-checks`

## 24C 类化边界评估

结论：本轮不新增类，不改运行语义。现有类化边界已经覆盖稳定状态对象：backend runner、transport communicator、runtime builder、KV/cache metadata。纯函数 helper 继续保持函数形式，用模块 facade 表明来源；后续只有在出现稳定生命周期和共享状态时再新增类。

| 候选类 | 当前函数 | 状态字段 | 收益 | 风险 | 是否本轮做 |
| --- | --- | --- | --- | --- | --- |
| `TensorParallelRunner` | `StageRunner.run_rank`、`_run_generate_phase_tp`、`run_tensor_parallel_rank` | manifest、rank/world size、device、dtype、debug config | TP 主入口已经有清晰 runner 生命周期 | 低 | 已有，保留 |
| `TextPipelineRunner` / `TextGeneratePipelineRunner` | `run_text_pipeline_rank`、`_run_text_generate_phase_impl` | manifest、rank/world size、device、dtype、stage communicator | PP 主入口已经有清晰 runner 生命周期 | 低 | 已有，保留 |
| `TextHybridRunner` | `_run_text_generate_hybrid_rank`、`_run_text_generate_hybrid_phase_impl` | manifest、rank context、TP group、PP handoff、device、dtype | HYBRID 作为 PP+TP 组合 runner 的边界清楚 | 低 | 已有，保留 |
| `StageCommunicator` | `send_payload`、`recv_payload`、`broadcast_payload`、`send_hidden_states`、`recv_hidden_states` | device、comm dtype、pin memory policy | transport/session 状态集中，调用语义清楚 | 低 | 已有，保留 |
| `DirectStageStateBuilder` / `StageStateLoader` | `build_direct_stage_state`、`materialize_text_stage_state`、manifest builders | model path、stage specs、runtime config、loaded model/processor resources | runtime 构建生命周期已经集中 | 低 | 已有，保留 |
| `StageKVCache` / `VideoWindowCacheIndex` | `build_stage_kv_cache`、`attach_video_window_cache_index`、video KV compression helpers | layer KV buffer、current length、window metadata、KV location | KV 管理是稳定状态对象，已独立 | 低 | 已有，保留 |
| `StartupContractBuilder` | `_seed_mm_startup_contract`、`prepare_mm_startup_contract`、`pack_mm_startup_transport`、`restore_mm_startup_transport` | runtime config、owner rank、payload keys/tensors、transport bytes | 可把 startup contract 和 pack/restore 生命周期收口 | 中到高：payload bytes 和 key schema 敏感 | 否，等 24E/24F 拆文件时再评估 |
| `GeneratePhaseState` | `_build_generate_phase_state`、`_build_runtime_only_generate_phase_state`、PP/HYBRID 同名 helper | attention mask buffer、decode token buffer、cache map、stage KV cache | 可减少 PP/TP/HYBRID generate phase 重复逻辑 | 中：容易碰 decode mask、KV length、generated ids | 否，先保留函数 |
| `BackendRunContext` / `BackendSession` | backend runner 构造参数、rank context 解析、startup seed helper | manifest、rank context、device、dtype、groups、communicator | 如果 runner 参数继续膨胀，可统一上下文 | 中：会大范围改调用签名 | 否，当前收益不足 |
| `ExecutionEngine` | `forward_*`、`trace_*`、`*_tp` execution helpers | 无稳定状态，主要依赖传入 `stage_state/layer_state` | 收益小 | 高：会把简单 forward helper 包成函数容器 | 否，明确不做 |
| `DistributedSession` | `broadcast_*`、`send_*`、`recv_*`、`startup_log`、`startup_timer` | process group、profile events、rank/world size | 理论上可集中 distributed 状态 | 中：当前 distributed helper 是薄封装，隐藏状态会增加调试成本 | 否，用 `distributed_backend` facade 足够 |
| `RuntimeBuilderFacade` | `runtime_builder.py` 里的 pack/restore、manifest、builder 工具函数 | 状态混杂，部分无状态、部分 builder 生命周期 | 名义上可减少 import | 高：只是为了 import 行数包类，没有清晰职责 | 否，明确不做 |

24C 冻结原则：

1. 新增类必须有稳定状态和生命周期，例如 runner、communicator、builder、cache。
2. 纯计算/trace/forward helper 不类化，优先保持函数式入口。
3. startup contract、transport payload、generate phase 这类敏感路径先只做审计，不在 cleanup 阶段顺手改语义。
4. 如果后续拆 `runtime_builder.py`，先拆模块边界，再决定是否引入 `StartupContractBuilder`。

## 24D 主路径命名清理

状态：完成。本轮只收窄命名边界，不改 payload schema、generated ids/text 或 replay compat。

本轮代码改动：

1. `build_direct_stage_state` 不再给 `StageSpec` 显式传 `bundle_path=None`。
2. `build_direct_pipeline_manifest` 不再给 direct `StageSpec` / `TextPipelineManifest` 显式传 `bundle_path=None` / `bundle_dir=None`。
3. 保留 replay/debug/capture 兼容入口，但不放进 direct runtime `__all__`。

命名边界冻结：

| 名称 | 位置 | 归类 | 是否保留 | 说明 |
| --- | --- | --- | --- | --- |
| `StageState` | `schema.py`、backend runner、execution | 主路径对象 | 保留 | PP/TP/HYBRID direct runtime 的唯一运行对象 |
| `model_input` | `hybrid_parallel.py`、`runtime_builder.py` transport pack/restore | 后端内部 helper | 保留 | 对齐 vLLM-style “model input”，只在 HYBRID scaffold broadcast 内部使用 |
| `runtime_inputs` | `HybridRuntimeInputSchema`、HYBRID wire key、transport labels | wire protocol / metrics | 保留 | 已冻结为 `hybrid_runtime_inputs_v1`，改名会动 payload schema 和 baseline log |
| `runtime_input_source` / `runtime_input_broadcast_skipped` | TP runtime metrics | metrics 字段 | 保留 | 用来说明 pure TP 是否避免 dense input broadcast |
| `--manifest-path` | `scripts/runtime.py`、`runtime_cli.py` | debug/replay CLI | 保留 | 需要 `--allow-debug-paths` 显式开启 |
| `--allow-debug-paths` | `scripts/runtime.py`、`runtime_cli.py` | debug gate | 保留 | 防止 replay/capture/debug 入口误进主路径 |
| `bundle_path` / `bundle_dir` | `StageReplaySpec`、`ManifestReplaySpec`、prepare/replay helper | replay compat | 保留 | 只表达 captured bundle 文件位置；direct builder 不再显式传 `None` |
| `StageBundleView` / `as_stage_bundle_view` / `build_stage_bundle` | `hexgen_core/stage.py`、lazy legacy exports | legacy replay compat | 保留 | `StageStateView` / `as_stage_state_view` / `build_stage_state` 是主路径名字；旧名不进 direct `__all__` |
| `DirectStageBundleBuilder` / `build_direct_stage_bundle` | `runtime_builder.py`、lazy legacy exports | legacy compat | 保留 | `DirectStageStateBuilder` / `build_direct_stage_state` 是主路径名字；旧名不进 direct `__all__` |
| `stage_bundle` / `bundle` / `replay_bundle*` | `_MM_STARTUP_FORBIDDEN_KEYS` | payload guard | 保留 | 这些名字作为 forbidden key，用来防止 legacy payload 回流到 startup contract |
| `TextStageWeightBundle` / `load_text_decoder_stage_weight_bundle` | `models/qwen3vl/weights/text.py` | weight loader 历史 API | 暂留 | 表示一组静态权重，不是 runtime/replay payload；若要改名，放到 24F 走 alias + tests |

审计结果：

| 范围 | 结果 | 结论 |
| --- | --- | --- |
| `models/qwen3vl/execution/*` | `rg "bundle"` 无命中 | execution 主路径已完成 `layer_state` / `stage_state` 命名 |
| `models/qwen3vl/runtime_builder.py` | 剩余 `bundle` 为 forbidden keys、legacy alias、weight loader 历史字段 | 主 direct builder 已不再显式构造 legacy bundle args |
| `hexgen_core/modules/*.py` | `bundle_path/bundle_dir` 只在 prepare/replay helper 和 replay manifest load 分支；`model_input/runtime_inputs` 在 HYBRID broadcast/wire path | backend direct runner 命名边界清楚 |
| `hexgen_core/schema.py` | `runtime_inputs` 是协议；`bundle_path/bundle_dir` 是 replay compat shim | schema 兼容保留 |
| `scripts/runtime.py` / `runtime_cli.py` | `--manifest-path` 和 `--allow-debug-paths` 只作为 debug gate | CLI 主路径默认走 direct `model_path` |

24D 冻结原则：

1. 主路径代码优先使用 `StageState`、`StageStateView`、`build_stage_state`。
2. 内部可使用 `model_input`，wire key 继续叫 `runtime_inputs`。
3. `bundle` 只能出现在 replay/debug/capture/legacy compat、payload guard 或明确的 weight-loader 历史 API。
4. 若后续要把 `TextStageWeightBundle` 改成 `TextStageWeights`，必须保留 alias 并同步测试，不和主路径 cleanup 混在一起。

## 24E 文件归属整理

状态：完成。本轮只做职责边界审计和 Code Map 同步，不搬文件、不改 import 路径、不改变主运行语义。

归属边界：

| 范围 | 当前职责 | 不应该放入 | 本轮动作 |
| --- | --- | --- | --- |
| `hexgen_core/modules/pipeline_parallel.py` | 纯 PP backend runner；direct `StageState` 执行；保留 captured-bundle prepare/replay legacy entrypoints | Qwen layer math、权重切片、通用 transport primitive | 保留现状；legacy prepare/replay 后续 24F 再评估是否拆出 |
| `hexgen_core/modules/tensor_parallel.py` | 纯 TP backend runner；TP manifest load；rank-local StageState 执行；TP generate phase | HYBRID 组合逻辑、PP stage group 逻辑、Qwen 权重加载细节 | 保留现状；TP 不反向依赖 HYBRID |
| `hexgen_core/modules/hybrid_parallel.py` | PP+TP 组合 backend；stage group scaffold/model input broadcast；PP handoff + TP stage execution | 基础 TP backend 独立逻辑、Qwen layer math | 保留现状；HYBRID 可调用 PP/TP helper |
| `hexgen_core/distributed.py` | process group init、CPU/object/tensor collective、startup/profile logging、transport staging primitive | StageState schema、handoff payload schema、backend runner | 保留为 low-level distributed helper |
| `hexgen_core/transport.py` | `StageCommunicator`、payload send/recv/broadcast、dtype/pinned-memory transport policy | Process-group setup、backend phase scheduling、Qwen execution | 保留为 stage transport/session 层 |
| `hexgen_core/schema.py` | manifest、rank context、`StageState` type alias、HYBRID runtime input schema、payload summary | Backend execution、Qwen model loading、transport send/recv | 保留为数据契约层 |
| `hexgen_core/stage.py` | `StageStateView`、stage handoff payload build/apply、stage execution dispatch | Backend rank orchestration、model loading、distributed primitive | 保留为 StageState adapter/dispatch 层 |
| `hexgen_core/generate_buffers.py` | runtime-only decode mask/token 小 buffer 复用 | Backend orchestration、Qwen attention math | 保留为 generate buffer utility |
| `models/qwen3vl/execution/*` | Qwen text layer forward/trace、TP math wrapper | distributed send/recv、manifest parsing、CLI、KV 管理实现 | 保持纯 execution；不引入 backend 依赖 |
| `models/qwen3vl/kv_cache/*` | `StageKVCache`、video window metadata、video KV compression/compaction helper | backend orchestration、distributed send/recv、CLI | 已从 execution 拆出为独立 KV cache 子包；旧 `execution.*` 路径保留 shim |
| `models/qwen3vl/runtime_builder.py` | 从 `model_path` 构建 direct `StageState`、startup contract、transport pack/restore、direct manifests | Backend rank loop、CLI parsing、capture replay runner | 当前较大但职责仍是 builder；拆分留到 24F/后续 |
| `models/qwen3vl/runtime_mm_stage.py` / `runtime_text_stage.py` | runtime state rebuild、stage materialization、text/mm scaffold helper | backend collective、CLI、debug replay | 保留为 builder 支撑模块 |
| `models/qwen3vl/weights/*` | safetensors index、load plan、text/vision weight loading、TP shard slicing | Runtime scheduling、distributed transport、generated baseline checker | 保留为权重层 |
| `models/qwen3vl/processing/*` | Qwen processor/input builder、完整视频/frame-dir 输入组装 | Backend runner、transport payload | 保留为输入处理层 |
| `models/qwen3vl/vision/*` | Qwen3-VL vision frontend runtime、deepstack/bridge/state | Distributed rank orchestration、baseline checker | 保留为视觉 frontend 层 |
| `models/qwen3vl/capture/*` | captured bundle 生成、保存、加载兼容 | Direct runtime 主路径、backend live orchestration | 保留为 capture/replay legacy 资源层 |
| `models/qwen3vl/live/*` | live HF/Qwen 单机 helper、旧 bundle/live 输入辅助 | PP/TP/HYBRID distributed backend 主路径 | 保留为 live/debug helper，后续可进一步隔离 |
| `debug/*` | replay runner、TP debug config | Direct runtime 主路径、production helper | 保留为 debug/replay 路径 |
| `scripts/runtime.py` / `runtime_cli.py` | 统一 CLI、debug gate、backend dispatch | Qwen layer math、transport primitive | 保留为入口层 |
| `scripts/helpers/*`、`smoke_matrix.py`、`check_baseline_logs.py`、`collect_runtime_perf.py` | smoke wrapper、baseline checker、perf table | runtime 主路径逻辑 | 保留为验证/复现实验层 |

Code Map 同步：

1. `README.md` 的 `目录职责` 改为 `Code Map`，补齐 `distributed.py`、`transport.py`、`stage.py`、`runtime_builder.py`、`processing/vision/capture/live/debug/scripts` 的真实职责。
2. 明确 `capture/debug/live` 是 replay/debug/live helper，不是 direct runtime 主路径。
3. 明确 `execution/*` 不负责 distributed I/O，backend 文件不负责 Qwen layer math。
4. 明确 `kv_cache/*` 是 KV 管理子包，和 decoder execution 主路径分开。

24E 冻结原则：

1. `PP / TP / HYBRID` backend 文件只做 rank/stage orchestration。
2. `distributed.py` 是 low-level collective/profile helper；`transport.py` 是 stage payload session。
3. `execution/*` 是模型执行逻辑；`kv_cache/*` 是 KV 管理逻辑，二者都不向上依赖 backend。
4. `runtime_builder.py` 仍偏大，但本轮不拆；后续如拆，优先候选是 `startup_contract.py` 和 `manifest_builder.py`。
5. `capture/debug/replay/live` 继续隔离，不能成为 direct runtime 的默认入口。

## 24F 合并/删除冗余

状态：完成。本轮只删除可证明无调用点的 dead import；不合并主路径 helper，不删除 legacy compat，不改 payload schema 或 generated ids/text。

审计方式：

1. `rg "get_stage_output|run_stage_tp"` 确认 PP/TP 文件有真实调用，HYBRID 只有 import。
2. 对 `hybrid_parallel.py` / `pipeline_parallel.py` / `tensor_parallel.py` 做 AST import 使用扫描，确认 backend 文件里只有 HYBRID 的 `get_stage_output` / `run_stage_tp` 是未使用 import。
3. `rg "GenerateWorker|DecodeWorker|StageHandoffTransport|DirectStageBundleBuilder|build_direct_stage_bundle|StageBundleView|as_stage_bundle_view|build_stage_bundle|load_stage_bundle"` 审计 wrapper / alias 调用点。
4. `pyflakes` / `ruff` 当前环境不可用，因此不把全仓静态扫描作为硬依据；本轮只处理 `rg` 和 AST 都能确认的低风险项。

本轮代码改动：

| 文件 | 动作 | 原因 | 风险 |
| --- | --- | --- | --- |
| `hexgen_core/modules/hybrid_parallel.py` | 删除 `get_stage_output` / `run_stage_tp` import | HYBRID 文件内无调用；对应 helper 仍在 PP/TP 文件各自保留 | 低，import-only |

保留项：

| 候选 | 调用点 / 保护 | 结论 |
| --- | --- | --- |
| `DirectStageBundleBuilder` / `build_direct_stage_bundle` | `test_model_weight_loader.py` 仍大量使用；`test_compat_package_exports.py` 验证其 legacy export 边界 | 保留 legacy alias，不进 direct `__all__` |
| `StageBundleView` / `as_stage_bundle_view` / `build_stage_bundle` | `hexgen_core/stage.py` legacy compat；package lazy export 有测试覆盖 | 保留 legacy replay compat |
| `load_stage_bundle_by_index` / `load_stage_bundle_for_rank` | captured-bundle prepare/replay helper；`LEGACY_REPLAY_EXPORTS` 有测试覆盖 | 保留 replay helper，不作为主路径 API |
| `GenerateWorker` / `DecodeWorker` | 已不在 backend module exports；测试确认 `__all__` 和 module attr 都不存在 | 无需动作 |
| `StageHandoffTransport` | 已不在 package exports；测试确认不存在 | 无需动作 |
| `runtime_inputs` | HYBRID wire protocol key，baseline 和 checker 依赖 | 保留 |
| `bundle_path` / `bundle_dir` | replay manifest compat shim | 保留 |
| `TextStageWeightBundle` / `load_text_decoder_stage_weight_bundle` | 权重加载历史 API，不是 runtime payload bundle | 本轮不改名，避免破坏外部 compat |

24F 冻结原则：

1. 能证明无调用点的 import 可以删。
2. 只在 replay/debug/capture/legacy compat 里出现的旧名字可以保留，但必须有归类。
3. 行为一致但覆盖不足的 helper 不合并；等 tests 或 smoke 能覆盖后再动。
4. `runtime_builder.py` 暂不做拆分或 alias 删除，避免误伤 startup contract / transport pack restore。

已验证：

- `git diff --check`
- `python -m py_compile qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
- `python -m unittest discover -s test -p 'test_compat_package_exports.py'`
- `python -m unittest discover -s test -p 'test_qwen3vl_exports.py'`
- `python -m unittest discover -s test -p 'test_hybrid_direct_loader.py'`

## 24G README 所见即所得

状态：完成。本轮只重排 README，不改代码、不改 runtime 行为、不改 baseline 语义。

README 结构：

| section | 内容 | 验收 |
| --- | --- | --- |
| `Overview` | 项目目标、当前 scope、固定输出、默认路径 | 新读者先知道这是 correctness-first runtime prototype |
| `Quickstart` | CLI help、HF text、HF frame-dir multimodal、HF full-video | 每条命令都有 inputs / outputs / expected / troubleshoot |
| `Reproduce` | Step 22 matrix、optional full-video、TP/PP/HYBRID single smoke、checker、perf table、本地回归 | 命令旁边写清产物和失败排查入口 |
| `Results` | before/after 高信号表和当前 baseline dir | 数字来源指向 `BASELINE.md` 和 `baseline_runs/*` |
| `Architecture` | 固定术语、payload 规则、parallel/input shortcuts | 文档术语和代码命名保持一致 |
| `Code Map` | backend、distributed、execution、builder、processing、scripts 职责 | 与 24E 文件归属一致 |
| `Debug` | debug opt-in flags 和常见失败 checklist | replay/debug 路径不会和主路径混淆 |

24G 冻结原则：

1. README 作为入口文档，优先给能直接跑的命令。
2. 详细 baseline 数字继续放 `BASELINE.md`，README 只放高信号结果和链接。
3. 每个主命令必须说明输入、产物、期望输出、排查入口。
4. Debug/replay flag 必须显式写出 `--allow-debug-paths`。

已验证：

- `git diff --check`
- README 主 section 检查：`Overview / Quickstart / Reproduce / Results / Architecture / Code Map / Debug`

## 24H 验证冻结

状态：完成。Step 24 代码整理冻结，不再继续顺手重构。

本地验证：

| 命令 | 结果 |
| --- | --- |
| `git diff --check` | PASS |
| `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --skip-baseline-checks` | PASS |

真实 Jetson 子集：

```bash
TP_HOSTS="local 10.126.126.3" \
PP_HOSTS="local 10.126.126.3" \
HYBRID_HOSTS="local 10.126.126.3 10.126.126.4" \
MASTER_ADDR="10.126.126.2" \
bash qwen3vl_tp_runtime/scripts/helpers/run-step22-smoke-matrix.sh \
  --out baseline_runs/20260503-step24h-verify \
  --case-id pp-mm-generate \
  --case-id tp-mm-generate \
  --case-id hybrid-mm-generate
```

结果：

| case | ranks | generated | key metrics |
| --- | ---: | --- | --- |
| `pp-mm-generate` | 2 | `[104455, 9909, 9286, 16488]` / `人工智能（Artificial` | startup `3.06 MiB`，handoff `3.06 MiB` |
| `tp-mm-generate` | 2 | same | startup `11.50 MiB`，TP collective `220.43 MiB` |
| `hybrid-mm-generate` | 3 | same | stage0 TP collective `113.28 MiB`，stage1 `0 B` |

产物：

- `baseline_runs/20260503-step24h-verify/README.md`
- `baseline_runs/20260503-step24h-verify/check-smoke-matrix.txt`
- `baseline_runs/20260503-step24h-verify/runtime-perf-records.json`
- `baseline_runs/20260503-step24h-verify/runtime-perf-table.md`

24H 冻结结论：

1. Step 24 cleanup 没有改变 generated ids/text。
2. Checker 和 perf collector 能正常解析主路径 rank logs。
3. HYBRID `tp_degree=1` stage 继续保持 `0 B` TP collective。
4. 下一步进入 Step 25 性能候选规划，不继续在 cleanup 阶段扩散重构。

## 24I KV cache 包拆分

状态：完成。本轮只调整代码脚手架，不改 KV cache 语义、不改 generated ids/text、不改 payload schema。

本轮代码改动：

1. 新增 `models/qwen3vl/kv_cache/` 子包，集中放置：
   - `kv_cache.py`：`LayerKVCache` / `StageKVCache` / `build_stage_kv_cache`
   - `video_window_cache.py`：`VideoWindowCacheIndex` 和 window -> KV location metadata
   - `video_kv_compression.py`：planner、selector、contract、opt-in compaction helper
2. `models/qwen3vl/execution/__init__.py` 继续 re-export KV API，backend facade 调用不变。
3. 旧路径 `models/qwen3vl/execution/kv_cache.py`、`video_window_cache.py`、`video_kv_compression.py` 已删除，不再保留子模块级兼容层。
4. `execution/attention.py`、`execution/stages.py` 的主路径直接从 `models/qwen3vl/kv_cache` 引入 cache 类型。

边界冻结：

| 范围 | 现在职责 | 说明 |
| --- | --- | --- |
| `execution/` | decoder/attention/MLP/stage forward 和 trace | 不再承载 KV cache 实现 |
| `kv_cache/` | KV buffer、video window index、video KV compression | 有状态 cache 和 KV 管理策略集中在这里 |
| `execution.__init__` re-export | package-level API compat | 只保留顶层 re-export，不保留 `execution.kv_cache` 子模块 |

验收口径：

- `execution` re-export 的 API 名字不变。
- 旧子模块 import path 不再支持；项目内无调用点。
- 本轮是 import/文件归属调整，不需要重新冻结 payload bytes。

## 24J 全仓冗余扫描

状态：完成第一轮。只删除确定无语义影响的生成缓存和 dead import；re-export、legacy compat、debug/replay 入口保留。

扫描动作：

1. 删除 `qwen3vl_tp_runtime/` 和 `test/` 下生成的 `__pycache__` / `.pyc` / `.pytest_cache`。
2. 用 AST 扫描全仓 Python 文件的文件内未使用 import。
3. 用 `rg` 复核候选名字在全仓的调用点。
4. 对 `__init__.py` re-export、compat shim、debug/replay lazy export 做保留归类。

本轮代码删除：

| 文件 | 删除内容 | 删除原因 | 风险 |
| --- | --- | --- | --- |
| `models/qwen3vl/runtime_builder.py` | `_run_live_prefill_stage_reference` import alias | 全仓无调用点；旧 stage reference 路径未暴露在 `__all__` | 低，import-only |
| `test/test_hybrid_direct_loader.py` | `restore_mm_startup_transport` import | 测试文件内无调用点；对应 API 本身仍保留 | 低，test import-only |

明确保留：

| 候选 | 保留原因 |
| --- | --- |
| `__init__.py` 里的大量 import | package public API / lazy export / compat export，不能按文件内未使用删除 |
| `debug/tensor_parallel_replay.py` 的 `build_stage_traces` / `tensor_diff_stats` | debug replay module re-export，保留 legacy 调试入口 |
| `hexgen_core/gen_p2p_lists.py` | HexGen-style p2p builder compatibility shim，README 仍有入口说明 |
| `processing/loaders.py` 的 `load_tensors_by_name` | processing package re-export，测试和外部 helper 仍可能引用 |
| `runtime_builder.py` 的 `load_text_tokenizer*` | `runtime_text.py` 通过 `_builder_dep` 支持测试 patch 和 builder dependency override |
| `runtime_builder.py` 的 text scaffold/model input transport helpers | backend facade 通过 `qwen_runtime_builder.*` 调用，属于稳定 helper surface |
| `weights/planner.py` 的 `TextModelConfigSpec` TYPE_CHECKING import | 字符串类型注解使用，静态 AST 会误报 |
| duplicate `tensor_diff_stats` | 暂不合并；分别处于 backend/debug/scripts 边界，合并会动 package export 和 debug compat |
| `scripts/runtime.py` 的 `build_direct_*_manifest` import | 文件内不直接调用，但 `runtime_cli` 测试和 lazy dependency patch 依赖这些模块级入口 |

24J 冻结原则：

1. 生成缓存可以删。
2. 文件内 dead import 可以删，但必须用 `rg` 复核调用点。
3. re-export、compat shim、debug/replay API 不按普通 dead code 删除。
4. 重复函数只有在职责边界一致、测试覆盖充分时再合并。

## 24K KV 子模块兼容层删除

状态：完成。删除 24I 临时保留的 KV 旧子模块 shim，只保留 `execution.__init__` 的顶层 re-export。

删除文件：

| 文件 | 替代路径 |
| --- | --- |
| `models/qwen3vl/execution/kv_cache.py` | `models/qwen3vl/kv_cache/kv_cache.py` 或 `models/qwen3vl/kv_cache` |
| `models/qwen3vl/execution/video_window_cache.py` | `models/qwen3vl/kv_cache/video_window_cache.py` 或 `models/qwen3vl/kv_cache` |
| `models/qwen3vl/execution/video_kv_compression.py` | `models/qwen3vl/kv_cache/video_kv_compression.py` 或 `models/qwen3vl/kv_cache` |

保留边界：

- `from qwen3vl_tp_runtime.models.qwen3vl.execution import StageKVCache` 仍可用。
- `from qwen3vl_tp_runtime.models.qwen3vl.execution.kv_cache import StageKVCache` 不再支持。
- 主路径和测试都不再直接 import 旧 KV 子模块。
