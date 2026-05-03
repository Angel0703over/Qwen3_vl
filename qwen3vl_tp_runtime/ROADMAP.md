# qwen3vl_tp_runtime Roadmap

这份文档只放当前状态和下一步规划。详细数字看 `BASELINE.md`，新对话接手看 `SESSION_HANDOFF.md`，原始日志看 `baseline_runs/*/README.md`。

## 当前状态

KV cache 管理先冻结到 20C-4，20D 历史窗口检索回取暂不推进。

| 方向 | 状态 |
| --- | --- |
| direct runtime | `PP / TP / HYBRID` 都从 `model_path` 构建 `StageState` |
| 后端边界 | `PP` 和 `TP` 是基础后端，`HYBRID` 是组合层 |
| TP 权重 | decoder/MLP projection 已 rank-local materialize |
| multimodal startup | 只传必要 runtime tensors/metadata，不传 root/full/replay payload |
| runtime input | pure TP 避免 dense `stage_input` broadcast；HYBRID schema 已固化 |
| comm dtype | 默认 `bfloat16` |
| KV cache | `StageKVCache -> VideoWindowCacheIndex -> opt-in video KV compaction` 已跑通；实现已独立到 `models/qwen3vl/kv_cache/` |
| 完整视频输入 | `--video-path` 已通过 HF/PP/TP/HYBRID smoke |
| smoke 自动化 | Step 22 required matrix 已通过，含 PP=3 和 3-rank HYBRID |
| API 清理 | Step 23 已完成第一轮：主路径命名、prompt 贯通、legacy lazy export |

当前固定输出：

| case | ids/text |
| --- | --- |
| text generate | `[104455, 9909, 9286, 16488]` / `人工智能（Artificial` |
| frame-dir multimodal with CLI prompt | `[104455, 9909, 9286, 16488]` / `人工智能（Artificial` |
| full-video default video prompt | `[87140, 108869, 100369, 102122]` / `视频展示了两个场景` |

## 已冻结 Baseline

| 阶段 | baseline | 用途 |
| --- | --- | --- |
| correctness | `baseline_runs/20260428/` | 早期固定输出 |
| current perf | `baseline_runs/20260430-bfloat16-default/` | 当前性能快照 |
| Step 15 | `baseline_runs/20260430-step15-derived-rebuild/` | payload derived tensor 本地重建 |
| Step 16 | `baseline_runs/20260501-step16-pinned-ab/` | buffer reuse / pinned A/B |
| Step 20A | `baseline_runs/20260501-step20a-kv-cache-smoke/` | `StageKVCache` smoke |
| Step 20B | `baseline_runs/20260501-step20b-video-window-cache/` | video window metadata |
| Step 20C-3 | `baseline_runs/20260502-step20c3-compaction/` | `uniform` physical compaction |
| Step 20C-4 | `baseline_runs/20260502-step20c4-infinipot-selector/` | `infinipot-v` selector |
| Step 21 | `baseline_runs/20260502-step21-video-input/` | 完整视频输入 |
| Step 22 | `baseline_runs/20260502-step22-full-smoke/` | required smoke matrix |
| Step 23C | `baseline_runs/20260502-step23c-prompt-smoke/` | prompt 贯通后的当前输出 |
| Step 24H | `baseline_runs/20260503-step24h-verify/` | 代码整理冻结验证，`pp-mm` / `tp-mm` / `hybrid-mm` 子集 |

## 关键效果

| 修改 | before | after |
| --- | ---: | ---: |
| startup contract 移除 `stage_output` | `7,563,328` bytes | `4,353,088` bytes |
| startup contract 移除 dense derived tensor | `4,353,088` bytes | `3,245,806` bytes |
| HYBRID stage1 `tp_degree=1` collective | `648.46 MiB` | `0 B` |
| pure TP comm dtype | `449.12 MiB` collective | `221.48 MiB` collective |
| pure TP runtime input broadcast | `4` events / rank | `0` events |
| Step 20C opt-in compaction | full active visual KV | active KV bytes 约减半 |
| Step 23C prompt | distributed mm 用默认视频 prompt | 使用 CLI `--prompt`，与 HF-mm 对齐 |

## 下一步

### 24. 代码整理

目标：在继续加性能功能前，把主路径代码和 debug/replay 边界再收窄一轮。

整理标准：借鉴优秀论文项目仓库，做到所见即所得：

- README 里看到的命令可以直接跑。
- 命令旁边写清楚会生成哪些文件、期望输出是什么。
- 文档中的模块名、函数名、CLI 参数名和代码里完全一致。
- 结果表只放当前可信 baseline，历史结果标清楚“历史 baseline”。
- 代码入口、核心模块、debug/replay 路径一眼能分清。

产出：

- 主路径命名复查：`StageState`、`model_input`、`runtime_inputs` 的边界保持清楚。
- 函数名 / 类名复查：主路径命名简单直接，和 vLLM-style 术语尽量对齐。
- 文件归属复查：执行逻辑、runtime builder、schema、transport、debug/replay helper 放在对应模块。
- 大文件 import 收口：优先整理 `hybrid_parallel.py` / `pipeline_parallel.py` / `tensor_parallel.py` 这类 backend 文件，避免顶部几十个函数级 import 淹没主逻辑。
- 类化边界评估：只把有稳定状态和清晰职责的逻辑类化，例如 runner/context/transport/session；不要为了减少 import 行数强行包类。
- 模块 facade 评估：对纯函数工具优先用模块别名或小 facade 收口，例如 `distributed as dist`、`tensor_parallel as tp_backend`，让调用点更能显示来源。
- 合并重复 helper：只合并行为一致、边界清楚、测试能覆盖的重复逻辑。
- 删除冗余代码：移除未引用的临时 helper、过期 alias、重复 wrapper；保留必要 legacy compat。
- debug/replay/capture 代码隔离：旧 `bundle` 命名只留在这些路径或 legacy compat。
- helper script 参数对齐：README / helper / smoke matrix 使用同一套参数名。
- README 重排成论文项目式结构：Overview、Quickstart、Reproduce、Results、Architecture、Code Map、Debug。

验收：

- `rg "bundle|runtime_input|model_input|manifest-path|allow-debug-paths"` 的结果有明确归类。
- `rg` 能说明被保留的 legacy/debug 名称为什么保留。
- backend 大文件的 import 区域明显变短，依赖来源更清楚。
- 新增类必须有明确状态和职责，不能只是函数容器。
- 删除/合并的函数有调用点审计，不能留下 dead import 或破坏外部 compat。
- README 中每条主命令都有对应脚本/参数/产物说明。
- Code Map 中列出的文件真实存在，并且职责和文件内容一致。
- 本地单测和 `run-runtime-core-regression.sh --skip-baseline-checks` 通过。
- 不改变 generated ids/text、payload schema 和 baseline 语义。

实现顺序：

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| 24A audit | 已完成初版：见 `CODE_CLEANUP_AUDIT.md`，统计大文件行数、import 数、公开函数数；列出 `bundle/runtime_input/model_input/debug` 命名归类 | 有 `文件 / 问题 / 是否主路径 / 建议动作 / 风险` 表 |
| 24B backend import 收口 | 已完成：`hybrid_parallel.py` 91 -> 61，`pipeline_parallel.py` 90 -> 58，`tensor_parallel.py` 68 -> 57；统一使用 `*_backend` / `qwen_*` facade | import 区明显变短；调用来源更清楚；单测过 |
| 24C 类化边界评估 | 已完成：见 `CODE_CLEANUP_AUDIT.md`；本轮不新增类，只保留现有 runner / transport / builder / cache 边界 | 有 `候选类 / 当前函数 / 状态字段 / 收益 / 风险 / 是否本轮做` 表；纯函数 helper 不强行包类 |
| 24D 主路径命名清理 | 已完成：见 `CODE_CLEANUP_AUDIT.md`；direct builder 不再显式传 legacy `bundle_path=None` / `bundle_dir=None`，`StageState/model_input/runtime_inputs` 边界已冻结 | execution 主路径无 `bundle`；legacy/debug/replay/capture/weight-loader 保留项有解释 |
| 24E 文件归属整理 | 已完成：见 `CODE_CLEANUP_AUDIT.md`；README `Code Map` 已对齐 backend、distributed、transport、execution、runtime_builder、debug/capture/replay/live/scripts 真实职责 | Code Map 和实际文件职责一致；本轮不搬文件、不改 import 路径 |
| 24F 合并/删除冗余 | 已完成：见 `CODE_CLEANUP_AUDIT.md`；只删除 HYBRID 明确未使用的 `get_stage_output/run_stage_tp` import，legacy compat 和 replay helper 保留并归类 | 无 backend dead import；compat 不破坏；相关测试通过 |
| 24G README 所见即所得 | 已完成：README 重排为 `Overview / Quickstart / Reproduce / Results / Architecture / Code Map / Debug`；主命令补齐 inputs、outputs、expected、troubleshoot | 每条主命令有输入、输出文件、expected output、失败排查入口 |
| 24H 验证冻结 | 已完成：本地回归通过；真实 Jetson `pp-mm/tp-mm/hybrid-mm` 子集冻结到 `baseline_runs/20260503-step24h-verify/` | `git diff --check`、本地回归、checker/perf table 通过 |
| 24I KV cache 包拆分 | 已完成：`StageKVCache`、video window metadata、video KV compression 从 `execution/` 拆到 `models/qwen3vl/kv_cache/` | 代码归属更清楚；backend/execution re-export 不变；不改运行语义 |
| 24J 全仓冗余扫描 | 已完成第一轮：删除生成缓存和少量 dead import；re-export / compat shim / debug replay API 明确保留 | `CODE_CLEANUP_AUDIT.md` 有删除项和保留项归类；不改运行语义 |
| 24K KV 子模块兼容层删除 | 已完成：删除 `execution/kv_cache.py`、`execution/video_window_cache.py`、`execution/video_kv_compression.py`；只保留 `execution.__init__` 顶层 re-export | 项目内无旧子模块调用点；KV 实现只在 `models/qwen3vl/kv_cache/` |

### 25. 后续性能候选

暂不推进。当前优先级继续放在代码脚手架整理，性能候选先保留为后续备忘。

| 候选 | 当前判断 |
| --- | --- |
| TP collective | Gloo 是主要耗时；继续观察 RowParallel attention/MLP 是否有重复 collective |
| PP handoff overlap | 可评估，但需要保持 correctness guard |
| stage partition 搜索 | 等代码整理完成后再排优先级 |
| `live-mm-generate` 完整视频卡顿 | 独立定位，不阻塞主 runtime |

## 暂不推进

| 方向 | 原因 |
| --- | --- |
| 20D 历史窗口检索回取 | KV cache 管理先冻结到 20C-4 |
| 默认启用 video KV compression | 还缺更长视频和更多问题质量评估 |
| vLLM-style serving engine | 当前目标不是 serving 系统 |
| BlockPool / prefix cache / scheduler | 暂不引入 serving 复杂度 |
| 远端 dense KV 回取 | 风险高，容易破坏 correctness guard |

## 固定规则

- 主路径对象叫 `StageState`。
- `bundle` 只保留给 replay/debug/capture/legacy compat。
- 内部 helper 可以用 `model_input`；wire protocol 继续保留 `runtime_inputs`。
- `hexgen_core/modules/` 只放 `pipeline_parallel.py`、`tensor_parallel.py`、`hybrid_parallel.py`。
- `models/qwen3vl/kv_cache/` 只放 KV buffer、video window index、video KV compression/compaction。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。
- payload/transport 改动必须记录 before/after keys、tensor count、bytes。
- runtime 主路径改动至少验证 generated ids/text、transport bytes、CUDA peak、weight shard scope。

## 常用命令

```bash
# 本地最小回归
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh

# Step 22 checker
PYTHONPATH=. python qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --matrix step22 \
  --baseline-dir baseline_runs/<new-step22-dir> \
  --require-transport-metrics

# 同步 Jetson
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```
