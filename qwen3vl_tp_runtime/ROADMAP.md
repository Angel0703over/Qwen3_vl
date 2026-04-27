# qwen3vl_tp_runtime Roadmap

## 目标

- `TP` 不能再是“每张卡先拿完整权重，再在计算时按 rank 切一刀”，而要改成“每张卡只拿自己那份权重”。
- `PP / TP / hybrid` 主运行路径不能再默认依赖 `bundle / manifest replay` 这类准备物，而要在启动时直接从 `model_path` 构建每个 `stage/rank` 的运行参数。
- 在上面两条主目标基础上，继续把 multimodal direct runtime 收口到更彻底的 `stage-only / shard-only` 形态。
- 当前主线已经完成 `direct schema` 与 `replay schema` 的初步分离，并把 package-level compat exports 收紧成：`__all__` 只代表 direct 主路径，legacy replay/capture 入口单独挂在 `LEGACY_*_EXPORTS` 下。
- 当前已继续把 concrete runtime modules 的导出面收紧：`pipeline_parallel / hybrid_parallel` 的 `__all__` 只保留 direct runner/loader，`prepare_* / load_*_manifest` 进入 `LEGACY_REPLAY_EXPORTS`；纯 bundle TP replay 进入 `DEBUG_REPLAY_EXPORTS`。
- 当前已把 `load_pipeline_manifest / load_hybrid_manifest` 从主 runtime import 面移出，`--manifest-path` 只通过 `runtime_replay` debug helper 进入 manifest replay。
- 当前 `pp / hybrid multimodal generate` runtime-only 主路径真实 smoke 已通过：所有 rank 的 `generated_token_ids` 一致，summary 能证明 `stage0` 负责 frontend + 自己的 stage/shard，non-stage0 只消费 handoff/metadata 且只加载自己的 stage/shard。主路径 `include_runtime_reference=false`，因此不再携带 full reference token/bundle。
- 当前已开始 `TP` 本地分片加载收口：`TextDecoderStageWeightPlan / load_text_decoder_stage_weight_bundle` 已经明确区分 TP 必须分片读取的 q/k/v/o、MLP 投影参数，以及仍需复制的 embedding、norm、bias、lm_head 等参数；TP 分片模式下缺少必要 `tensor_slices` 会直接报错。
- 当前已确认 `materialize_text_stage_bundle -> backend=tp|hybrid` direct runtime 链路：`backend=tp` 走 hybrid-family direct manifest；所有 direct `tp_degree > 1` stage 先广播无权重 scaffold，再由每个 rank 用本地 `tp_shard_rank/tp_shard_world_size` materialize 自己的 shard。若 materialize 出来的 bundle 不是 `tp_weight_sharded=True`，会直接报错。
- 当前 `backend=tp` 与 `backend=hybrid` 的真实 text generate smoke 已通过，确认 shard-only 权重路径可以稳定跑通。
- 当前 hybrid-family runtime summary 已输出 `weight_load`，用于记录每个 rank 的 `tp_weight_sharded / tp_shard_rank / tp_shard_world_size`、本地权重 tensor 数量/字节数、TP 分片参数与复制参数计数、TP 投影矩阵 shard shape proof，以及同一 TP stage 内的 `loaded_weight_tensor_bytes` 一致性检查结果，后续 smoke 不再只依赖启动日志肉眼判断。
- 当前 `backend=hybrid` text generate 新版 summary smoke 已通过：stage0 的两个 TP rank 分别显示 `tp_shard_rank=0/2` 和 `1/2`，stage1 单卡显示 `tp_weight_sharded=false` 且只加载 `18:35 + final_norm/lm_head`。
- 当前 `backend=tp` text generate 新版 summary smoke 已通过：两个 TP rank 分别显示 `tp_shard_rank=0/2` 和 `1/2`，`loaded_weight_tensor_bytes` 完全一致，确认整段 text decoder 在 TP 模式下走 rank-local shard materialize。

## Multimodal 过线标准

- 已完成（runtime guard + unit tests）：`non-stage0` 不再跑视觉前端。multimodal direct path 的 non-frontend stage 现在没有 startup contract 会直接报错，不能调用整模 `load_model()`，不能初始化完整 Qwen3-VL，不能跑 image/video processor，只消费 `stage handoff + decoder metadata`。
- 已完成（scope guard + `weight_load` summary）：`PP stage` 只加载自己的层。`weight_load` 现在输出 `stage_start_idx / stage_end_idx / loaded_layer_indices / loaded_top_level_weight_names / stage_weight_scope_ok / unexpected_layer_indices`；例如 `stage_ranges=0:17 18:35` 时，stage0 应只出现 `0..17`，stage1 应只出现 `18..35 + final_norm/lm_head`，如果 bundle 混入无关 layer 会直接报错。
- 已完成（shape guard + TP stage bytes equality）：`TP rank` 只加载自己的 shard。对 `tp_degree=2` 的 stage，rank0/rank1 都必须是 `tp_weight_sharded=true`，分别是 `tp_shard_rank=0/2` 和 `1/2`；`weight_load.tp_shard_shape_ok=true`，`tp_sharded_projection_examples` 能看到 q/k/v/o 与 MLP projection 的 shard 后形状；同一 TP stage 内 `tp_stage_loaded_weight_tensor_bytes_equal=true`，否则启动时直接报错。
- 已完成（thin startup transport + forbidden payload guard）：跨 rank 启动契约保持薄。`multimodal startup transport` 只允许 `shared`、本地 stage 的 `stage_handoffs`、本地 stage 的 `stage_visuals`、`num_frames/frame_paths` 这类必要 metadata/tensor；`root_input / boundaries / hidden_states / replay_bundle / stage_bundle` 等 full/root/replay payload 会被拒绝，`seed_mm_startup_runtime_config()` 也不会写入 `_mm_startup_root_input`。
- 已完成（real Jetson smoke）：`pp multimodal generate` 与 `hybrid multimodal generate` 均稳定通过。验收以 runtime-only 主路径的 rank summary 为准：各 rank `generated_token_ids=[87140, 15946, 3837, 101177]`、`generated_text=视频中，一名`；`pp` 的 stage0 active frontend 且只加载 `0..17 + embed_tokens`，stage1 consume-only 且只加载 `18..35 + final_norm/lm_head`；`hybrid` 的 stage0 两个 TP rank 分别是 `tp_shard_rank=0/2` 与 `1/2` 且 `tp_stage_loaded_weight_tensor_bytes_equal=true`，stage1 consume-only 且只加载自己的 PP stage。
