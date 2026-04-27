# qwen3vl_tp_runtime Roadmap

## 目标

- `TP` 不能再是“每张卡先拿完整权重，再在计算时按 rank 切一刀”，而要改成“每张卡只拿自己那份权重”。
- `PP / TP / hybrid` 主运行路径不能再默认依赖 `bundle / manifest replay` 这类准备物，而要在启动时直接从 `model_path` 构建每个 `stage/rank` 的运行参数。
- 在上面两条主目标基础上，继续把 multimodal direct runtime 收口到更彻底的 `stage-only / shard-only` 形态。
- 当前主线已经完成 `direct schema` 与 `replay schema` 的初步分离，并把 package-level compat exports 收紧成：`__all__` 只代表 direct 主路径，legacy replay/capture 入口单独挂在 `LEGACY_*_EXPORTS` 下。
- 当前已继续把 concrete runtime modules 的导出面收紧：`pipeline_parallel / hybrid_parallel` 的 `__all__` 只保留 direct runner/loader，`prepare_* / load_*_manifest` 进入 `LEGACY_REPLAY_EXPORTS`；纯 bundle TP replay 进入 `DEBUG_REPLAY_EXPORTS`。
- 当前已把 `load_pipeline_manifest / load_hybrid_manifest` 从主 runtime import 面移出，`--manifest-path` 只通过 `runtime_replay` debug helper 进入 manifest replay。
- 当前 `pp / hybrid multimodal generate` 真实 smoke 已通过，`token_match=true`，可以把这轮 `schema / legacy / debug-only transport` cleanup 判定为过线。
- 当前已开始 `TP` 本地分片加载收口：`TextDecoderStageWeightPlan / load_text_decoder_stage_weight_bundle` 已经明确区分 TP 必须分片读取的 q/k/v/o、MLP 投影参数，以及仍需复制的 embedding、norm、bias、lm_head 等参数；TP 分片模式下缺少必要 `tensor_slices` 会直接报错。
- 当前已确认 `materialize_text_stage_bundle -> backend=tp|hybrid` direct runtime 链路：`backend=tp` 走 hybrid-family direct manifest；所有 direct `tp_degree > 1` stage 先广播无权重 scaffold，再由每个 rank 用本地 `tp_shard_rank/tp_shard_world_size` materialize 自己的 shard。若 materialize 出来的 bundle 不是 `tp_weight_sharded=True`，会直接报错。
- 当前 `backend=tp` 与 `backend=hybrid` 的真实 text generate smoke 已通过，确认 shard-only 权重路径可以稳定跑通。
- 当前 hybrid-family runtime summary 已输出 `weight_load`，用于记录每个 rank 的 `tp_weight_sharded / tp_shard_rank / tp_shard_world_size`、本地权重 tensor 数量/字节数，以及 TP 分片参数与复制参数计数，后续 smoke 不再只依赖启动日志肉眼判断。
- 当前 `backend=hybrid` text generate 新版 summary smoke 已通过：stage0 的两个 TP rank 分别显示 `tp_shard_rank=0/2` 和 `1/2`，stage1 单卡显示 `tp_weight_sharded=false` 且只加载 `18:35 + final_norm/lm_head`。
- 当前 `backend=tp` text generate 新版 summary smoke 已通过：两个 TP rank 分别显示 `tp_shard_rank=0/2` 和 `1/2`，`loaded_weight_tensor_bytes` 完全一致，确认整段 text decoder 在 TP 模式下走 rank-local shard materialize。

## 下一步

- 保持 `pp / hybrid multimodal` smoke 作为回归门槛，确保 multimodal direct 路径仍然 `token_match=true`。
- 补一轮启动时间 / 峰值显存记录，确认 TP rank 不再持有完整 decoder 大矩阵，PP rank 不再加载无关 stage 权重。
- 之后再考虑 embedding / lm_head 的进一步并行化；在实现 vocab/embedding parallel 前，这两类权重仍按执行语义复制。
