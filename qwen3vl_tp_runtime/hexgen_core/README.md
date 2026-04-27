# `hexgen_core`

这一层放的是运行时内核，而不是具体命令入口。

当前目录按 HexGen / Jupiter 风格分成几类：

- `modules/`: 并行运行时主入口
  - `tensor_parallel.py`: `backend=tp` 的 standalone direct 入口，主公开名是 `TensorParallelRunner / run_tensor_parallel_rank / run_stage_state_tp`，负责单 stage TP 校验、TP rank 入口、以及 TP debug replay 兼容
  - `pipeline_parallel.py`: `backend=pp` 的纯 PP stage runtime
  - `hybrid_parallel.py`: `backend=hybrid` 的 PP + TP hybrid runtime
- `config.py`: 默认路径、replay bundle / manifest 常量
- `distributed.py`: 最小分布式初始化和 CPU/gloo 通信辅助
- `gen_hetero_groups.py`: HexGen 风格 stage/tp/pp rank group 生成
- `gen_p2p_lists.py`: p2p send/recv lane 生成辅助
- `schema.py`: manifest、layout、handoff payload、rank context 的 dataclass
- `stage.py`: `StageState` view、handoff 构造与 stage replay
- `transport.py`: 多 tensor multimodal payload 的 stage handoff 通信
- `__init__.py`: runtime-facing 聚合导出

模型相关实现不放在这里，而是放在 `../models/qwen3vl/`，这样可以把 runtime 结构和模型适配层拆开。
