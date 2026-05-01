# Step 16 Buffer Reuse Audit

范围：只看 generate 主路径里的 `hidden`、`handoff`、`decode` tensor 分配和 copy。`capture`、`replay`、`live correctness` 路径不作为本轮优化目标。

vLLM 对照：

- vLLM 的 serving 方向把 KV cache 做成 block/page 管理，避免每步重新拼一整段连续 KV。
- CPU offload / host-device copy 优化是独立层；我们当前 Jetson/Gloo 路径已做成 opt-in pinned buffer 实验，不默认启用。
- 因此本轮先做低风险临时 tensor / transport buffer 减量，KV cache manager 放到 step 16 之后。

参考：

- vLLM `CacheConfig` / KV cache：<https://docs.vllm.ai/en/stable/api/vllm/config/cache/>
- vLLM offload config：<https://docs.vllm.ai/en/v0.17.1/api/vllm/config/>

## 盘点表

| 位置 | 当前分配 / copy | 分类 | 结论 |
| --- | --- | --- | --- |
| `live/common.py:_runtime_tensor` | `tensor.detach().to(device)`，必要时再 `.to(dtype)` | 必须保留 | StageState materialize 的统一入口，暂不改语义；后续可减少重复调用。 |
| `runtime_builder.py:_clone_mm_*_to_cpu` | startup shared / handoff / visuals clone 到 CPU | transport 临时 | startup contract 需要 CPU payload；可作为 pinned transport buffer 实验对象。 |
| `runtime_builder.py` prefill boundary capture | stage boundary `hidden_states.detach().clone()` | 必须保留 | 当前 startup contract 和 reference boundary 依赖这些稳定快照，暂不删。 |
| `runtime_builder.py` file-backed/reference generate | 每层 hidden、decode state、logits 多处 `.detach().clone()` | 非主优化目标 | 主要是 reference/debug 状态保存，不作为 step 16 第一刀。 |
| `stage.py:build_stage_handoff_payload` | 直接引用 `stage_output` 组 handoff | 必须保留 | 本身不 clone；真正 CPU copy 在 transport 层发生。 |
| `transport.py:send_payload/recv_payload` | send: device -> CPU contiguous；recv: CPU empty -> device | 已支持 opt-in | `--transport-pin-memory` 下 best-effort pinned CPU staging，不改 payload schema。 |
| `transport.py:broadcast_payload` | leader tensor -> CPU；followers `torch.empty` 后 broadcast，再 `.to(device)` | 已支持 opt-in | HYBRID model input/scaffold broadcast 可用 pinned CPU staging。 |
| `distributed.py:all_reduce_cpu/all_gather_cpu/broadcast_cpu` | device -> CPU -> Gloo -> device | 已支持 opt-in | TP collective profile 记录 pinned requested/used；默认关闭。 |
| PP / TP / HYBRID runtime-only decode loop | 每 step `torch.cat([mask, ones])` 增长 attention mask | 已复用 | `generate_buffers.py` 预分配 max length mask，每步取 view。 |
| PP / TP / HYBRID decode input | 每 step 新建 `torch.tensor([[token]])` | 已复用 | 复用固定 shape token buffer，按 step 写入 token。 |
| runtime-only multimodal decode helper | 每 step 构造 dummy `decode_input_ids` / dummy embedding weight | 已复用 | `decode_input_ids` 和 dummy embedding weight 在 generate loop 外分配。 |
| `execution/stages.py` cache update | 每层 `full_key/full_value.detach().clone()` | 必须保留 | 当前 cache 生命周期靠 clone 保证稳定；等 KV cache manager 再改。 |
| `execution/attention.py:_concat_past_key_value` | decode 时 `torch.cat([past, current])` 生成完整 KV | 必须保留 | 这是后续 KV cache session/paged cache 的核心目标，不在 step 16 直接改。 |
| `runtime_mm_stage.py` local rebuild | 本地重建 attention mask / RoPE / decode embeds | 必须保留 | 属于 Step 15 后的正确性路径；只能做局部缓存，不能恢复传 dense tensor。 |

## 第一刀结果

已完成：

- 新增 `hexgen_core/generate_buffers.py`：
  - `build_decode_attention_mask_buffer`
  - `decode_attention_mask_view`
  - `fill_decode_input_ids`
- `pipeline_parallel.py`、`tensor_parallel.py`、`hybrid_parallel.py` 的 runtime-only generate decode loop 已复用：
  - attention mask buffer
  - `(batch, 1)` decode token buffer
  - multimodal dummy embedding weight

效果边界：

- 不改变 payload key、tensor count、bytes。
- 不改变 stage/handoff 语义。
- 只减少 decode loop 小 tensor 分配；真实 CUDA peak / elapsed 需要 Jetson smoke 复测。

已验证：

- `python -m py_compile qwen3vl_tp_runtime/hexgen_core/generate_buffers.py qwen3vl_tp_runtime/hexgen_core/modules/pipeline_parallel.py qwen3vl_tp_runtime/hexgen_core/modules/tensor_parallel.py qwen3vl_tp_runtime/hexgen_core/modules/hybrid_parallel.py`
- `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --skip-baseline-checks`

## 第二刀结果

已完成：

- 新增 CLI：`--transport-pin-memory`。
- 新增 runtime 开关：默认关闭，也可通过 `HEXGEN_TRANSPORT_PIN_MEMORY=1` 初始化。
- 覆盖：
  - `transport.py:send_payload / recv_payload / broadcast_payload`
  - `distributed.py:all_reduce_cpu / all_gather_cpu / broadcast_cpu`
- profile 新增：
  - 每个 event 的 `transport_pin_memory_requested`
  - 每个 event 的 `transport_pin_memory_used`
  - `runtime_metrics.transport.pin_memory`

效果边界：

- payload keys / tensor count / bytes 不变。
- Gloo collective 语义不变。
- pinned allocation 是 best-effort；环境不支持时自动退回普通 CPU tensor。
- Jetson A/B 已完成，收益偏小，保持默认关闭。

真实 A/B：

- 目录：`baseline_runs/20260501-step16-pinned-ab/`。
- `tp-mm-generate` generated ids/text 不变，payload bytes 不变。
- total time：`53.47 / 53.21s -> 53.01 / 52.97s`。
- TP collective time：`24.34 / 23.76s -> 23.91 / 23.51s`。
- CUDA peak allocated 不上升；rank0 reserved 约多 `2 MiB`。
- HYBRID A/B 因 rank0/rank1 共用 jetson2，只作为功能性验证。

## Step 16 结论

- decode loop 小 buffer 复用可以保留，风险低。
- `--transport-pin-memory` 可以保留为 opt-in 调参开关，暂不默认启用。
- `full_key/full_value.clone()` 和 `torch.cat([past, current])` 不在本步继续改，交给下一阶段 KV cache manager。
