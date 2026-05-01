# Step 20C-1 Video KV Selector

日期：2026-05-01

目标：实现 planner-only `uniform` / `swa` token selector。默认 `--video-kv-compression none` 不变；opt-in 只记录 selected token stats，不压缩、不删除、不回取 KV。

## 运行命令

TP：

```bash
NODE_RANK=0 NNODES=2 MASTER_ADDR=10.126.126.3 MASTER_PORT=29651 \
  CASE_ID=tp-mm-generate-step20c1-uniform-j23 \
  OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/20260501-step20c1-selector \
  bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh \
  --video-kv-compression uniform --video-kv-keep-ratio 0.5
```

HYBRID：

```bash
NODE_RANK=0 NNODES=3 MASTER_ADDR=10.126.126.3 MASTER_PORT=29653 \
  CASE_ID=hybrid-mm-generate-step20c1-uniform-j23shared \
  OUT=/mnt/ssd/code/Qwen3_vl/baseline_runs/20260501-step20c1-selector \
  bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh \
  --video-kv-compression uniform --video-kv-keep-ratio 0.5
```

## 结果

| case | rank | method | original/keep/drop tokens | keep/savable KV bytes | generated |
| --- | ---: | --- | --- | ---: | --- |
| `tp-mm-generate-step20c1-uniform-j23` | 0 | `uniform` | `576 / 288 / 288` | `21,233,664 / 21,233,664` | pass |
| `tp-mm-generate-step20c1-uniform-j23` | 1 | `uniform` | `576 / 288 / 288` | `21,233,664 / 21,233,664` | pass |
| `hybrid-mm-generate-step20c1-uniform-j23shared` | 0 | `uniform` | `576 / 288 / 288` | `10,616,832 / 10,616,832` | pass |
| `hybrid-mm-generate-step20c1-uniform-j23shared` | 1 | `uniform` | `576 / 288 / 288` | `10,616,832 / 10,616,832` | pass |
| `hybrid-mm-generate-step20c1-uniform-j23shared` | 2 | `uniform` | `576 / 288 / 288` | `21,233,664 / 21,233,664` | pass |

固定输出：

- generated ids：`[87140, 15946, 3837, 101177]`
- generated text：`视频中，一名`

Performance：见 `runtime-perf-table.md`。

## 结论

- `video_kv_compression_plan.selector_enabled=true`。
- `selected_token_sample` 和 `selected_token_ranges` 已进入 rank log。
- `uniform` 真实 Jetson TP/HYBRID smoke 通过。
- `swa` selector 由 `test/test_video_kv_compression.py` 覆盖：保留窗口尾部 recent tokens。
- HYBRID stage-group runtime input schema 已允许 selector 标量字段；`runtime_inputs_meta` 从 `796` bytes 增到 `860` bytes，tensor bytes 不变。
- 仍是 planner-only：`mutates_kv=false`、`compression_enabled=false`，attention/KV 路径不变。
