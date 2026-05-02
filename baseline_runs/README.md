# baseline_runs

这里只保留当前回归和汇报还会用到的 baseline。更早的临时 profile 已移出项目目录。

## 保留目录

| 目录 | 用途 |
| --- | --- |
| `20260428/` | correctness baseline，固定 generated ids/text |
| `20260430-bfloat16-default/` | 当前性能 baseline |
| `20260430-step15-derived-rebuild/` | Step 15 payload baseline |
| `20260501-step16-pinned-ab/` | Step 16 pinned memory A/B |
| `20260501-step20a-kv-cache-smoke/` | Step 20A `StageKVCache` smoke |
| `20260501-step20a-kv-cache-long-decode/` | Step 20A long decode profile |
| `20260501-step20b-video-window-cache/` | Step 20B video window metadata |
| `20260501-step20c0-video-kv-plan/` | Step 20C-0 planner |
| `20260501-step20c1-selector/` | Step 20C-1 selector stats |
| `20260502-step20c3-compaction/` | Step 20C-3 `uniform` compaction |
| `20260502-step20c4-infinipot-selector/` | Step 20C-4 `infinipot-v` selector |
| `20260502-step21-video-input/` | Step 21 完整视频输入 smoke |
| `20260502-step22-2node-smoke/` | Step 22 两节点 smoke matrix 子集，checker/perf table 产物完整 |

## 已移出

旧 profile 已从 `baseline_runs/` 移到：

```text
/tmp/qwen3vl_removed_baseline_runs_20260502/
```

这些目录已由当前 baseline 或后续 step 替代；需要恢复时从上面的临时目录移回即可。
