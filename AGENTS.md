# Agent 指令

本仓库是面向 Qwen3-VL 的 correctness-first 分布式推理 runtime 原型。

修改代码前，默认先读 `qwen3vl_tp_runtime/SESSION_HANDOFF.md`。如果任务涉及架构、runtime 行为、验收或命名，还要同时查看 `qwen3vl_tp_runtime/README.md`、`qwen3vl_tp_runtime/ROADMAP.md` 和 `qwen3vl_tp_runtime/BASELINE.md`。

## 项目规则

- 保持架构方向清晰：`PP` 和 `TP` 是基础后端，`HYBRID` 是建立在二者之上的组合层。
- 主运行路径对象统一使用 `StageState`。`bundle` 术语只保留给 replay、debug 和 capture 路径。
- 修改 runtime 行为时，按 `BASELINE.md` 和 `ROADMAP.md` 中的验收口径验证。
- 对 startup contract、scaffold broadcast、transport payload 等敏感 payload 路径，修改前或最终交付前必须列出 before/after bytes、payload keys 和 tensor count。
- 不做顺手重构。改动范围应贴合用户请求，避免触碰无关代码。

## 工作方式

- 提出抽象前，优先阅读现有代码和本地文档。
- 做小而可验证的改动。
- 除非当前任务明确要求重命名或调整边界，否则保持现有命名和模块边界。
- 不回滚或覆盖用户改动。
- 根据改动面选择聚焦的测试或 smoke case。
