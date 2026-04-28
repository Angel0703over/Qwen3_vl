#!/usr/bin/env python3
"""Check frozen baseline rank logs for distributed runtime cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BaselineCheckError(AssertionError):
    pass


@dataclass(frozen=True)
class RankSummary:
    path: Path
    log_text: str
    summary: dict[str, Any]

    @property
    def label(self) -> str:
        rank = self.summary.get("rank")
        if rank is None:
            return str(self.path)
        return f"rank{rank} ({self.path})"


def _iter_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            return objects
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        cursor = start + max(end, 1)


def extract_last_json_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return extract_last_json_summary_from_text(path, text)


def extract_last_json_summary_from_text(path: Path, text: str) -> dict[str, Any]:
    objects = _iter_json_objects(text)
    for obj in reversed(objects):
        if "generated_token_ids" in obj or "runtime_generated_token_ids" in obj:
            return obj
    raise BaselineCheckError(f"{path}: no JSON runtime summary with generated ids found")


def _fail(errors: list[str], label: str, field: str, expected: Any, actual: Any) -> None:
    errors.append(f"{label}: {field} expected {expected!r}, got {actual!r}")


def _require_equal(errors: list[str], item: RankSummary, field: str, expected: Any) -> None:
    actual = _get_path(item.summary, field)
    if actual != expected:
        _fail(errors, item.label, field, expected, actual)


def _require_true(errors: list[str], item: RankSummary, field: str) -> None:
    actual = _get_path(item.summary, field)
    if actual is not True:
        _fail(errors, item.label, field, True, actual)


def _require_frontend_mode(errors: list[str], item: RankSummary, expected: str) -> None:
    actual = _get_path(item.summary, "weight_load.multimodal_frontend_mode")
    if actual is None:
        needle = f"multimodal_frontend_mode={expected}"
        if needle in item.log_text:
            return
    if actual != expected:
        _fail(errors, item.label, "weight_load.multimodal_frontend_mode", expected, actual)


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _check_common(case_id: str, items: list[RankSummary]) -> list[str]:
    errors: list[str] = []
    first_ids = items[0].summary.get("generated_token_ids")
    first_text = items[0].summary.get("generated_text")
    if not isinstance(first_ids, list) or not first_ids:
        errors.append(f"{items[0].label}: generated_token_ids missing or empty")
    if first_text is None:
        errors.append(f"{items[0].label}: generated_text missing")

    for item in items:
        summary = item.summary
        if summary.get("generated_token_ids") != first_ids:
            _fail(errors, item.label, "generated_token_ids", first_ids, summary.get("generated_token_ids"))
        if summary.get("generated_text") != first_text:
            _fail(errors, item.label, "generated_text", first_text, summary.get("generated_text"))
        if "token_match" in summary and summary.get("token_match") is not True:
            _fail(errors, item.label, "token_match", True, summary.get("token_match"))

    expected_backend = case_id.split("-", 1)[0]
    backend_aliases = {"pp": "pp", "tp": "tp", "hybrid": "hybrid"}
    expected_backend = backend_aliases.get(expected_backend)
    if expected_backend is not None:
        for item in items:
            if item.summary.get("backend") != expected_backend:
                _fail(errors, item.label, "backend", expected_backend, item.summary.get("backend"))
    return errors


def _check_tp(case_id: str, items: list[RankSummary]) -> list[str]:
    errors = _check_common(case_id, items)
    ranks = sorted(item.summary.get("rank") for item in items)
    expected_world = len(items)
    if ranks != list(range(expected_world)):
        errors.append(f"{case_id}: expected ranks {list(range(expected_world))}, got {ranks}")
    for item in items:
        rank = item.summary.get("rank")
        _require_true(errors, item, "weight_load.tp_weight_sharded")
        _require_equal(errors, item, "weight_load.tp_shard_rank", rank)
        _require_equal(errors, item, "weight_load.tp_shard_world_size", expected_world)
        _require_true(errors, item, "weight_load.tp_shard_shape_ok")
        if _get_path(item.summary, "weight_load.tp_stage_loaded_weight_tensor_bytes_equal") is not None:
            _require_true(errors, item, "weight_load.tp_stage_loaded_weight_tensor_bytes_equal")
        if _get_path(item.summary, "weight_load.stage_weight_scope_ok") is not None:
            _require_true(errors, item, "weight_load.stage_weight_scope_ok")
    byte_values = [item.summary.get("weight_load", {}).get("loaded_weight_tensor_bytes") for item in items]
    if len(set(byte_values)) != 1:
        errors.append(f"{case_id}: loaded_weight_tensor_bytes differ across TP ranks: {byte_values!r}")
    return errors


def _check_pp(case_id: str, items: list[RankSummary]) -> list[str]:
    errors = _check_common(case_id, items)
    by_stage = {item.summary.get("stage_idx"): item for item in items}
    expected_stages = list(range(len(items)))
    if sorted(by_stage) != expected_stages:
        errors.append(f"{case_id}: expected stages {expected_stages}, got {sorted(by_stage)}")

    for stage_idx, item in by_stage.items():
        _require_true(errors, item, "weight_load.stage_weight_scope_ok")
        _require_equal(errors, item, "start_idx", _get_path(item.summary, "weight_load.stage_start_idx"))
        _require_equal(errors, item, "end_idx", _get_path(item.summary, "weight_load.stage_end_idx"))
        if case_id.endswith("-mm-generate"):
            if stage_idx == 0:
                _require_frontend_mode(errors, item, "active")
                _require_equal(errors, item, "weight_load.loaded_top_level_weight_names", ["embed_tokens_weight"])
            elif stage_idx == len(items) - 1:
                _require_frontend_mode(errors, item, "consume-only")
                _require_equal(
                    errors,
                    item,
                    "weight_load.loaded_top_level_weight_names",
                    ["final_norm_weight", "lm_head_weight"],
                )
    return errors


def _check_hybrid(case_id: str, items: list[RankSummary]) -> list[str]:
    errors = _check_common(case_id, items)
    stage0_items = sorted(
        [item for item in items if item.summary.get("stage_idx") == 0],
        key=lambda item: item.summary.get("local_rank", -1),
    )
    stage1_items = [item for item in items if item.summary.get("stage_idx") == 1]
    if len(stage0_items) != 2 or len(stage1_items) != 1:
        errors.append(
            f"{case_id}: expected 2 ranks in stage0 and 1 rank in stage1, "
            f"got stage0={len(stage0_items)} stage1={len(stage1_items)}"
        )
        return errors

    for local_rank, item in enumerate(stage0_items):
        _require_equal(errors, item, "local_rank", local_rank)
        _require_equal(errors, item, "tp_degree", 2)
        _require_true(errors, item, "weight_load.tp_weight_sharded")
        _require_equal(errors, item, "weight_load.tp_shard_rank", local_rank)
        _require_equal(errors, item, "weight_load.tp_shard_world_size", 2)
        _require_true(errors, item, "weight_load.tp_shard_shape_ok")
        _require_true(errors, item, "weight_load.tp_stage_loaded_weight_tensor_bytes_equal")
        _require_true(errors, item, "weight_load.stage_weight_scope_ok")
        if case_id.endswith("-mm-generate"):
            _require_equal(errors, item, "weight_load.loaded_top_level_weight_names", ["embed_tokens_weight"])

    stage1 = stage1_items[0]
    _require_equal(errors, stage1, "local_rank", 0)
    _require_equal(errors, stage1, "tp_degree", 1)
    _require_equal(errors, stage1, "weight_load.tp_weight_sharded", False)
    _require_true(errors, stage1, "weight_load.stage_weight_scope_ok")
    _require_equal(errors, stage1, "weight_load.loaded_top_level_weight_names", ["final_norm_weight", "lm_head_weight"])
    if case_id.endswith("-mm-generate"):
        _require_frontend_mode(errors, stage1, "consume-only")
    return errors


def check_baseline_logs(case_id: str, paths: list[Path]) -> list[RankSummary]:
    if not paths:
        raise BaselineCheckError("at least one log path is required")
    items = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        items.append(
            RankSummary(
                path=path,
                log_text=text,
                summary=extract_last_json_summary_from_text(path, text),
            )
        )
    case_prefix = case_id.split("-", 1)[0]
    if case_prefix == "tp":
        errors = _check_tp(case_id, items)
    elif case_prefix == "pp":
        errors = _check_pp(case_id, items)
    elif case_prefix == "hybrid":
        errors = _check_hybrid(case_id, items)
    else:
        errors = _check_common(case_id, items)
    if errors:
        raise BaselineCheckError("\n".join(errors))
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Rank log files to check")
    parser.add_argument("--case-id", required=True, help="Baseline case id, e.g. tp-mm-generate")
    args = parser.parse_args(argv)

    items = check_baseline_logs(args.case_id, args.logs)
    first = items[0].summary
    print(
        "PASS "
        f"case={args.case_id} ranks={len(items)} "
        f"generated_token_ids={first.get('generated_token_ids')!r} "
        f"generated_text={first.get('generated_text')!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
