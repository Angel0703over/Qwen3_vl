#!/usr/bin/env python3
"""Check frozen baseline rank logs for distributed runtime cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen3vl_tp_runtime.scripts.smoke_matrix import SmokeCase, get_smoke_case, iter_smoke_cases


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


def _get_video_input(summary: dict[str, Any]) -> dict[str, Any]:
    direct = summary.get("video_input")
    if isinstance(direct, dict):
        return direct
    prefill = summary.get("prefill")
    if isinstance(prefill, dict) and isinstance(prefill.get("video_input"), dict):
        return prefill["video_input"]
    return {}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_rank_count(case_id: str, items: list[RankSummary], smoke_case: SmokeCase | None) -> list[str]:
    if smoke_case is None or smoke_case.expected_rank_count is None:
        return []
    if len(items) != smoke_case.expected_rank_count:
        return [f"{case_id}: expected {smoke_case.expected_rank_count} rank logs, got {len(items)}"]
    return []


def _check_transport_metrics(
    case_id: str,
    items: list[RankSummary],
    *,
    require_transport_metrics: bool,
) -> list[str]:
    errors: list[str] = []
    required_buckets = ("startup_contract", "scaffold", "stage_handoff", "tp_collective")
    forbidden_payload_fragments = (
        "root_input",
        "boundaries",
        "stage_output",
        "frontend_paths",
        "frame_paths",
        "full_payload",
        "replay",
    )
    for item in items:
        transport = _get_path(item.summary, "runtime_metrics.transport")
        if not isinstance(transport, dict):
            if require_transport_metrics:
                errors.append(f"{item.label}: runtime_metrics.transport missing")
            continue
        totals_by_kind = transport.get("totals_by_kind")
        if not isinstance(totals_by_kind, dict):
            if require_transport_metrics:
                errors.append(f"{item.label}: runtime_metrics.transport.totals_by_kind missing")
            continue
        for kind in required_buckets:
            bucket = totals_by_kind.get(kind)
            if not isinstance(bucket, dict):
                if require_transport_metrics:
                    errors.append(f"{item.label}: transport bucket {kind!r} missing")
                continue
            for field in ("event_count", "elapsed_seconds", "object_bytes", "tensor_bytes", "total_bytes"):
                value = bucket.get(field)
                if value is None:
                    if require_transport_metrics:
                        errors.append(f"{item.label}: transport {kind}.{field} missing")
                    continue
                if not _is_number(value) or value < 0:
                    errors.append(f"{item.label}: transport {kind}.{field} must be non-negative number, got {value!r}")

        events = transport.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                payload_keys = event.get("payload_keys")
                if not isinstance(payload_keys, list):
                    continue
                for key in payload_keys:
                    key_text = str(key)
                    if any(fragment in key_text for fragment in forbidden_payload_fragments):
                        errors.append(f"{item.label}: forbidden transport payload key {key_text!r}")
    return errors


def _check_expected_smoke_case(
    case_id: str,
    items: list[RankSummary],
    smoke_case: SmokeCase | None,
    *,
    require_transport_metrics: bool,
) -> list[str]:
    errors = _check_rank_count(case_id, items, smoke_case)
    if smoke_case is None:
        errors.extend(_check_transport_metrics(case_id, items, require_transport_metrics=require_transport_metrics))
        return errors

    expected_video_source_seen = False
    for item in items:
        if item.summary.get("generated_token_ids") != smoke_case.expected_ids:
            _fail(
                errors,
                item.label,
                "generated_token_ids",
                smoke_case.expected_ids,
                item.summary.get("generated_token_ids"),
            )
        if item.summary.get("generated_text") != smoke_case.expected_text:
            _fail(errors, item.label, "generated_text", smoke_case.expected_text, item.summary.get("generated_text"))
        if smoke_case.backend is not None and item.summary.get("backend") != smoke_case.backend:
            _fail(errors, item.label, "backend", smoke_case.backend, item.summary.get("backend"))
        video_input = _get_video_input(item.summary)
        if smoke_case.expected_video_source is not None and video_input:
            actual_source = video_input.get("source")
            if actual_source != smoke_case.expected_video_source:
                _fail(errors, item.label, "video_input.source", smoke_case.expected_video_source, actual_source)
            else:
                expected_video_source_seen = True
    if smoke_case.expected_video_source is not None and not expected_video_source_seen:
        errors.append(f"{case_id}: no rank reported video_input.source={smoke_case.expected_video_source!r}")
    errors.extend(
        _check_transport_metrics(
            case_id,
            items,
            require_transport_metrics=require_transport_metrics or smoke_case.require_transport_metrics,
        )
    )
    return errors


def _check_multimodal_consume_only(
    case_id: str,
    items: list[RankSummary],
    smoke_case: SmokeCase | None,
    *,
    strict: bool = False,
) -> list[str]:
    if smoke_case is not None:
        is_multimodal = smoke_case.modality == "multimodal"
        require_consume_only = smoke_case.require_consume_only
    else:
        is_multimodal = "-mm-" in case_id
        require_consume_only = False
    if not is_multimodal or not require_consume_only:
        return []
    if not strict:
        return []

    errors: list[str] = []
    for item in items:
        backend = item.summary.get("backend")
        rank = item.summary.get("rank")
        stage_idx = item.summary.get("stage_idx")
        if backend == "tp":
            if rank not in (None, 0):
                _require_frontend_mode(errors, item, "consume-only")
        elif backend in {"pp", "hybrid"}:
            if stage_idx not in (None, 0):
                _require_frontend_mode(errors, item, "consume-only")
    return errors


def _check_common(
    case_id: str,
    items: list[RankSummary],
    *,
    smoke_case: SmokeCase | None = None,
    require_transport_metrics: bool = False,
) -> list[str]:
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
    backend_aliases = {"hf": "hf", "pp": "pp", "tp": "tp", "hybrid": "hybrid"}
    expected_backend = backend_aliases.get(expected_backend)
    if expected_backend is not None:
        for item in items:
            if item.summary.get("backend") != expected_backend:
                _fail(errors, item.label, "backend", expected_backend, item.summary.get("backend"))
    errors.extend(
        _check_expected_smoke_case(
            case_id,
            items,
            smoke_case,
            require_transport_metrics=require_transport_metrics,
        )
    )
    errors.extend(
        _check_multimodal_consume_only(
            case_id,
            items,
            smoke_case,
            strict=require_transport_metrics or bool(smoke_case and smoke_case.require_transport_metrics),
        )
    )
    return errors


def _check_tp(
    case_id: str,
    items: list[RankSummary],
    *,
    smoke_case: SmokeCase | None = None,
    require_transport_metrics: bool = False,
) -> list[str]:
    errors = _check_common(
        case_id,
        items,
        smoke_case=smoke_case,
        require_transport_metrics=require_transport_metrics,
    )
    ranks = sorted(item.summary.get("rank") for item in items)
    expected_world = smoke_case.expected_rank_count if smoke_case and smoke_case.expected_rank_count else len(items)
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


def _check_pp(
    case_id: str,
    items: list[RankSummary],
    *,
    smoke_case: SmokeCase | None = None,
    require_transport_metrics: bool = False,
) -> list[str]:
    errors = _check_common(
        case_id,
        items,
        smoke_case=smoke_case,
        require_transport_metrics=require_transport_metrics,
    )
    by_stage = {item.summary.get("stage_idx"): item for item in items}
    expected_stages = list(range(len(items)))
    if sorted(by_stage) != expected_stages:
        errors.append(f"{case_id}: expected stages {expected_stages}, got {sorted(by_stage)}")

    for stage_idx, item in by_stage.items():
        _require_true(errors, item, "weight_load.stage_weight_scope_ok")
        _require_equal(errors, item, "start_idx", _get_path(item.summary, "weight_load.stage_start_idx"))
        _require_equal(errors, item, "end_idx", _get_path(item.summary, "weight_load.stage_end_idx"))
        if smoke_case is not None and smoke_case.modality == "multimodal":
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


def _check_hybrid(
    case_id: str,
    items: list[RankSummary],
    *,
    smoke_case: SmokeCase | None = None,
    require_transport_metrics: bool = False,
) -> list[str]:
    errors = _check_common(
        case_id,
        items,
        smoke_case=smoke_case,
        require_transport_metrics=require_transport_metrics,
    )
    stage_indices = sorted({item.summary.get("stage_idx") for item in items})
    if not stage_indices or any(stage_idx is None for stage_idx in stage_indices):
        errors.append(f"{case_id}: missing stage_idx in hybrid rank summaries")
        return errors
    expected_stages = list(range(max(stage_indices) + 1))
    if stage_indices != expected_stages:
        errors.append(f"{case_id}: expected stages {expected_stages}, got {stage_indices}")

    last_stage_idx = max(stage_indices)
    for stage_idx in stage_indices:
        stage_items = sorted(
            [item for item in items if item.summary.get("stage_idx") == stage_idx],
            key=lambda item: item.summary.get("local_rank", -1),
        )
        expected_tp_degree = stage_items[0].summary.get("tp_degree")
        if expected_tp_degree is not None and len(stage_items) != expected_tp_degree:
            errors.append(
                f"{case_id}: stage{stage_idx} expected tp_degree={expected_tp_degree}, got {len(stage_items)} ranks"
            )
        for local_rank, item in enumerate(stage_items):
            _require_equal(errors, item, "local_rank", local_rank)
            _require_true(errors, item, "weight_load.stage_weight_scope_ok")
            tp_degree = item.summary.get("tp_degree")
            if tp_degree and tp_degree > 1:
                _require_true(errors, item, "weight_load.tp_weight_sharded")
                _require_equal(errors, item, "weight_load.tp_shard_rank", local_rank)
                _require_equal(errors, item, "weight_load.tp_shard_world_size", tp_degree)
                _require_true(errors, item, "weight_load.tp_shard_shape_ok")
                if _get_path(item.summary, "weight_load.tp_stage_loaded_weight_tensor_bytes_equal") is not None:
                    _require_true(errors, item, "weight_load.tp_stage_loaded_weight_tensor_bytes_equal")
            elif tp_degree == 1:
                _require_equal(errors, item, "weight_load.tp_weight_sharded", False)
            if smoke_case is not None and smoke_case.modality == "multimodal":
                if stage_idx == 0:
                    _require_equal(errors, item, "weight_load.loaded_top_level_weight_names", ["embed_tokens_weight"])
                elif stage_idx == last_stage_idx:
                    _require_equal(
                        errors,
                        item,
                        "weight_load.loaded_top_level_weight_names",
                        ["final_norm_weight", "lm_head_weight"],
                    )
                    _require_frontend_mode(errors, item, "consume-only")
    return errors


def check_baseline_logs(
    case_id: str,
    paths: list[Path],
    *,
    require_transport_metrics: bool = False,
    use_smoke_matrix: bool = True,
) -> list[RankSummary]:
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
    smoke_case = get_smoke_case(case_id) if use_smoke_matrix else None
    if case_prefix == "tp":
        errors = _check_tp(
            case_id,
            items,
            smoke_case=smoke_case,
            require_transport_metrics=require_transport_metrics,
        )
    elif case_prefix == "pp" or case_prefix.startswith("pp"):
        errors = _check_pp(
            case_id,
            items,
            smoke_case=smoke_case,
            require_transport_metrics=require_transport_metrics,
        )
    elif case_prefix == "hybrid":
        errors = _check_hybrid(
            case_id,
            items,
            smoke_case=smoke_case,
            require_transport_metrics=require_transport_metrics,
        )
    else:
        errors = _check_common(
            case_id,
            items,
            smoke_case=smoke_case,
            require_transport_metrics=require_transport_metrics,
        )
    if errors:
        raise BaselineCheckError("\n".join(errors))
    return items


def _discover_logs_for_case(baseline_dir: Path, case_id: str) -> list[Path]:
    paths = sorted(baseline_dir.glob(f"{case_id}-rank*.log"))
    if paths:
        return paths
    candidates = [
        baseline_dir / f"{case_id}.log",
        baseline_dir / f"{case_id}.stdout",
    ]
    return [path for path in candidates if path.exists()]


def check_smoke_matrix(
    baseline_dir: Path,
    *,
    include_optional: bool = False,
    require_transport_metrics: bool = False,
) -> list[tuple[SmokeCase, list[RankSummary]]]:
    results: list[tuple[SmokeCase, list[RankSummary]]] = []
    missing: list[str] = []
    for smoke_case in iter_smoke_cases(include_optional=include_optional):
        paths = _discover_logs_for_case(baseline_dir, smoke_case.case_id)
        if not paths:
            missing.append(smoke_case.case_id)
            continue
        items = check_baseline_logs(
            smoke_case.case_id,
            paths,
            require_transport_metrics=require_transport_metrics or smoke_case.require_transport_metrics,
            use_smoke_matrix=True,
        )
        results.append((smoke_case, items))
    if missing:
        raise BaselineCheckError(f"missing smoke matrix case logs: {', '.join(missing)}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, help="Rank log files to check")
    parser.add_argument("--case-id", help="Baseline case id, e.g. tp-mm-generate")
    parser.add_argument("--baseline-dir", type=Path, help="Check the frozen Step 22 smoke matrix in this directory.")
    parser.add_argument("--matrix", choices=["step22"], help="Check a named smoke matrix from --baseline-dir.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional smoke cases.")
    parser.add_argument("--require-transport-metrics", action="store_true", help="Fail if transport bytes are missing.")
    parser.add_argument("--no-smoke-matrix", action="store_true", help="Disable built-in expected ids/text rules.")
    args = parser.parse_args(argv)

    if args.matrix:
        if args.baseline_dir is None:
            parser.error("--matrix requires --baseline-dir")
        results = check_smoke_matrix(
            args.baseline_dir,
            include_optional=args.include_optional,
            require_transport_metrics=args.require_transport_metrics,
        )
        print(f"PASS matrix={args.matrix} baseline_dir={args.baseline_dir} cases={len(results)}")
        for smoke_case, items in results:
            first = items[0].summary
            print(
                "PASS "
                f"case={smoke_case.case_id} ranks={len(items)} "
                f"generated_token_ids={first.get('generated_token_ids')!r} "
                f"generated_text={first.get('generated_text')!r}"
            )
        return 0

    if args.case_id is None:
        parser.error("--case-id is required unless --matrix is used")
    if not args.logs:
        parser.error("at least one log path is required unless --matrix is used")

    items = check_baseline_logs(
        args.case_id,
        args.logs,
        require_transport_metrics=args.require_transport_metrics,
        use_smoke_matrix=not args.no_smoke_matrix,
    )
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
