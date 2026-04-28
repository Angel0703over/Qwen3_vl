#!/usr/bin/env python3
"""Collect timing and memory baseline records from runtime logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen3vl_tp_runtime.scripts.check_baseline_logs import (
    BaselineCheckError,
    extract_last_json_summary_from_text,
)


STARTUP_DONE_RE = re.compile(
    r"^\[startup\]\[(?P<component>[^\]]+)\].* done (?P<message>.*?) in (?P<seconds>[0-9]+(?:\.[0-9]+)?)s$"
)
TIME_REAL_RE = re.compile(r"^real (?P<seconds>[0-9]+(?:\.[0-9]+)?)$")
RANK_LOG_RE = re.compile(r"^(?P<case_id>.+)-rank(?P<rank>[0-9]+)\.log$")


def _round_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _classify_startup_event(component: str, message: str) -> str:
    lowered = message.lower()
    if "post-load barrier" in lowered:
        return "post_load_barrier_seconds"
    if "startup_contract" in lowered:
        if component in {"object-send", "object-recv", "tensor-send", "tensor-recv"}:
            return "startup_contract_transport_seconds"
        return "startup_contract_prepare_seconds"
    if "prepare " in lowered and " session" in lowered:
        return "prepare_session_seconds"
    if lowered.startswith("materialize ") or "materialize local direct shard" in lowered:
        return "materialize_stage_seconds"
    if "stage_scaffold" in lowered or "text_scaffold" in lowered or "stage_state_" in lowered:
        return "scaffold_transport_seconds"
    return "other_seconds"


def _parse_legacy_startup_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = STARTUP_DONE_RE.match(line)
        if not match:
            continue
        component = match.group("component")
        message = match.group("message")
        elapsed = float(match.group("seconds"))
        events.append(
            {
                "component": component,
                "message": message,
                "status": "done",
                "elapsed_seconds": elapsed,
                "kind": _classify_startup_event(component, message),
            }
        )
    return events


def _summarize_event_kinds(events: list[dict[str, Any]]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for event in events:
        kind = str(event.get("kind") or _classify_startup_event(
            str(event.get("component", "")),
            str(event.get("message", "")),
        ))
        totals[kind] += float(event.get("elapsed_seconds") or 0.0)

    for key in (
        "prepare_session_seconds",
        "startup_contract_prepare_seconds",
        "startup_contract_transport_seconds",
        "materialize_stage_seconds",
        "post_load_barrier_seconds",
        "scaffold_transport_seconds",
        "other_seconds",
    ):
        totals.setdefault(key, 0.0)
    totals["startup_contract_seconds"] = (
        totals["startup_contract_prepare_seconds"]
        + totals["startup_contract_transport_seconds"]
    )
    return {key: round(value, 6) for key, value in sorted(totals.items())}


def _parse_real_seconds(text: str) -> float | None:
    for line in reversed(text.splitlines()):
        match = TIME_REAL_RE.match(line.strip())
        if match:
            return float(match.group("seconds"))
    return None


def _read_with_auxiliary_logs(path: Path) -> str:
    chunks = [path.read_text(encoding="utf-8", errors="replace")]
    aux_paths: list[Path] = []
    if path.name.endswith(".stdout"):
        aux_paths.append(path.with_name(path.name.removesuffix(".stdout") + ".stderr"))
    elif path.name.endswith(".log"):
        aux_paths.append(path.with_name(path.name.removesuffix(".log") + ".wrapper.stderr"))
    for aux_path in aux_paths:
        if aux_path.exists():
            chunks.append(aux_path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _discover_case_logs(baseline_dir: Path) -> dict[str, list[Path]]:
    cases: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(baseline_dir.glob("*-rank*.log")):
        match = RANK_LOG_RE.match(path.name)
        if match:
            cases[match.group("case_id")].append(path)
    for path in sorted(baseline_dir.glob("*.stdout")):
        if path.name.endswith(".wrapper.stdout"):
            continue
        cases[path.name.removesuffix(".stdout")].append(path)
    return {case_id: sorted(paths) for case_id, paths in sorted(cases.items())}


def collect_log_record(case_id: str, path: Path) -> dict[str, Any]:
    primary_text = path.read_text(encoding="utf-8", errors="replace")
    combined_text = _read_with_auxiliary_logs(path)
    summary = extract_last_json_summary_from_text(path, primary_text)
    runtime_metrics = summary.get("runtime_metrics") or {}
    startup_metrics = runtime_metrics.get("startup") or {}
    memory_metrics = runtime_metrics.get("memory") or {}
    events = startup_metrics.get("events")
    if not isinstance(events, list):
        events = _parse_legacy_startup_events(combined_text)

    totals_by_kind = startup_metrics.get("totals_by_kind")
    if not isinstance(totals_by_kind, dict):
        totals_by_kind = _summarize_event_kinds(events)

    rank_match = RANK_LOG_RE.match(path.name)
    rank = summary.get("rank")
    if rank is None and rank_match:
        rank = int(rank_match.group("rank"))

    timing = runtime_metrics.get("timing") or {}
    runtime_total = timing.get("runtime_total_seconds")
    if runtime_total is None:
        runtime_total = _parse_real_seconds(combined_text)

    return {
        "case_id": case_id,
        "path": str(path),
        "rank": rank,
        "backend": summary.get("backend"),
        "pipeline_type": summary.get("pipeline_type"),
        "mode": summary.get("mode") or "generate",
        "generated_token_ids": summary.get("generated_token_ids"),
        "generated_text": summary.get("generated_text"),
        "timing": {
            "runtime_total_seconds": _round_seconds(runtime_total),
            "prepare_session_seconds": _round_seconds(totals_by_kind.get("prepare_session_seconds")),
            "startup_contract_seconds": _round_seconds(totals_by_kind.get("startup_contract_seconds")),
            "startup_contract_transport_seconds": _round_seconds(
                totals_by_kind.get("startup_contract_transport_seconds")
            ),
            "materialize_stage_seconds": _round_seconds(totals_by_kind.get("materialize_stage_seconds")),
            "post_load_barrier_seconds": _round_seconds(totals_by_kind.get("post_load_barrier_seconds")),
            "scaffold_transport_seconds": _round_seconds(totals_by_kind.get("scaffold_transport_seconds")),
        },
        "memory": {
            "cpu_max_rss_bytes": memory_metrics.get("cpu_max_rss_bytes"),
            "cuda_peak_allocated_bytes": memory_metrics.get("peak_allocated_bytes"),
            "cuda_peak_reserved_bytes": memory_metrics.get("peak_reserved_bytes"),
            "cuda_available": memory_metrics.get("cuda_available"),
        },
        "weight_load": {
            "loaded_weight_tensor_bytes": _get_path(summary, "weight_load.loaded_weight_tensor_bytes"),
            "tp_weight_sharded": _get_path(summary, "weight_load.tp_weight_sharded"),
            "stage_weight_scope_ok": _get_path(summary, "weight_load.stage_weight_scope_ok"),
        },
    }


def collect_records(baseline_dir: Path, case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    cases = _discover_case_logs(baseline_dir)
    if case_ids:
        cases = {case_id: cases.get(case_id, []) for case_id in case_ids}

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id, paths in cases.items():
        if not paths:
            missing.append(case_id)
            continue
        for path in paths:
            records.append(collect_log_record(case_id, path))
    if missing:
        raise BaselineCheckError(f"missing case logs: {', '.join(missing)}")
    return records


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    value = int(value)
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{value} B"
        amount /= 1024
    return f"{value} B"


def records_to_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "| case | rank | total s | prepare s | startup contract s | transport s | materialize s | barrier s | cuda peak alloc | cuda peak reserved | loaded weights |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        timing = record["timing"]
        memory = record["memory"]
        weight_load = record["weight_load"]
        lines.append(
            "| "
            + " | ".join(
                [
                    record["case_id"],
                    "-" if record["rank"] is None else str(record["rank"]),
                    _format_seconds(timing["runtime_total_seconds"]),
                    _format_seconds(timing["prepare_session_seconds"]),
                    _format_seconds(timing["startup_contract_seconds"]),
                    _format_seconds(timing["startup_contract_transport_seconds"]),
                    _format_seconds(timing["materialize_stage_seconds"]),
                    _format_seconds(timing["post_load_barrier_seconds"]),
                    _format_bytes(memory["cuda_peak_allocated_bytes"]),
                    _format_bytes(memory["cuda_peak_reserved_bytes"]),
                    _format_bytes(weight_load["loaded_weight_tensor_bytes"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("baseline_runs/20260428"),
        help="Directory containing baseline stdout/rank logs.",
    )
    parser.add_argument("--case-id", action="append", help="Collect one case id; can be repeated.")
    parser.add_argument("--output-json", type=Path, help="Write machine-readable records.")
    parser.add_argument("--output-md", type=Path, help="Write markdown table.")
    parser.add_argument("--strict-memory", action="store_true", help="Fail if CUDA peak memory is missing.")
    args = parser.parse_args(argv)

    records = collect_records(args.baseline_dir, args.case_id)
    if args.strict_memory:
        missing_memory = [
            f"{record['case_id']} rank={record['rank']}"
            for record in records
            if record["memory"].get("cuda_peak_allocated_bytes") is None
        ]
        if missing_memory:
            raise BaselineCheckError("missing CUDA peak memory: " + ", ".join(missing_memory))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"baseline_dir": str(args.baseline_dir), "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    markdown = records_to_markdown(records)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
