"""Regression checker for multimodal decode hybrid parity."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


RANKS = (0, 1, 2, 3)
RANK_LAUNCH_SCRIPT = "run-hybrid-multimodal-decode-rank.sh"

CHECK_DIFF_KEYS = (
    "boundary_max_diff",
    "direct_max_diff",
    "stage_max_diff",
    "tp_direct_max_diff",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check multimodal decode hybrid regression parity.")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run prepare-hybrid-multimodal-decode.sh before launching the regression check.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Override MANIFEST_PATH for both prepare and run.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Override MASTER_PORT for the 4-rank launch.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-process timeout in seconds for prepare/run commands.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of last-token top-k entries to compare on the final stage leader.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for scalar diffs and top-k logits.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store prepare/rank logs. Uses a temporary directory by default.",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="Keep temporary logs even when the check passes.",
    )
    return parser.parse_args()


def extract_last_json_payload(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)

    # Pretty-printed summaries contain many nested ``{...}`` blocks, so the
    # lexically last JSON object in the log is often just an inner dict such as
    # one top-k item. Prefer the last dict that looks like the full rank
    # summary, then fall back to the final dict candidate if needed.
    for obj in reversed(candidates):
        if "rank" in obj and "pipeline_type" in obj:
            return obj
        if any(key in obj for key in CHECK_DIFF_KEYS):
            return obj

    if candidates:
        return candidates[-1]
    return None


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout_seconds: int,
    log_path: Path,
) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        log_path.write_text(output, encoding="utf-8")
        raise RuntimeError(
            f"Command timed out after {timeout_seconds}s: {' '.join(cmd)}\nlog={log_path}"
        )
    log_path.write_text(output, encoding="utf-8")
    return proc.returncode, output


def validate_scalar(summary: dict, key: str, tolerance: float) -> list[str]:
    value = summary.get(key)
    if value is None:
        return [f"{key} is null"]
    if abs(float(value)) > tolerance:
        return [f"{key}={value} exceeds tolerance={tolerance}"]
    return []


def validate_topk(summary: dict, tolerance: float) -> list[str]:
    last_stage_topk = summary.get("last_stage_topk")
    reference_topk = summary.get("reference_topk")
    if last_stage_topk is None or reference_topk is None:
        return []
    if len(last_stage_topk) != len(reference_topk):
        return [
            "last_stage_topk and reference_topk length mismatch: "
            f"{len(last_stage_topk)} vs {len(reference_topk)}"
        ]

    errors = []
    for idx, (lhs, rhs) in enumerate(zip(last_stage_topk, reference_topk)):
        if lhs.get("token_id") != rhs.get("token_id"):
            errors.append(
                f"topk[{idx}] token mismatch: {lhs.get('token_id')} vs {rhs.get('token_id')}"
            )
        lhs_logit = lhs.get("logit")
        rhs_logit = rhs.get("logit")
        if lhs_logit is None or rhs_logit is None:
            errors.append(f"topk[{idx}] missing logit")
            continue
        if abs(float(lhs_logit) - float(rhs_logit)) > tolerance:
            errors.append(
                f"topk[{idx}] logit mismatch: {lhs_logit} vs {rhs_logit} "
                f"(tolerance={tolerance})"
            )
    return errors


def validate_summary(summary: dict, tolerance: float) -> list[str]:
    errors = []
    for key in CHECK_DIFF_KEYS:
        errors.extend(validate_scalar(summary, key, tolerance))
    errors.extend(validate_topk(summary, tolerance))
    return errors


def build_rank_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["HYBRID_DEBUG"] = "0"
    env["COMPARE_DIRECT"] = "1"
    env["TRACE_LAYERS"] = "0"
    env["DUMP_LAYER"] = ""
    env["DUMP_TOPK"] = str(args.topk)
    env["PYTHONUNBUFFERED"] = "1"
    env["WORLD_SIZE"] = str(len(RANKS))
    if args.manifest_path is not None:
        env["MANIFEST_PATH"] = args.manifest_path
    if args.master_port is not None:
        env["MASTER_PORT"] = str(args.master_port)
    return env


def build_prepare_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.manifest_path is not None:
        env["MANIFEST_PATH"] = args.manifest_path
    return env


def main() -> int:
    args = parse_args()
    runtime_root = Path(__file__).resolve().parent.parent

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.log_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="qwen3vl_multimodal_decode_hybrid_check_")
        log_dir = Path(temp_dir.name)
    else:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    keep_logs = bool(args.keep_logs or args.log_dir is not None)
    success = False

    try:
        print(f"[check] runtime_root={runtime_root}")
        print(f"[check] log_dir={log_dir}")
        if args.manifest_path is not None:
            print(f"[check] manifest_path={args.manifest_path}")
        if args.master_port is not None:
            print(f"[check] master_port={args.master_port}")

        if args.prepare:
            prepare_log = log_dir / "prepare.log"
            print("[check] running prepare-hybrid-multimodal-decode.sh")
            returncode, _ = run_command(
                ["bash", str(runtime_root / "prepare-hybrid-multimodal-decode.sh")],
                env=build_prepare_env(args),
                cwd=runtime_root.parent,
                timeout_seconds=args.timeout_seconds,
                log_path=prepare_log,
            )
            if returncode != 0:
                raise RuntimeError(
                    "prepare-hybrid-multimodal-decode.sh failed "
                    f"(exit_code={returncode}) log={prepare_log}"
                )

        rank_env = build_rank_env(args)
        processes: dict[int, subprocess.Popen] = {}
        log_paths: dict[int, Path] = {}
        script_path = runtime_root / RANK_LAUNCH_SCRIPT

        print("[check] launching 4 hybrid ranks")
        for rank in RANKS:
            log_paths[rank] = log_dir / f"rank{rank}.log"
            rank_specific_env = rank_env.copy()
            rank_specific_env["RANK"] = str(rank)
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=str(runtime_root.parent),
                env=rank_specific_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes[rank] = proc

        outputs: dict[int, str] = {}
        try:
            for rank, proc in processes.items():
                output, _ = proc.communicate(timeout=args.timeout_seconds)
                outputs[rank] = output
                log_paths[rank].write_text(output, encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            for proc in processes.values():
                if proc.poll() is None:
                    proc.kill()
            for rank, proc in processes.items():
                try:
                    output, _ = proc.communicate()
                except Exception:
                    output = ""
                outputs[rank] = output
                log_paths[rank].write_text(output, encoding="utf-8")
            raise RuntimeError(
                f"rank launch timed out after {args.timeout_seconds}s; last rank={rank} log={log_paths[rank]}"
            ) from exc

        parsed_summaries: dict[int, dict] = {}
        errors: list[str] = []
        saw_topk = False
        for rank in RANKS:
            proc = processes[rank]
            exit_code = proc.returncode
            if exit_code != 0:
                errors.append(f"rank{rank} exited with code {exit_code} log={log_paths[rank]}")
                continue

            summary = extract_last_json_payload(outputs[rank])
            if summary is None:
                errors.append(f"rank{rank} output does not contain a JSON summary log={log_paths[rank]}")
                continue

            parsed_summaries[rank] = summary
            summary_errors = validate_summary(summary, args.tolerance)
            if summary.get("last_stage_topk") is not None and summary.get("reference_topk") is not None:
                saw_topk = True
            if summary_errors:
                for message in summary_errors:
                    errors.append(f"rank{rank}: {message} log={log_paths[rank]}")

        if not saw_topk:
            errors.append("no final-stage topk summary was found in any rank output")

        if errors:
            lines = "\n".join(f"- {message}" for message in errors)
            raise RuntimeError(f"multimodal decode hybrid regression FAILED:\n{lines}")

        print("[check] PASS multimodal_decode hybrid parity is exact")
        for rank in sorted(parsed_summaries):
            summary = parsed_summaries[rank]
            print(
                "[check] "
                f"rank={rank} "
                f"boundary={summary['boundary_max_diff']} "
                f"direct={summary['direct_max_diff']} "
                f"stage={summary['stage_max_diff']} "
                f"tp_direct={summary['tp_direct_max_diff']}"
            )
            if summary.get("last_stage_topk") is not None:
                print(f"[check] rank={rank} last_stage_topk={summary['last_stage_topk']}")
        success = True
        if keep_logs:
            print(f"[check] logs kept at {log_dir}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        print(f"[check] logs kept at {log_dir}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None and success and not keep_logs:
            shutil.rmtree(temp_dir.name, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
