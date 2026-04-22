"""Stable user-facing generation entrypoint built on the unified runtime CLI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen3vl_tp_runtime.scripts.runtime import main as runtime_main


def _has_option(argv: list[str], option_name: str) -> bool:
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in argv)


def build_runtime_argv(argv: list[str]) -> list[str]:
    runtime_argv = list(argv)
    if _has_option(runtime_argv, "--mode"):
        raise SystemExit("scripts/generate.py 固定走 generate 模式，请不要显式传 --mode。")
    if not _has_option(runtime_argv, "--modality"):
        runtime_argv = ["--modality", "multimodal", *runtime_argv]
    if not _has_option(runtime_argv, "--backend"):
        runtime_argv = ["--backend", "hf", *runtime_argv]
    return ["--mode", "generate", *runtime_argv]


def main(argv: list[str] | None = None) -> None:
    runtime_main(build_runtime_argv(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
