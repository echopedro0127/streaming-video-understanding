from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streaming_video_understanding.analysis.pairs import (
    analyze_pair_records,
    write_pair_artifacts,
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"input does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"malformed JSON at {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object at {path}, line {line_number}")
            records.append(record)
    return records


_ARTIFACT_SUFFIXES = {
    "csv": "-paired.csv",
    "summary_json": "-summary.json",
    "helped_jsonl": "-helped.jsonl",
    "hurt_jsonl": "-hurt.jsonl",
}


def _validate_output_prefix(value: str) -> str:
    if not value or value in {".", ".."}:
        raise ValueError("--output-prefix must be a safe single filename component")
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("--output-prefix must be a safe single filename component")
    return value


def _artifact_paths(output_dir: Path, output_prefix: str) -> dict[str, Path]:
    resolved_output_dir = output_dir.resolve()
    prefix = resolved_output_dir / output_prefix
    paths = {
        key: Path(f"{prefix}{suffix}")
        for key, suffix in _ARTIFACT_SUFFIXES.items()
    }
    try:
        for path in paths.values():
            path.resolve().relative_to(resolved_output_dir)
    except ValueError as exc:
        raise ValueError("output artifacts must remain under --output-dir") from exc
    return paths


def _existing_artifacts(paths: dict[str, Path]) -> list[Path]:
    return [path for path in paths.values() if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze paired streaming results from two policy JSONL files."
    )
    parser.add_argument("--recent-input", type=Path, required=True)
    parser.add_argument("--recent-policy", required=True)
    parser.add_argument("--option-input", type=Path, required=True)
    parser.add_argument("--option-policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--video-min", type=int, default=None)
    parser.add_argument("--video-max", type=int, default=None)
    parser.add_argument("--representative-limit", type=int, default=20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output artifacts",
    )
    args = parser.parse_args()

    if args.representative_limit < 0:
        parser.error("--representative-limit must be non-negative")
    if (
        args.video_min is not None
        and args.video_max is not None
        and args.video_min > args.video_max
    ):
        parser.error("--video-min cannot exceed --video-max")
    try:
        output_prefix = _validate_output_prefix(args.output_prefix)
        artifact_paths = _artifact_paths(args.output_dir, output_prefix)
    except ValueError as exc:
        parser.error(str(exc))
    existing = _existing_artifacts(artifact_paths)
    if existing and not args.overwrite:
        parser.error(
            "refusing to overwrite existing output artifact(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them"
        )

    recent_records = _load_jsonl(args.recent_input)
    option_records = _load_jsonl(args.option_input)
    result = analyze_pair_records(
        recent_records,
        option_records,
        args.recent_policy,
        args.option_policy,
        min_video_index=args.video_min,
        max_video_index=args.video_max,
    )
    paths = write_pair_artifacts(
        result,
        args.output_dir.resolve() / output_prefix,
        representative_limit=args.representative_limit,
    )
    print(f"paired={result.summary['n_pairs']}")
    print(f"summary={paths['summary_json']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
