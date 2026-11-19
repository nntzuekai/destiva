#!/usr/bin/env python3
"""Join a Phase 2 manifest, run outcomes, and function-level IR features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from phase2_common import read_jsonl


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "mp" / "phase1"))
    from phase1_extract import ir_features  # imported after local path setup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--boundary-features", type=Path,
                        help="optional JSONL from extract_boundary_features.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-ir-features", action="store_true",
                        help="skip original-function IR characteristics")
    args = parser.parse_args()
    args.root = args.root.resolve()

    manifest = {str(row["segment_id"]): row for row in read_jsonl(args.manifest)}
    results = read_jsonl(args.results)
    boundary_features = {
        str(row["segment_id"]): row for row in read_jsonl(args.boundary_features)
    } if args.boundary_features else {}
    feature_cache: dict[str, dict[str, object]] = {}
    rows = []
    errors = []
    for result in results:
        segment_key = str(result.get("segment_id", ""))
        segment = manifest.get(segment_key)
        if segment is None:
            errors.append({"run_id": result.get("run_id"), "error": "segment missing from manifest"})
            continue
        relative_ir = str(segment["relative_ir"])
        if not args.no_ir_features and relative_ir not in feature_cache:
            try:
                feature_cache[relative_ir] = ir_features(
                    args.root / "benchmarks" / "funcs" / relative_ir,
                    str(segment["function"]),
                )
            except Exception as error:
                errors.append({"relative_ir": relative_ir, "error": str(error)})
                feature_cache[relative_ir] = {}
        outcome = result.get("outcome", {})
        row: dict[str, object] = {
            "schema_version": 1,
            "run_id": result.get("run_id", ""),
            "segment_id": segment_key,
            "mode": result.get("mode", ""),
            "repeat": result.get("repeat", ""),
            "benchmark": segment["benchmark"],
            "function": segment["function"],
            "relative_ir": relative_ir,
            "start_index": segment["start_index"],
            "end_index": segment["end_index"],
            "pass_count": segment["pass_count"],
            "selected_by": ";".join(segment["selected_by"]),
            "verdict": outcome.get("verdict", ""),
            "censored": outcome.get("censored", ""),
            "wall_time_seconds": outcome.get("wall_time_seconds", ""),
            "verification_time_seconds": outcome.get("verification_time_seconds", ""),
            "returncode": outcome.get("returncode", ""),
        }
        row.update(segment.get("features", {}))
        row.update(boundary_features.get(segment_key, {}).get("boundary_features", {}))
        row.update(feature_cache.get(relative_ir, {}))
        rows.append(row)

    identity = [
        "schema_version", "run_id", "segment_id", "mode", "repeat", "benchmark",
        "function", "relative_ir", "start_index", "end_index", "pass_count",
        "selected_by", "verdict", "censored", "wall_time_seconds",
        "verification_time_seconds", "returncode",
    ]
    extra = sorted({key for row in rows for key in row} - set(identity))
    fields = identity + extra
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "results": str(args.results.resolve()),
        "boundary_features": str(args.boundary_features.resolve()) if args.boundary_features else None,
        "row_count": len(rows),
        "field_count": len(fields),
        "function_feature_count": len(feature_cache),
        "boundary_feature_segment_count": len(boundary_features),
        "errors": errors,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
