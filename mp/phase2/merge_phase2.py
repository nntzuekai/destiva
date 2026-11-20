#!/usr/bin/env python3
"""Join a Phase 2 manifest, run outcomes, and function-level IR features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from phase2_common import read_jsonl


EXPECTED_METHOD = "exact_snapshot_pair_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "mp" / "phase1"))
    from phase1_extract import ir_features  # imported after local path setup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--boundary-features", type=Path, required=True,
                        help="complete schema-v2 JSONL from extract_boundary_features.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-ir-features", action="store_true",
                        help="skip original-function IR characteristics")
    args = parser.parse_args()
    args.root = args.root.resolve()

    manifest = {str(row["segment_id"]): row for row in read_jsonl(args.manifest)}
    results = read_jsonl(args.results)
    boundary_features = {
        str(row["segment_id"]): row for row in read_jsonl(args.boundary_features)
    }
    boundary_metadata_path = args.boundary_features.with_suffix(".metadata.json")
    if not boundary_metadata_path.exists():
        raise SystemExit(f"boundary metadata not found: {boundary_metadata_path}")
    boundary_metadata = json.loads(boundary_metadata_path.read_text())
    if (
        boundary_metadata.get("schema_version") != 2
        or not boundary_metadata.get("complete")
        or boundary_metadata.get("errors")
    ):
        raise SystemExit("boundary extraction is incomplete or contains alignment errors")
    if set(boundary_features) != set(manifest):
        raise SystemExit("boundary-feature coverage does not exactly match the manifest")
    if any(
        row.get("schema_version") != 2
        or row.get("snapshot_method") != "post_pass_ir_v1"
        or not row.get("pass_names_match")
        for row in boundary_features.values()
    ):
        raise SystemExit("boundary records failed strict snapshot/alignment validation")
    boundary_digest = file_sha256(args.boundary_features)
    run_ids = [str(row.get("run_id", "")) for row in results]
    if any(count > 1 for count in Counter(run_ids).values()):
        raise SystemExit("duplicate run IDs in result file")
    result_segment_ids = {str(row.get("segment_id", "")) for row in results}
    if result_segment_ids != set(manifest):
        raise SystemExit("result coverage does not exactly match the manifest")
    for result in results:
        outcome = result.get("outcome", {})
        if result.get("schema_version") != 2 or result.get("measurement_method") != EXPECTED_METHOD:
            raise SystemExit("legacy or mixed measurement methods cannot be merged")
        if result.get("boundary_features_sha256") != boundary_digest:
            raise SystemExit("result file refers to a different boundary-feature dataset")
        if not outcome.get("measurement_valid"):
            raise SystemExit(f"invalid measurement: {result.get('run_id')}")
        if outcome.get("verdict") != "timeout" and outcome.get("verification_time_seconds") is None:
            raise SystemExit(f"non-timeout result lacks verification timing: {result.get('run_id')}")
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
            "schema_version": 2,
            "run_id": result.get("run_id", ""),
            "segment_id": segment_key,
            "measurement_method": result.get("measurement_method", ""),
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
            "measurement_valid": outcome.get("measurement_valid", ""),
            "wall_time_seconds": outcome.get("wall_time_seconds", ""),
            "verification_time_seconds": outcome.get("verification_time_seconds", ""),
            "returncode": outcome.get("returncode", ""),
        }
        row.update(segment.get("features", {}))
        row.update(boundary_features[segment_key].get("boundary_features", {}))
        row.update(feature_cache.get(relative_ir, {}))
        rows.append(row)

    identity = [
        "schema_version", "run_id", "segment_id", "measurement_method", "mode", "repeat", "benchmark",
        "function", "relative_ir", "start_index", "end_index", "pass_count",
        "selected_by", "verdict", "censored", "measurement_valid", "wall_time_seconds",
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
        "schema_version": 2,
        "complete": not errors,
        "manifest": str(args.manifest.resolve()),
        "results": str(args.results.resolve()),
        "boundary_features": str(args.boundary_features.resolve()),
        "boundary_features_sha256": boundary_digest,
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
    if errors:
        raise SystemExit("merged dataset contains feature-extraction errors; see metadata")


if __name__ == "__main__":
    main()
