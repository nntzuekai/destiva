#!/usr/bin/env python3
"""Join live Phase 2 segment characteristics and validation outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from phase2_common import read_jsonl
from run_segments import MEASUREMENT_METHOD, load_live_manifest


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.manifest = args.manifest.resolve()
    args.results = args.results.resolve()

    try:
        manifest_rows, manifest_digest = load_live_manifest(args.manifest, args.root)
    except (OSError, ValueError) as error:
        raise SystemExit(f"live manifest validation failed: {error}") from error
    manifest = {str(row["segment_id"]): row for row in manifest_rows}
    results = read_jsonl(args.results)
    run_ids = [str(row.get("run_id", "")) for row in results]
    if any(not value for value in run_ids):
        raise SystemExit("empty run ID in result file")
    if any(count > 1 for count in Counter(run_ids).values()):
        raise SystemExit("duplicate run IDs in result file")
    result_segment_ids = {str(row.get("segment_id", "")) for row in results}
    if result_segment_ids != set(manifest):
        raise SystemExit("result coverage does not exactly match the live manifest")
    for result in results:
        outcome = result.get("outcome", {})
        if result.get("schema_version") != 3 or result.get("measurement_method") != MEASUREMENT_METHOD:
            raise SystemExit("legacy or mixed measurement methods cannot be merged")
        if result.get("manifest_sha256") != manifest_digest:
            raise SystemExit("result file refers to a different live manifest")
        if not outcome.get("measurement_valid"):
            raise SystemExit(f"invalid measurement: {result.get('run_id')}")
        if outcome.get("verdict") != "timeout" and outcome.get("verification_time_seconds") is None:
            raise SystemExit(f"non-timeout result lacks verification timing: {result.get('run_id')}")

    rows: list[dict[str, object]] = []
    for result in results:
        segment_key = str(result["segment_id"])
        segment = manifest[segment_key]
        outcome = result.get("outcome", {})
        row: dict[str, object] = {
            "schema_version": 3,
            "run_id": result["run_id"],
            "segment_id": segment_key,
            "measurement_method": result["measurement_method"],
            "mode": result.get("mode", ""),
            "repeat": result.get("repeat", ""),
            "benchmark": segment["benchmark"],
            "function": segment["function"],
            "relative_ir": segment["relative_ir"],
            "start_index": segment["start_index"],
            "end_index": segment["end_index"],
            "start_pass": segment["start_pass"],
            "start_occurrence": segment["start_occurrence"],
            "end_pass": segment["end_pass"],
            "end_occurrence": segment["end_occurrence"],
            "pass_count": segment["pass_count"],
            "selected_by": ";".join(segment["selected_by"]),
            "selection_policy": segment.get("selection_policy", "heuristic"),
            "sampling_stratum": segment.get("sampling_stratum", ""),
            "sampling_diff_tertile": segment.get("sampling_diff_tertile", ""),
            "sampling_stratum_candidate_count": segment.get(
                "sampling_stratum_candidate_count", "",
            ),
            "verdict": outcome.get("verdict", ""),
            "censored": outcome.get("censored", ""),
            "measurement_valid": outcome.get("measurement_valid", ""),
            "wall_time_seconds": outcome.get("wall_time_seconds", ""),
            "verification_time_seconds": outcome.get("verification_time_seconds", ""),
            "returncode": outcome.get("returncode", ""),
        }
        row.update({f"input_{key}": value for key, value in segment["input_features"].items()})
        row.update(segment["features"])
        row.update(segment["boundary_features"])
        rows.append(row)

    identity = [
        "schema_version", "run_id", "segment_id", "measurement_method", "mode", "repeat",
        "benchmark", "function", "relative_ir", "start_index", "end_index", "start_pass",
        "start_occurrence", "end_pass", "end_occurrence", "pass_count", "selected_by",
        "selection_policy", "sampling_stratum", "sampling_diff_tertile",
        "sampling_stratum_candidate_count",
        "verdict", "censored", "measurement_valid", "wall_time_seconds",
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
        "schema_version": 3,
        "complete": True,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_digest,
        "results": str(args.results),
        "row_count": len(rows),
        "field_count": len(fields),
        "segment_count": len(manifest),
        "run_count": len(results),
        "errors": [],
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
