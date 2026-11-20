#!/usr/bin/env python3
"""Generate a strict quality-control report for Phase 2 collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from phase2_common import read_jsonl


EXPECTED_METHOD = "exact_snapshot_pair_v1"


def table(headers: list[str], rows: list[list[object]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--boundary-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    result_rows = read_jsonl(args.results)
    boundary_rows = read_jsonl(args.boundary_features)
    metadata_path = args.boundary_features.with_suffix(".metadata.json")
    boundary_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    boundary_digest = file_sha256(args.boundary_features)

    manifest = {str(row["segment_id"]): row for row in manifest_rows}
    boundary = {str(row.get("segment_id", "")): row for row in boundary_rows}
    run_ids = [str(row.get("run_id", "")) for row in result_rows]
    duplicate_runs = sum(count - 1 for count in Counter(run_ids).values() if count > 1)
    result_segment_ids = {str(row.get("segment_id", "")) for row in result_rows}
    missing = set(manifest) - result_segment_ids
    unknown = result_segment_ids - set(manifest)
    missing_boundaries = set(manifest) - set(boundary)
    extra_boundaries = set(boundary) - set(manifest)
    verdicts = Counter(str(row.get("outcome", {}).get("verdict", "missing")) for row in result_rows)
    methods = Counter(str(row.get("measurement_method", "missing")) for row in result_rows)
    benchmarks = Counter(str(row["benchmark"]) for row in manifest_rows)
    heuristic_manifest = Counter()
    heuristic_results = Counter()
    for row in manifest_rows:
        heuristic_manifest.update(row.get("selected_by", []))
    for row in result_rows:
        heuristic_results.update(row.get("selected_by", []))

    timed = sum(
        row.get("outcome", {}).get("verification_time_seconds") is not None
        for row in result_rows
    )
    invalid_measurements = sum(
        not bool(row.get("outcome", {}).get("measurement_valid"))
        for row in result_rows
    )
    missing_non_timeout_timing = sum(
        row.get("outcome", {}).get("verdict") != "timeout"
        and row.get("outcome", {}).get("verification_time_seconds") is None
        for row in result_rows
    )
    digest_mismatches = sum(
        row.get("boundary_features_sha256") != boundary_digest
        for row in result_rows
    )
    boundary_alignment_failures = sum(
        row.get("schema_version") != 2
        or row.get("snapshot_method") != "post_pass_ir_v1"
        or not row.get("pass_names_match")
        for row in boundary_rows
    )
    failures_without_output = sum(
        row.get("outcome", {}).get("verdict") not in {"correct", "syntactic_skip", "timeout"}
        and not row.get("stdout_tail") and not row.get("stderr_tail")
        for row in result_rows
    )

    checks: list[str] = []
    if missing:
        checks.append("Collection is incomplete; resume with the same exact-snapshot output file.")
    if duplicate_runs:
        checks.append("Duplicate run IDs must be removed before analysis.")
    if unknown:
        checks.append("Some results were produced from a different manifest.")
    if missing_boundaries or extra_boundaries:
        checks.append("Boundary-feature coverage does not exactly match the manifest.")
    if not boundary_metadata.get("complete") or boundary_metadata.get("errors"):
        checks.append("Boundary extraction metadata is incomplete or reports errors.")
    if boundary_alignment_failures:
        checks.append("Some boundary records failed schema, snapshot-method, or pass-alignment checks.")
    if methods and set(methods) != {EXPECTED_METHOD}:
        checks.append("Results contain legacy or mixed measurement methods.")
    if digest_mismatches:
        checks.append("Some results refer to a different boundary-feature dataset.")
    if invalid_measurements:
        checks.append("Some runs are not valid exact-snapshot measurements.")
    if missing_non_timeout_timing:
        checks.append("Some non-timeout runs lack exact internal verification timing.")
    if failures_without_output:
        checks.append("Some failed runs have no saved stdout/stderr diagnostic tail.")
    status = "FAIL" if checks else "PASS"

    lines = [
        "# DESTIVA Phase 2 quality-control report", "",
        f"**QC status: {status}**", "",
        "## Collection status", "",
        f"- Manifest segments: **{len(manifest_rows)}**",
        f"- Boundary records: **{len(boundary_rows)}**",
        f"- Result records: **{len(result_rows)}**",
        f"- Manifest segments with no result: **{len(missing)}**",
        f"- Results absent from the manifest: **{len(unknown)}**",
        f"- Duplicate run IDs: **{duplicate_runs}**",
        f"- Valid exact-snapshot measurements: **{len(result_rows) - invalid_measurements}**",
        f"- Records with internal verification timing: **{timed}**",
        f"- Non-timeout records missing internal timing: **{missing_non_timeout_timing}**",
        f"- Boundary digest mismatches: **{digest_mismatches}**",
        f"- Failed records lacking diagnostic output: **{failures_without_output}**", "",
        "## Measurement methods", "",
        *table(["Method", "Runs"], [[key, value] for key, value in sorted(methods.items())]), "",
        "## Verdicts", "",
        *table(["Verdict", "Runs"], [[key, value] for key, value in sorted(verdicts.items())]), "",
        "## Manifest coverage by benchmark", "",
        *table(["Benchmark", "Segments"], [[key, value] for key, value in sorted(benchmarks.items())]), "",
        "## Heuristic membership", "",
        *table(
            ["Heuristic", "Manifest segments", "Collected records"],
            [[key, heuristic_manifest[key], heuristic_results[key]]
             for key in sorted(set(heuristic_manifest) | set(heuristic_results))],
        ), "",
        "## Checks requiring attention", "",
    ]
    lines.extend(f"- {item}" for item in checks or ["No structural collection problems detected."])
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output} (status={status})")
    if checks:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
