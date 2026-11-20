#!/usr/bin/env python3
"""Generate a strict quality-control report for live Phase 2 collection."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from phase2_common import read_jsonl
from run_segments import MEASUREMENT_METHOD, load_live_manifest


def table(headers: list[str], rows: list[list[object]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


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

    checks: list[str] = []
    try:
        manifest_rows, manifest_digest = load_live_manifest(args.manifest, args.root)
    except (OSError, ValueError) as error:
        manifest_rows = read_jsonl(args.manifest)
        manifest_digest = "invalid"
        checks.append(f"Live manifest validation failed: {error}")
    result_rows = read_jsonl(args.results)

    manifest_ids = [str(row.get("segment_id", "")) for row in manifest_rows]
    manifest = set(manifest_ids)
    duplicate_segments = sum(
        count - 1 for count in Counter(manifest_ids).values() if count > 1
    )
    run_ids = [str(row.get("run_id", "")) for row in result_rows]
    duplicate_runs = sum(count - 1 for count in Counter(run_ids).values() if count > 1)
    empty_run_ids = sum(not value for value in run_ids)
    result_segment_ids = {str(row.get("segment_id", "")) for row in result_rows}
    missing = manifest - result_segment_ids
    unknown = result_segment_ids - manifest
    verdicts = Counter(
        str(row.get("outcome", {}).get("verdict", "missing")) for row in result_rows
    )
    methods = Counter(str(row.get("measurement_method", "missing")) for row in result_rows)
    benchmarks = Counter(str(row.get("benchmark", "missing")) for row in manifest_rows)
    heuristic_manifest: Counter[str] = Counter()
    heuristic_results: Counter[str] = Counter()
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
        row.get("manifest_sha256") != manifest_digest for row in result_rows
    )
    schema_mismatches = sum(row.get("schema_version") != 3 for row in result_rows)
    failures_without_output = sum(
        row.get("outcome", {}).get("verdict")
        not in {"correct", "syntactic_skip", "timeout"}
        and not row.get("stdout_tail")
        and not row.get("stderr_tail")
        for row in result_rows
    )

    if not manifest_rows:
        checks.append("The live manifest contains no sampled segments.")
    if missing:
        checks.append("Collection is incomplete; resume using the same manifest and result file.")
    if duplicate_segments:
        checks.append("The manifest contains duplicate segment IDs.")
    if duplicate_runs or empty_run_ids:
        checks.append("Result run IDs are duplicate or empty.")
    if unknown:
        checks.append("Some results do not belong to this live manifest.")
    if methods and set(methods) != {MEASUREMENT_METHOD}:
        checks.append("Results contain legacy or mixed measurement methods.")
    if schema_mismatches:
        checks.append("Results contain records outside schema version 3.")
    if digest_mismatches:
        checks.append("Some results refer to a different live manifest digest.")
    if invalid_measurements:
        checks.append("Some runs are not valid exact-snapshot measurements.")
    if missing_non_timeout_timing:
        checks.append("Some non-timeout runs lack internal verification timing.")
    if failures_without_output:
        checks.append("Some failed runs have no saved stdout/stderr diagnostic tail.")
    status = "FAIL" if checks else "PASS"

    lines = [
        "# DESTIVA Phase 2 quality-control report", "",
        f"**QC status: {status}**", "",
        "## Collection status", "",
        f"- Live manifest segments: **{len(manifest_rows)}**",
        f"- Result records: **{len(result_rows)}**",
        f"- Manifest segments with no result: **{len(missing)}**",
        f"- Results absent from the manifest: **{len(unknown)}**",
        f"- Duplicate manifest segment IDs: **{duplicate_segments}**",
        f"- Duplicate result run IDs: **{duplicate_runs}**",
        f"- Valid exact-snapshot measurements: **{len(result_rows) - invalid_measurements}**",
        f"- Records with internal verification timing: **{timed}**",
        f"- Non-timeout records missing internal timing: **{missing_non_timeout_timing}**",
        f"- Manifest digest mismatches: **{digest_mismatches}**",
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
