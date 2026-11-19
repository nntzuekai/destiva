#!/usr/bin/env python3
"""Generate a compact quality-control report for Phase 2 collection."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from phase2_common import read_jsonl


def table(headers: list[str], rows: list[list[object]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    result_rows = read_jsonl(args.results)
    manifest = {str(row["segment_id"]): row for row in manifest_rows}
    run_ids = [str(row.get("run_id", "")) for row in result_rows]
    duplicate_runs = sum(count - 1 for count in Counter(run_ids).values() if count > 1)
    result_segment_ids = {str(row.get("segment_id", "")) for row in result_rows}
    missing = set(manifest) - result_segment_ids
    unknown = result_segment_ids - set(manifest)
    verdicts = Counter(str(row.get("outcome", {}).get("verdict", "missing")) for row in result_rows)
    benchmarks = Counter(str(row["benchmark"]) for row in manifest_rows)
    heuristic_manifest = Counter()
    heuristic_results = Counter()
    for row in manifest_rows:
        heuristic_manifest.update(row.get("selected_by", []))
    for row in result_rows:
        heuristic_results.update(row.get("selected_by", []))
    completed_with_timing = sum(
        row.get("outcome", {}).get("verification_time_seconds") is not None
        for row in result_rows
    )
    failures_without_output = sum(
        row.get("outcome", {}).get("verdict") not in {"correct", "syntactic_skip"}
        and not row.get("stdout_tail") and not row.get("stderr_tail")
        for row in result_rows
    )

    lines = [
        "# DESTIVA Phase 2 quality-control report", "",
        "## Collection status", "",
        f"- Manifest segments: **{len(manifest_rows)}**",
        f"- Result records: **{len(result_rows)}**",
        f"- Manifest segments with no result: **{len(missing)}**",
        f"- Results absent from the manifest: **{len(unknown)}**",
        f"- Duplicate run IDs: **{duplicate_runs}**",
        f"- Records with internal verification timing: **{completed_with_timing}**",
        f"- Failed records lacking diagnostic output: **{failures_without_output}**", "",
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
    checks = []
    if missing:
        checks.append("Collection is incomplete or was intentionally limited; resume with the same output file.")
    if duplicate_runs:
        checks.append("Duplicate run IDs should be removed before statistical analysis.")
    if unknown:
        checks.append("Some results were produced from a different manifest.")
    if result_rows and completed_with_timing == 0:
        checks.append("No PHASE2_TIME telemetry was found; rebuild alive2-decomp with the indexed-pair patch.")
    if failures_without_output:
        checks.append("Some failed runs have no saved stdout/stderr tail.")
    if not checks:
        checks.append("No structural collection problems detected.")
    lines.extend(f"- {item}" for item in checks)
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

