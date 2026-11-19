#!/usr/bin/env python3
"""Create a deterministic manifest of sampled DESTIVA temporal segments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from phase2_common import (
    DEFAULT_HEURISTICS,
    HEURISTICS,
    SCHEMA_VERSION,
    collect_candidates,
    load_passes,
    segment_features,
    segment_id,
    select_candidates,
    stable_order_key,
)


def load_function_paths(functions_file: Path) -> list[str]:
    paths = []
    for line in functions_file.read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            paths.append(value if value.endswith(".ll") else value + ".ll")
    return paths


def sample_functions(paths: list[str], count: int, seed: int) -> list[str]:
    """Deterministically sample functions round-robin across benchmarks."""
    if count <= 0 or count >= len(paths):
        return sorted(paths)
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        groups[Path(path).parts[0]].append(path)
    for benchmark, values in groups.items():
        values.sort(key=lambda value: stable_order_key(seed, benchmark, value))
    selected = []
    offsets = Counter()
    benchmarks = sorted(groups)
    while len(selected) < count:
        progressed = False
        for benchmark in benchmarks:
            offset = offsets[benchmark]
            if offset < len(groups[benchmark]):
                selected.append(groups[benchmark][offset])
                offsets[benchmark] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    funcs_dir = args.root / "benchmarks" / "funcs"
    paths = sample_functions(load_function_paths(args.functions), args.sample_functions, args.seed)
    records: list[dict[str, object]] = []
    errors = []
    benchmark_counts: Counter[str] = Counter()
    heuristic_counts: Counter[str] = Counter()

    for relative_ir in paths:
        ir_path = funcs_dir / relative_ir
        pass_path = ir_path.with_name(ir_path.stem + "_passes.json")
        if not ir_path.exists() or not pass_path.exists():
            errors.append({"relative_ir": relative_ir, "error": "missing IR or pass file"})
            continue
        try:
            function, passes = load_passes(pass_path)
            candidates = collect_candidates(passes, args.heuristics)
            chosen = select_candidates(
                candidates, args.max_segments_per_function, args.seed, relative_ir,
            )
            for start, end in chosen:
                selected_by = sorted(candidates[(start, end)])
                features = segment_features(passes, start, end)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "segment_id": segment_id(relative_ir, start, end),
                    "benchmark": Path(relative_ir).parts[0],
                    "function": function,
                    "relative_ir": relative_ir,
                    "relative_passes": str(Path(relative_ir).with_name(Path(relative_ir).stem + "_passes.json")),
                    "pass_count": len(passes),
                    "start_index": start,
                    "end_index": end,
                    "index_pair": f"{start}:{end}",
                    "selected_by": selected_by,
                    "features": features,
                }
                records.append(record)
                benchmark_counts[record["benchmark"]] += 1
                heuristic_counts.update(selected_by)
        except Exception as error:  # preserve failures in metadata for server audits
            errors.append({"relative_ir": relative_ir, "error": str(error)})

    records.sort(key=lambda row: (str(row["benchmark"]), str(row["function"]), int(row["start_index"]), int(row["end_index"])))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "root": str(args.root.resolve()),
        "functions_file": str(args.functions.resolve()),
        "requested_function_count": len(paths),
        "completed_function_count": len({row["relative_ir"] for row in records}),
        "segment_count": len(records),
        "heuristics": list(args.heuristics),
        "max_segments_per_function": args.max_segments_per_function,
        "seed": args.seed,
        "benchmark_segment_counts": dict(sorted(benchmark_counts.items())),
        "heuristic_membership_counts": dict(sorted(heuristic_counts.items())),
        "errors": errors,
    }
    return records, metadata


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--functions", type=Path, default=root / "benchmarks" / "funcs" / "acyclic.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-functions", type=int, default=0,
                        help="stratified function sample size; 0 means all listed functions")
    parser.add_argument("--max-segments-per-function", type=int, default=16,
                        help="heuristic-balanced cap; 0 retains the full heuristic union")
    parser.add_argument("--heuristics", nargs="+", choices=HEURISTICS,
                        default=list(DEFAULT_HEURISTICS))
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.functions = args.functions.resolve()

    records, metadata = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(records)} segments to {args.output}")
    print(f"metadata: {metadata_path}")
    if metadata["errors"]:
        print(f"warning: {len(metadata['errors'])} functions could not be processed")


if __name__ == "__main__":
    main()

