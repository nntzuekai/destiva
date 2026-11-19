#!/usr/bin/env python3
"""Extract start/end LLVM IR features for segments in a Phase 2 manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from phase2_common import read_jsonl


NOISE_RE = re.compile(r";\d+$")
HEADER_RE = re.compile(r"^\s*;\s*\*\*\* IR Dump (Before|After) (.+?) on ")
STAR_RE = re.compile(r"^\s*;\s*\*\*\*")


def parse_sections(dump_text: str) -> list[tuple[str, str, str]]:
    sections = []
    direction = None
    pass_name = None
    lines: list[str] = []
    for line in dump_text.splitlines():
        match = HEADER_RE.match(line)
        if match:
            if direction is not None:
                sections.append((direction, pass_name, "\n".join(lines)))
            direction, pass_name = match.groups()
            lines = []
        elif STAR_RE.match(line):
            if direction is not None:
                sections.append((direction, pass_name, "\n".join(lines)))
            direction = pass_name = None
            lines = []
        elif direction is not None and not NOISE_RE.search(line):
            lines.append(line)
    if direction is not None:
        sections.append((direction, pass_name, "\n".join(lines)))
    return sections


def pair_sections(sections: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    pairs = []
    index = 0
    while index < len(sections):
        if (
            sections[index][0] == "Before"
            and index + 1 < len(sections)
            and sections[index + 1][0] == "After"
        ):
            _, pass_name, before_ir = sections[index]
            _, _, after_ir = sections[index + 1]
            pairs.append((pass_name, before_ir, after_ir))
            index += 2
        else:
            index += 1
    return pairs


def flatten_boundary_features(
    start: dict[str, object], end: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in sorted(set(start) | set(end)):
        start_value = start.get(key, "")
        end_value = end.get(key, "")
        result[f"start_{key}"] = start_value
        result[f"end_{key}"] = end_value
        if isinstance(start_value, (int, float)) and isinstance(end_value, (int, float)):
            result[f"delta_{key}"] = end_value - start_value
            result[f"abs_delta_{key}"] = abs(end_value - start_value)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "mp" / "phase1"))
    from phase1_extract import ir_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opt", type=Path, default=root / "llvm-project" / "build" / "bin" / "opt")
    parser.add_argument("--limit-functions", type=int, default=0)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.opt = args.opt.resolve()
    if not args.opt.exists():
        raise SystemExit(f"LLVM opt not found: {args.opt}")

    manifest = read_jsonl(args.manifest)
    by_function: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in manifest:
        by_function[str(record["relative_ir"])].append(record)
    function_items = sorted(by_function.items())
    if args.limit_functions > 0:
        function_items = function_items[:args.limit_functions]

    output_records = []
    errors = []
    for number, (relative_ir, segments) in enumerate(function_items, 1):
        ir_path = args.root / "benchmarks" / "funcs" / relative_ir
        function = str(segments[0]["function"])
        command = [
            str(args.opt), "-passes=default<O3>",
            "--print-before-all", "--print-after-all", "--print-module-scope",
            str(ir_path), "-o", "/dev/null",
        ]
        run = subprocess.run(command, capture_output=True, text=True)
        pairs = pair_sections(parse_sections(run.stderr))
        if run.returncode != 0 or not pairs:
            errors.append({
                "relative_ir": relative_ir,
                "returncode": run.returncode,
                "stderr_tail": run.stderr[-4000:],
            })
            continue
        expected_pass_count = int(segments[0]["pass_count"])
        if len(pairs) != expected_pass_count:
            errors.append({
                "relative_ir": relative_ir,
                "error": f"pass count mismatch: manifest={expected_pass_count}, opt={len(pairs)}",
            })
            continue
        endpoint_indices = sorted({
            int(record[key]) for record in segments for key in ("start_index", "end_index")
        })
        endpoint_features: dict[int, dict[str, object]] = {}
        with tempfile.TemporaryDirectory(prefix="destiva-phase2-") as directory:
            temp = Path(directory)
            for index in endpoint_indices:
                endpoint_path = temp / f"endpoint-{index}.ll"
                endpoint_path.write_text(pairs[index][2])
                endpoint_features[index] = ir_features(endpoint_path, function)
        for record in segments:
            start = int(record["start_index"])
            end = int(record["end_index"])
            output_records.append({
                "schema_version": 1,
                "segment_id": record["segment_id"],
                "relative_ir": relative_ir,
                "start_index": start,
                "end_index": end,
                "start_pass_dump": pairs[start][0],
                "end_pass_dump": pairs[end][0],
                "start_pass_manifest": record["features"]["segment_start_pass"],
                "end_pass_manifest": record["features"]["segment_end_pass"],
                "pass_names_match": (
                    pairs[start][0] == record["features"]["segment_start_pass"]
                    and pairs[end][0] == record["features"]["segment_end_pass"]
                ),
                "boundary_features": flatten_boundary_features(
                    endpoint_features[start], endpoint_features[end],
                ),
            })
        print(f"[{number}/{len(function_items)}] {relative_ir}: {len(segments)} segments", flush=True)

    output_records.sort(key=lambda row: str(row["segment_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for record in output_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "opt": str(args.opt),
        "requested_functions": len(function_items),
        "completed_functions": len({row["relative_ir"] for row in output_records}),
        "segment_count": len(output_records),
        "pass_name_mismatches": sum(not row["pass_names_match"] for row in output_records),
        "errors": errors,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(output_records)} segment feature records to {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()

