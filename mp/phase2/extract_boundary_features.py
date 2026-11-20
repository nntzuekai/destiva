#!/usr/bin/env python3
"""Legacy sidecar-aligned Phase 2 boundary extractor.

New experiments must use ``collect_live.py``, which extracts characteristics
and snapshots from one live LLVM trace without reading ``*_passes.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from phase2_common import load_passes, read_jsonl


NOISE_RE = re.compile(r";\d+$")
HEADER_RE = re.compile(r"^\s*;\s*\*\*\* IR Dump (Before|After) (.+?) on ")
STAR_RE = re.compile(r"^\s*;\s*\*\*\*")
SNAPSHOT_METHOD = "post_pass_ir_v1"


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
            and sections[index][1] == sections[index + 1][1]
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


def sequence_sha256(names: list[str]) -> str:
    return hashlib.sha256("\0".join(names).encode()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def first_sequence_difference(expected: list[str], observed: list[str]) -> dict[str, object] | None:
    for index, (expected_name, observed_name) in enumerate(zip(expected, observed)):
        if expected_name != observed_name:
            return {
                "index": index,
                "expected": expected_name,
                "observed": observed_name,
            }
    if len(expected) != len(observed):
        index = min(len(expected), len(observed))
        return {
            "index": index,
            "expected": expected[index] if index < len(expected) else "<end>",
            "observed": observed[index] if index < len(observed) else "<end>",
        }
    return None


def stored_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def tool_output(command: list[str], cwd: Path | None = None) -> str:
    run = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return (run.stdout + run.stderr).strip()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "mp" / "phase1"))
    from phase1_extract import ir_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path,
                        help="persistent IR snapshots (default: OUTPUT stem + _snapshots)")
    parser.add_argument("--opt", type=Path, default=root / "llvm-project" / "build" / "bin" / "opt")
    parser.add_argument("--limit-functions", type=int, default=0)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    args.opt = args.opt.resolve()
    snapshots_dir = (
        args.snapshots_dir.resolve() if args.snapshots_dir
        else args.output.parent / f"{args.output.stem}_snapshots"
    )
    metadata_path = args.output.with_suffix(".metadata.json")
    if not args.opt.exists():
        raise SystemExit(f"LLVM opt not found: {args.opt}")

    manifest = read_jsonl(args.manifest)
    if not manifest:
        raise SystemExit(f"manifest is empty: {args.manifest}")
    by_function: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in manifest:
        by_function[str(record["relative_ir"])].append(record)
    all_function_items = sorted(by_function.items())
    function_items = all_function_items
    if args.limit_functions > 0:
        function_items = function_items[:args.limit_functions]

    output_records: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    aligned_functions = 0
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    for number, (relative_ir, segments) in enumerate(function_items, 1):
        ir_path = args.root / "benchmarks" / "funcs" / relative_ir
        pass_path = args.root / "benchmarks" / "funcs" / str(segments[0]["relative_passes"])
        try:
            function, expected_passes = load_passes(pass_path)
        except Exception as error:
            errors.append({"relative_ir": relative_ir, "error": f"cannot load pass data: {error}"})
            continue
        expected_names = [str(entry["name"]) for entry in expected_passes]
        command = [
            str(args.opt), "-passes=default<O3>",
            "--print-before-all", "--print-after-all", "--print-module-scope",
            str(ir_path), "-o", "/dev/null",
        ]
        run = subprocess.run(command, capture_output=True, text=True)
        pairs = pair_sections(parse_sections(run.stderr))
        observed_names = [pair[0] for pair in pairs]
        difference = first_sequence_difference(expected_names, observed_names)
        if run.returncode != 0 or not pairs:
            errors.append({
                "relative_ir": relative_ir,
                "error": "opt failed or produced no paired IR dumps",
                "returncode": run.returncode,
                "stderr_tail": run.stderr[-4000:],
            })
            continue
        if difference is not None:
            errors.append({
                "relative_ir": relative_ir,
                "error": "pass sequence mismatch",
                "expected_pass_count": len(expected_names),
                "observed_pass_count": len(observed_names),
                "first_difference": difference,
                "expected_sequence_sha256": sequence_sha256(expected_names),
                "observed_sequence_sha256": sequence_sha256(observed_names),
            })
            continue

        aligned_functions += 1
        endpoint_indices = sorted({
            int(record[key]) for record in segments for key in ("start_index", "end_index")
        })
        endpoint_features: dict[int, dict[str, object]] = {}
        endpoint_paths: dict[int, Path] = {}
        endpoint_hashes: dict[int, str] = {}
        for index in endpoint_indices:
            if index >= len(pairs):
                errors.append({
                    "relative_ir": relative_ir,
                    "error": f"endpoint index {index} exceeds aligned pass sequence",
                })
                break
            snapshot_path = (
                snapshots_dir / Path(relative_ir).with_suffix("") / f"after-{index:04d}.ll"
            )
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_text = pairs[index][2].rstrip() + "\n"
            snapshot_path.write_text(snapshot_text)
            endpoint_paths[index] = snapshot_path
            endpoint_hashes[index] = text_sha256(snapshot_text)
            endpoint_features[index] = ir_features(snapshot_path, function)
        else:
            sequence_digest = sequence_sha256(expected_names)
            for record in segments:
                start = int(record["start_index"])
                end = int(record["end_index"])
                output_records.append({
                    "schema_version": 2,
                    "segment_id": record["segment_id"],
                    "relative_ir": relative_ir,
                    "function": function,
                    "start_index": start,
                    "end_index": end,
                    "start_pass_dump": pairs[start][0],
                    "end_pass_dump": pairs[end][0],
                    "start_pass_manifest": record["features"]["segment_start_pass"],
                    "end_pass_manifest": record["features"]["segment_end_pass"],
                    "pass_names_match": True,
                    "pass_sequence_sha256": sequence_digest,
                    "snapshot_method": SNAPSHOT_METHOD,
                    "start_snapshot": stored_path(endpoint_paths[start], args.root),
                    "end_snapshot": stored_path(endpoint_paths[end], args.root),
                    "start_snapshot_sha256": endpoint_hashes[start],
                    "end_snapshot_sha256": endpoint_hashes[end],
                    "boundary_features": flatten_boundary_features(
                        endpoint_features[start], endpoint_features[end],
                    ),
                })
            print(
                f"[{number}/{len(function_items)}] {relative_ir}: "
                f"{len(segments)} segments, {len(endpoint_indices)} snapshots",
                flush=True,
            )

    output_records.sort(key=lambda row: str(row["segment_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for record in output_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    completed_functions = len({str(row["relative_ir"]) for row in output_records})
    requested_segment_ids = {
        str(record["segment_id"]) for _, segments in function_items for record in segments
    }
    output_segment_ids = {str(record["segment_id"]) for record in output_records}
    complete = (
        not errors
        and aligned_functions == len(function_items)
        and output_segment_ids == requested_segment_ids
    )
    metadata = {
        "schema_version": 2,
        "complete": complete,
        "manifest_complete": len(function_items) == len(all_function_items),
        "manifest": str(args.manifest),
        "opt": str(args.opt),
        "opt_version": tool_output([str(args.opt), "--version"]),
        "llvm_git_commit": tool_output(
            ["git", "rev-parse", "HEAD"], cwd=args.root / "llvm-project",
        ) if (args.root / "llvm-project" / ".git").exists() else "unknown",
        "snapshot_method": SNAPSHOT_METHOD,
        "snapshots_dir": str(snapshots_dir.resolve()),
        "requested_functions": len(function_items),
        "manifest_functions": len(all_function_items),
        "aligned_functions": aligned_functions,
        "completed_functions": completed_functions,
        "requested_segments": len(requested_segment_ids),
        "segment_count": len(output_records),
        "alignment_error_count": len(errors),
        "errors": errors,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(output_records)} segment feature records to {args.output}")
    print(f"metadata: {metadata_path}")
    if not complete:
        raise SystemExit(
            "boundary extraction failed strict pass-alignment checks; "
            f"see {metadata_path}"
        )


if __name__ == "__main__":
    main()
