#!/usr/bin/env python3
"""Build the offline DESTIVA Phase 1 function-level feature dataset.

This script does not invoke LLVM, Alive2, or Z3.  It joins the existing
experimental results with the prepared per-function LLVM IR and pass-diff
files, then extracts static, inexpensive features from those artifacts.

The output is a flat CSV: one row per supported function.  Method outcomes
are named ``<method>_result``, ``<method>_time``, and (where available)
``<method>_num_pairs``.  All other numeric columns are candidate explanatory
features or identifiers documented in README.md in this directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1

# LLVM IR opcodes from the LLVM language reference.  Keeping this explicit
# prevents switch case lines, metadata, and multiline call arguments from
# being mistaken for instructions by the lightweight text parser.
OPCODES = {
    "ret", "br", "switch", "indirectbr", "invoke", "callbr", "resume",
    "catchswitch", "catchret", "cleanupret", "unreachable",
    "fneg", "add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv",
    "fdiv", "urem", "srem", "frem", "shl", "lshr", "ashr", "and", "or", "xor",
    "extractelement", "insertelement", "shufflevector", "extractvalue", "insertvalue",
    "alloca", "load", "store", "fence", "cmpxchg", "atomicrmw", "getelementptr",
    "trunc", "zext", "sext", "fptrunc", "fpext", "fptoui", "fptosi", "uitofp",
    "sitofp", "ptrtoint", "inttoptr", "bitcast", "addrspacecast",
    "icmp", "fcmp", "phi", "select", "freeze", "call", "va_arg", "landingpad",
    "catchpad", "cleanuppad",
}

TERMINATORS = {
    "ret", "br", "switch", "indirectbr", "invoke", "callbr", "resume",
    "catchswitch", "catchret", "cleanupret", "unreachable",
}

CATEGORIES = {
    "int_arith": {"add", "sub", "mul", "udiv", "sdiv", "urem", "srem"},
    "float_arith": {"fneg", "fadd", "fsub", "fmul", "fdiv", "frem"},
    "bitwise": {"shl", "lshr", "ashr", "and", "or", "xor"},
    "divrem": {"udiv", "sdiv", "fdiv", "urem", "srem", "frem"},
    "compare": {"icmp", "fcmp"},
    "conversion": {
        "trunc", "zext", "sext", "fptrunc", "fpext", "fptoui", "fptosi",
        "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast", "addrspacecast",
    },
    "memory": {"alloca", "load", "store", "fence", "cmpxchg", "atomicrmw", "getelementptr"},
    "control": TERMINATORS,
    "vector": {"extractelement", "insertelement", "shufflevector"},
    "aggregate": {"extractvalue", "insertvalue"},
}

# These are common transformation passes in the artifact.  Their counts are
# useful controls without creating hundreds of sparse columns.
TRACKED_PASSES = (
    "SROAPass", "EarlyCSEPass", "PromotePass", "InstCombinePass",
    "SimplifyCFGPass", "JumpThreadingPass", "ReassociatePass", "GVNPass",
    "SCCPPass", "ADCEPass", "DSEPass", "LoopVectorizePass", "SLPVectorizerPass",
)

LABEL_RE = re.compile(r'^\s*(?:[-A-Za-z$._0-9]+|"[^"]+"):\s*$')
RESULT_RE = re.compile(r'^\s*(%(?:[-A-Za-z$._0-9]+|"[^"]+"))\s*=\s*(.*)$', re.S)
LOCAL_RE = re.compile(r'%(?:[-A-Za-z$._0-9]+|"[^"]+")')
GLOBAL_RE = re.compile(r'@(?:[-A-Za-z$._0-9]+|"[^"]+")')
INT_TYPE_RE = re.compile(r'\bi(\d+)\b')
VECTOR_TYPE_RE = re.compile(r'<\s*(\d+)\s+x\s+[^>]+>')
NUMBER_RE = re.compile(
    r'(?<![A-Za-z0-9_.$!%])(?:[-+]?\d+(?:\.\d*)?|0x[0-9A-Fa-f]+)(?![A-Za-z0-9_.$])'
)


def strip_ir_comment(line: str) -> str:
    """Remove an LLVM ``;`` comment while respecting quoted strings."""
    quoted = False
    escaped = False
    for i, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == ";" and not quoted:
            return line[:i]
    return line


def rhs_and_opcode(text: str) -> tuple[str | None, str | None, str]:
    """Return result name, opcode, and right-hand side for an IR line."""
    stripped = text.strip()
    result = None
    match = RESULT_RE.match(stripped)
    if match:
        result, stripped = match.group(1), match.group(2).lstrip()

    # Tail markers precede call and are not opcodes themselves.
    words = stripped.split(None, 2)
    if words and words[0] in {"tail", "musttail", "notail"}:
        stripped = stripped[len(words[0]):].lstrip()
    first = stripped.split(None, 1)[0] if stripped else ""
    return result, first if first in OPCODES else None, stripped


def find_function(lines: list[str], function: str) -> tuple[str, list[str]]:
    """Return the definition header and body for ``function``."""
    candidates = (f"@{function}(", f'@"{function}"(')
    start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("define ") and any(token in line for token in candidates):
            start = i
            break
    if start is None:
        raise ValueError(f"definition for @{function} not found")

    header_parts: list[str] = []
    body: list[str] = []
    in_body = False
    for line in lines[start:]:
        if not in_body:
            header_parts.append(strip_ir_comment(line).strip())
            if "{" in strip_ir_comment(line):
                in_body = True
            continue
        if line.strip() == "}":
            return " ".join(header_parts), body
        body.append(line)
    raise ValueError(f"unterminated definition for @{function}")


def logical_instructions(body: list[str]) -> list[tuple[str | None, str, str]]:
    """Coalesce multiline LLVM instructions and return (result, opcode, text)."""
    instructions: list[tuple[str | None, str, str]] = []
    current: tuple[str | None, str, str] | None = None

    def finish() -> None:
        nonlocal current
        if current is not None:
            instructions.append(current)
            current = None

    for raw in body:
        line = strip_ir_comment(raw).strip()
        if not line:
            continue
        if LABEL_RE.match(line):
            finish()
            continue
        result, opcode, rhs = rhs_and_opcode(line)
        if opcode is not None:
            finish()
            current = (result, opcode, rhs)
        elif current is not None:
            # switch cases, invoke destinations, or multiline operands
            current = (current[0], current[1], current[2] + " " + line)
    finish()
    return instructions


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def entropy(values: Iterable[int | float]) -> float:
    vals = [float(v) for v in values if v > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log(v / total) for v in vals)


def count_function_arguments(header: str) -> int:
    at = header.find("@")
    open_paren = header.find("(", at)
    close_paren = header.rfind(")")
    if open_paren < 0 or close_paren < open_paren:
        return 0
    return len(LOCAL_RE.findall(header[open_paren + 1:close_paren]))


def ir_features(path: Path, function: str) -> dict[str, int | float]:
    header, body = find_function(path.read_text(errors="replace").splitlines(), function)
    instructions = logical_instructions(body)
    opcodes = Counter(opcode for _, opcode, _ in instructions)
    n = len(instructions)

    features: dict[str, int | float] = {
        "ir_instruction_count": n,
        "function_arguments": count_function_arguments(header),
        "unique_opcode_count": len(opcodes),
        "opcode_entropy": entropy(opcodes.values()),
    }
    for opcode in sorted(OPCODES):
        if opcodes[opcode]:
            features[f"opcode_{opcode}"] = opcodes[opcode]
    for name, members in CATEGORIES.items():
        value = sum(opcodes[op] for op in members)
        features[f"category_{name}_count"] = value
        features[f"category_{name}_ratio"] = safe_ratio(value, n)

    # High-level instruction counts that are particularly relevant to Alive2.
    for opcode in (
        "alloca", "load", "store", "getelementptr", "call", "invoke", "phi",
        "select", "br", "switch", "icmp", "fcmp", "atomicrmw", "cmpxchg", "fence",
    ):
        features[f"inst_{opcode}_count"] = opcodes[opcode]
        features[f"inst_{opcode}_ratio"] = safe_ratio(opcodes[opcode], n)

    all_text = "\n".join(rhs for _, _, rhs in instructions)
    int_widths = [int(width) for width in INT_TYPE_RE.findall(all_text)]
    vector_lanes = [int(lanes) for lanes in VECTOR_TYPE_RE.findall(all_text)]
    float_mentions = sum(
        len(re.findall(rf"\b{re.escape(kind)}\b", all_text))
        for kind in ("half", "bfloat", "float", "double", "fp128", "x86_fp80", "ppc_fp128")
    )
    pointer_mentions = len(re.findall(r"\bptr\b", all_text))

    features.update({
        "integer_type_mentions": len(int_widths),
        "integer_type_mentions_per_instruction": safe_ratio(len(int_widths), n),
        "integer_width_mean": statistics.fmean(int_widths) if int_widths else 0.0,
        "integer_width_max": max(int_widths, default=0),
        "wide_integer_mentions": sum(width >= 64 for width in int_widths),
        "wide_integer_ratio": safe_ratio(sum(width >= 64 for width in int_widths), len(int_widths)),
        "float_type_mentions": float_mentions,
        "float_type_mentions_per_instruction": safe_ratio(float_mentions, n),
        "pointer_type_mentions": pointer_mentions,
        "pointer_mentions_per_instruction": safe_ratio(pointer_mentions, n),
        "vector_type_mentions": len(vector_lanes),
        "vector_type_mentions_per_instruction": safe_ratio(len(vector_lanes), n),
        "vector_lanes_max": max(vector_lanes, default=0),
        "atomic_instruction_count": len(re.findall(r"\batomic\b", all_text)) + opcodes["atomicrmw"] + opcodes["cmpxchg"],
        "volatile_instruction_count": len(re.findall(r"\bvolatile\b", all_text)),
        "memory_intrinsic_call_count": len(re.findall(r"@llvm\.(?:memcpy|memmove|memset)", all_text)),
    })

    numeric_mentions = len(NUMBER_RE.findall(all_text))
    zero_mentions = len(re.findall(r"(?<![A-Za-z0-9_.$!%])-?0(?![A-Za-z0-9_.$])", all_text))
    one_mentions = len(re.findall(r"(?<![A-Za-z0-9_.$!%])-?1(?![A-Za-z0-9_.$])", all_text))
    special_mentions = sum(
        len(re.findall(rf"\b{word}\b", all_text))
        for word in ("undef", "poison", "null")
    )

    local_use_counts: Counter[str] = Counter()
    defined_values: set[str] = set()
    dependency_depth: dict[str, int] = {}
    depths: list[int] = []
    dependency_edges = 0
    direct_calls = 0
    indirect_calls = 0
    for result, opcode, rhs in instructions:
        refs = LOCAL_RE.findall(rhs)
        local_use_counts.update(refs)
        dependency_edges += sum(ref in defined_values for ref in refs)
        depth = 1 + max((dependency_depth.get(ref, 0) for ref in refs), default=0)
        depths.append(depth)
        if result:
            defined_values.add(result)
            dependency_depth[result] = depth
        if opcode in {"call", "invoke", "callbr"}:
            call_match = re.search(r"\b(?:call|invoke|callbr)\b.*?([@%](?:[-A-Za-z$._0-9]+|\"[^\"]+\"))\s*\(", rhs)
            if call_match and call_match.group(1).startswith("@"):
                direct_calls += 1
            else:
                indirect_calls += 1

    local_refs = sum(local_use_counts.values())
    global_refs = len(GLOBAL_RE.findall(all_text))
    features.update({
        "ssa_value_reference_count": local_refs,
        "ssa_refs_per_instruction": safe_ratio(local_refs, n),
        "global_reference_count": global_refs,
        "global_refs_per_instruction": safe_ratio(global_refs, n),
        "numeric_literal_mentions": numeric_mentions,
        "numeric_literals_per_instruction": safe_ratio(numeric_mentions, n),
        "zero_literal_mentions": zero_mentions,
        "one_literal_mentions": one_mentions,
        "special_value_mentions": special_mentions,
        "special_values_per_instruction": safe_ratio(special_mentions, n),
        "max_ssa_use_count": max(local_use_counts.values(), default=0),
        "dependency_edge_count": dependency_edges,
        "dependency_edges_per_instruction": safe_ratio(dependency_edges, n),
        "dependency_depth_max": max(depths, default=0),
        "dependency_depth_mean": statistics.fmean(depths) if depths else 0.0,
        "direct_call_count": direct_calls,
        "indirect_call_count": indirect_calls,
    })

    # Terminators give a robust block count even when the entry block has no
    # printed label.  Edge counts are exact for the common acyclic constructs.
    block_count = sum(opcodes[op] for op in TERMINATORS)
    edge_count = 0
    for _, opcode, rhs in instructions:
        if opcode == "br":
            edge_count += 2 if re.match(r"br\s+i1\b", rhs) else 1
        elif opcode == "switch":
            edge_count += max(1, len(re.findall(r"\blabel\s+%", rhs)))
        elif opcode in {"invoke", "callbr"}:
            edge_count += max(2, len(re.findall(r"\blabel\s+%", rhs)))
        elif opcode == "indirectbr":
            edge_count += len(re.findall(r"\blabel\s+%", rhs))
        elif opcode in {"catchswitch"}:
            edge_count += len(re.findall(r"\blabel\s+%", rhs))
        elif opcode in {"catchret", "cleanupret"}:
            edge_count += 1
    features.update({
        "basic_block_count": block_count,
        "cfg_edge_count": edge_count,
        "cyclomatic_complexity": max(1, edge_count - block_count + 2) if block_count else 0,
        "instructions_per_block": safe_ratio(n, block_count),
    })
    return features


def pass_features(path: Path) -> dict[str, int | float | str]:
    data = json.loads(path.read_text())
    passes = data["passes"]
    diffs = [int(entry.get("diff", 0)) for entry in passes]
    changed = [(i, entry["name"], diffs[i]) for i, entry in enumerate(passes) if diffs[i] > 0]
    nonzero = [diff for diff in diffs if diff > 0]
    sorted_nonzero = sorted(nonzero, reverse=True)
    total = sum(nonzero)
    positions = [i for i, _, _ in changed]
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    mean = statistics.fmean(nonzero) if nonzero else 0.0
    median = statistics.median(nonzero) if nonzero else 0.0
    stdev = statistics.pstdev(nonzero) if len(nonzero) > 1 else 0.0
    weighted_position = (
        sum(i * diff for i, _, diff in changed) / total / max(1, len(passes) - 1)
        if total else 0.0
    )

    result: dict[str, int | float | str] = {
        "pass_count": len(passes),
        "modified_pass_count": len(nonzero),
        "unmodified_pass_count": len(passes) - len(nonzero),
        "pass_diff_total_recomputed": total,
        "pass_diff_mean_nonzero": mean,
        "pass_diff_median_nonzero": median,
        "pass_diff_max": max(nonzero, default=0),
        "pass_diff_stddev": stdev,
        "pass_diff_cv": safe_ratio(stdev, mean),
        "pass_diff_top1_share": safe_ratio(sum(sorted_nonzero[:1]), total),
        "pass_diff_top2_share": safe_ratio(sum(sorted_nonzero[:2]), total),
        "pass_diff_top4_share": safe_ratio(sum(sorted_nonzero[:4]), total),
        "pass_diff_entropy": entropy(nonzero),
        "pass_diff_effective_passes": math.exp(entropy(nonzero)) if nonzero else 0.0,
        "passes_above_mean_diff": sum(diff > mean for diff in nonzero),
        "passes_above_median_diff": sum(diff > median for diff in nonzero),
        "modified_pass_position_mean": statistics.fmean(positions) / max(1, len(passes) - 1) if positions else 0.0,
        "modified_pass_weighted_position": weighted_position,
        "modified_pass_gap_mean": statistics.fmean(gaps) if gaps else 0.0,
        "modified_pass_gap_max": max(gaps, default=0),
        "first_modified_pass_position": safe_ratio(min(positions), max(1, len(passes) - 1)) if positions else 0.0,
        "last_modified_pass_position": safe_ratio(max(positions), max(1, len(passes) - 1)) if positions else 0.0,
        "dominant_modified_pass": max(changed, key=lambda item: item[2])[1] if changed else "",
    }
    names = Counter(name for _, name, _ in changed)
    diff_by_name: Counter[str] = Counter()
    for _, name, diff in changed:
        diff_by_name[name] += diff
    for name in TRACKED_PASSES:
        result[f"changed_{name}_count"] = names[name]
        result[f"changed_{name}_diff"] = diff_by_name[name]
    return result


def locate_artifacts(funcs_dir: Path, recorded_file: str) -> tuple[str, Path, Path]:
    normalized = recorded_file.replace("\\", "/")
    marker = "/funcs/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
    elif normalized.startswith("benchmarks/funcs/"):
        relative = normalized[len("benchmarks/funcs/"):]
    else:
        raise ValueError(f"cannot derive benchmark/function from recorded path: {recorded_file}")
    relative_path = Path(relative)
    benchmark = relative_path.parts[0]
    ir_path = funcs_dir / relative_path
    pass_path = ir_path.with_name(ir_path.stem + "_passes.json")
    return benchmark, ir_path, pass_path


def flatten_record(record: dict) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "recorded_file": record["file"],
        "function": record["function"],
        "num_instructions": record.get("num_instructions", ""),
        "total_diff": record.get("total_diff", ""),
    }
    for name, value in record.items():
        if not isinstance(value, dict) or "result" not in value:
            continue
        row[f"{name}_result"] = value.get("result", "")
        row[f"{name}_time"] = value.get("time", "")
        if "num_pairs" in value:
            row[f"{name}_num_pairs"] = value["num_pairs"]
        if "sizes" in value:
            row[f"{name}_spatial_slice_count"] = len(value["sizes"])
    return row


def write_csv(rows: list[dict[str, object]], output: Path) -> list[str]:
    identity = [
        "schema_version", "benchmark", "function", "relative_ir", "recorded_file",
        "num_instructions", "total_diff",
    ]
    all_fields = set().union(*(row.keys() for row in rows))
    fields = [field for field in identity if field in all_fields]
    fields += sorted(all_fields - set(fields))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser.add_argument("--merged", type=Path, default=root / "analysis" / "merged.jsonl")
    parser.add_argument("--funcs-dir", type=Path, default=root / "benchmarks" / "funcs")
    parser.add_argument("--output", type=Path, default=here / "phase1_output" / "features.csv")
    parser.add_argument("--metadata", type=Path, help="metadata JSON path (default: beside output)")
    parser.add_argument(
        "--include-unsupported", action="store_true",
        help="include records for which vanilla alive_tv returned incorrect",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="skip records whose IR/pass files cannot be parsed instead of failing",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    raw_records = 0
    excluded = 0
    with args.merged.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw_records += 1
            record = json.loads(line)
            if not args.include_unsupported and record.get("alive_tv", {}).get("result") == "incorrect":
                excluded += 1
                continue
            try:
                benchmark, ir_path, pass_path = locate_artifacts(args.funcs_dir, record["file"])
                row = flatten_record(record)
                row.update({
                    "benchmark": benchmark,
                    "relative_ir": str(ir_path.relative_to(args.funcs_dir)),
                })
                row.update(ir_features(ir_path, record["function"]))
                row.update(pass_features(pass_path))
                rows.append(row)
            except Exception as exc:  # report complete artifact path context
                message = f"line {line_number} ({record.get('file', '?')}): {exc}"
                if not args.allow_missing:
                    print(f"error: {message}", file=sys.stderr)
                    return 1
                errors.append(message)

    if not rows:
        print("error: no rows were produced", file=sys.stderr)
        return 1
    fields = write_csv(rows, args.output)
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_merged": str(args.merged.resolve()),
        "source_funcs_dir": str(args.funcs_dir.resolve()),
        "raw_records": raw_records,
        "excluded_alive_tv_incorrect": excluded,
        "output_records": len(rows),
        "parse_errors": errors,
        "field_count": len(fields),
        "fields": fields,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(rows)} records and {len(fields)} columns to {args.output}", file=sys.stderr)
    print(f"Wrote metadata to {metadata_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
