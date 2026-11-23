#!/usr/bin/env python3
"""Collect a live LLVM pass trace and build a self-contained Phase 2 manifest.

This is the authoritative Phase 2 preprocessing stage.  It does not read the
artifact's pre-generated ``*_passes.json`` files.  One instrumented ``opt``
run per function supplies the live pass order, instruction identities,
per-pass change characteristics, whole-state characteristics, and exact IR
snapshots needed by the sampled temporal segments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from phase2_common import (
    DEFAULT_HEURISTICS,
    DESTIVA_CUTPOINT_HEURISTICS,
    HEURISTICS,
    collect_candidates,
    nonheuristic_augmentation_segments,
    read_jsonl,
    segment_features,
    segment_id,
    select_candidates,
    stable_order_key,
)


SCHEMA_VERSION = 3
PROFILE_METHOD = "live_instrumented_llvm_trace_v1"
SNAPSHOT_METHOD = "live_post_pass_ir_v1"
NOISE_RE = re.compile(r";\d+$")
HEADER_RE = re.compile(r"^\s*;\s*\*\*\* IR Dump (Before|After) (.+?) on ")
STAR_RE = re.compile(r"^\s*;\s*\*\*\*")
MD_STRING_RE = re.compile(r'^!(\d+)\s*=\s*!\{!"(\d+)"\}\s*$')
MD_INT_RE = re.compile(r"^!(\d+)\s*=\s*!\{i\d+\s+(\d+)\}\s*$")
MYMD_RE = re.compile(r"!mymd\s+!(\d+)")
MODULE_METADATA_RE = re.compile(r"^!\d+\s*=")
TRAILING_METADATA_RE = re.compile(r",\s*![A-Za-z0-9_.-]+\s+!\d+.*$")
LOCAL_RE = re.compile(r'%(?:[-A-Za-z$._0-9]+|"[^"]+")')
GLOBAL_RE = re.compile(r'@(?:[-A-Za-z$._0-9]+|"[^"]+")')
INT_TYPE_RE = re.compile(r"\bi(\d+)\b")
VECTOR_TYPE_RE = re.compile(r"<\s*(\d+)\s+x\s+[^>]+>")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.$!%])(?:[-+]?\d+(?:\.\d*)?|0x[0-9A-Fa-f]+)(?![A-Za-z0-9_.$])"
)


def clean_dump_lines(lines: list[str]) -> str:
    """Discard tracking diagnostics that older patched LLVM builds emitted."""
    last_metadata = max(
        (index for index, line in enumerate(lines) if MODULE_METADATA_RE.match(line)),
        default=-1,
    )
    if last_metadata >= 0:
        lines = lines[:last_metadata + 1]
    return "\n".join(line for line in lines if line.strip() != "Overload not found")


def parse_sections(dump_text: str) -> list[tuple[str, str, str]]:
    sections: list[tuple[str, str, str]] = []
    direction: str | None = None
    pass_name: str | None = None
    lines: list[str] = []
    for line in dump_text.splitlines():
        match = HEADER_RE.match(line)
        if match:
            if direction is not None and pass_name is not None:
                sections.append((direction, pass_name, clean_dump_lines(lines)))
            direction, pass_name = match.groups()
            lines = []
        elif STAR_RE.match(line):
            if direction is not None and pass_name is not None:
                sections.append((direction, pass_name, clean_dump_lines(lines)))
            direction = pass_name = None
            lines = []
        elif direction is not None and not NOISE_RE.search(line):
            lines.append(line)
    if direction is not None and pass_name is not None:
        sections.append((direction, pass_name, clean_dump_lines(lines)))
    return sections


def pair_sections(sections: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Return adjacent, same-name before/after leaf-pass dumps."""
    pairs: list[tuple[str, str, str]] = []
    index = 0
    while index < len(sections):
        current = sections[index]
        if (
            current[0] == "Before"
            and index + 1 < len(sections)
            and sections[index + 1][0] == "After"
            and current[1] == sections[index + 1][1]
        ):
            pairs.append((current[1], current[2], sections[index + 1][2]))
            index += 2
        else:
            index += 1
    return pairs


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stored_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def tool_output(command: list[str], cwd: Path | None = None) -> str:
    run = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return (run.stdout + run.stderr).strip()


def load_function_paths(path: Path) -> list[str]:
    values = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(value if value.endswith(".ll") else value + ".ll")
    return values


def load_excluded_profiles(path: Path) -> set[str]:
    """Load functions already collected by a compatible live profile file."""
    excluded: set[str] = set()
    for row in read_jsonl(path):
        relative_ir = str(row.get("relative_ir", ""))
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("profile_method") != PROFILE_METHOD
            or not relative_ir
            or relative_ir in excluded
        ):
            raise ValueError(f"invalid or duplicate live profile in {path}: {relative_ir!r}")
        excluded.add(relative_ir)
    if not excluded:
        raise ValueError(f"exclusion profile file is empty: {path}")
    return excluded


def profile_trace_identity(profile: dict[str, object]) -> str:
    """Identify the pass order and change data that define segment indices."""
    trace = {
        "function": profile.get("function"),
        "pipeline": profile.get("pipeline"),
        "pass_count": profile.get("pass_count"),
        "pass_sequence_sha256": profile.get("pass_sequence_sha256"),
        "passes": profile.get("passes"),
    }
    return text_sha256(json.dumps(trace, sort_keys=True))


def load_excluded_segments(
    paths: list[Path],
) -> tuple[set[tuple[str, int, int]], dict[str, str]]:
    """Load exact live segments and their pass-trace identities.

    Excluding a numeric index pair is safe only when the current trace has the
    same pass sequence and change data as the earlier trace.
    """
    excluded: set[tuple[str, int, int]] = set()
    trace_identities: dict[str, str] = {}
    for path in paths:
        metadata_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise ValueError(f"exclusion manifest or metadata not found: {path}")
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("schema_version") != SCHEMA_VERSION
            or not metadata.get("complete")
            or metadata.get("profile_method") != PROFILE_METHOD
            or metadata.get("snapshot_method") != SNAPSHOT_METHOD
            or metadata.get("manifest_sha256") != file_sha256(path)
        ):
            raise ValueError(f"incomplete or incompatible exclusion manifest: {path}")

        recorded_profiles = Path(str(metadata.get("profiles", "")))
        profile_path = recorded_profiles
        if not profile_path.is_file():
            profile_path = path.parent / recorded_profiles.name
        if (
            not profile_path.is_file()
            or metadata.get("profiles_sha256") != file_sha256(profile_path)
        ):
            raise ValueError(f"profiles for exclusion manifest are unavailable or modified: {path}")

        profiles: dict[str, tuple[str, str]] = {}
        for row in read_jsonl(profile_path):
            relative_ir = str(row.get("relative_ir", ""))
            sequence = str(row.get("pass_sequence_sha256", ""))
            if (
                row.get("schema_version") != SCHEMA_VERSION
                or row.get("profile_method") != PROFILE_METHOD
                or not relative_ir
                or not sequence
                or relative_ir in profiles
            ):
                raise ValueError(f"invalid live profile in {profile_path}: {relative_ir!r}")
            digest = text_sha256(json.dumps(row, sort_keys=True) + "\n")
            profiles[relative_ir] = (profile_trace_identity(row), digest)

        for row in read_jsonl(path):
            relative_ir = str(row.get("relative_ir", ""))
            try:
                start = int(row["start_index"])
                end = int(row["end_index"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid segment indices in {path}") from error
            if (
                row.get("schema_version") != SCHEMA_VERSION
                or row.get("profile_method") != PROFILE_METHOD
                or row.get("snapshot_method") != SNAPSHOT_METHOD
                or relative_ir not in profiles
                or not (0 <= start < end)
                or row.get("segment_id") != segment_id(relative_ir, start, end)
                or row.get("profile_record_sha256") != profiles[relative_ir][1]
            ):
                raise ValueError(
                    f"invalid live segment in exclusion manifest {path}: "
                    f"{relative_ir}:{start}:{end}"
                )
            trace_identity = profiles[relative_ir][0]
            previous = trace_identities.setdefault(relative_ir, trace_identity)
            if previous != trace_identity:
                raise ValueError(f"exclusion manifests disagree on live trace: {relative_ir}")
            excluded.add((relative_ir, start, end))
    return excluded, trace_identities


def sample_functions(paths: list[str], count: int, seed: int) -> list[str]:
    if count <= 0 or count >= len(paths):
        return sorted(paths)
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        groups[Path(path).parts[0]].append(path)
    for benchmark, values in groups.items():
        values.sort(key=lambda value: stable_order_key(seed, benchmark, value))
    offsets: Counter[str] = Counter()
    selected: list[str] = []
    while len(selected) < count:
        progressed = False
        for benchmark in sorted(groups):
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


def discover_function_name(ir_text: str, preferred: str) -> str:
    """Return the defined function, allowing prepared filenames to add prefixes."""
    definitions: list[str] = []
    for line in ir_text.splitlines():
        if not line.lstrip().startswith("define "):
            continue
        match = GLOBAL_RE.search(line)
        if match:
            token = match.group(0)[1:]
            definitions.append(token[1:-1] if token.startswith('"') else token)
    if preferred in definitions:
        return preferred
    if len(definitions) == 1:
        return definitions[0]
    raise ValueError(
        f"cannot identify function for {preferred!r}; definitions={definitions!r}"
    )


def metadata_ids(ir_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in ir_text.splitlines():
        match = MD_STRING_RE.match(line.strip()) or MD_INT_RE.match(line.strip())
        if match:
            result[match.group(1)] = match.group(2)
    return result


def tracked_instruction_groups(
    ir_text: str, function: str, find_function, logical_instructions,
) -> dict[str, list[tuple[str, str]]]:
    """Group instructions by ``!mymd`` ID, including IDs copied by cloning."""
    md_ids = metadata_ids(ir_text)
    _, body = find_function(ir_text.splitlines(), function)
    result: dict[str, list[tuple[str, str]]] = defaultdict(list)
    untracked = 0
    for value, opcode, rhs in logical_instructions(body):
        text = f"{value} = {rhs}" if value else rhs
        attachment = MYMD_RE.search(text)
        if attachment is None or attachment.group(1) not in md_ids:
            untracked += 1
            continue
        stable_id = md_ids[attachment.group(1)]
        result[stable_id].append(
            (opcode, TRAILING_METADATA_RE.sub("", text).strip())
        )
    if untracked:
        raise ValueError(f"{untracked} instructions in @{function} lack resolvable !mymd IDs")
    return result


def summarize_instructions(records: Iterable[tuple[str, str]]) -> dict[str, int]:
    rows = list(records)
    opcodes = Counter(opcode for opcode, _ in rows)
    text = "\n".join(value for _, value in rows)
    categories = {
        "arithmetic": {"add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv", "fdiv", "urem", "srem", "frem"},
        "bitwise": {"shl", "lshr", "ashr", "and", "or", "xor"},
        "comparison": {"icmp", "fcmp"},
        "conversion": {"trunc", "zext", "sext", "fptrunc", "fpext", "fptoui", "fptosi", "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast", "addrspacecast"},
        "memory": {"alloca", "load", "store", "fence", "cmpxchg", "atomicrmw", "getelementptr"},
        "control": {"ret", "br", "switch", "indirectbr", "invoke", "callbr", "resume", "unreachable"},
        "call": {"call", "invoke", "callbr"},
        "vector": {"extractelement", "insertelement", "shufflevector"},
    }
    result: dict[str, int] = {"instruction_count": len(rows)}
    for opcode, count in sorted(opcodes.items()):
        result[f"opcode_{opcode}_count"] = count
    for name, members in categories.items():
        result[f"category_{name}_count"] = sum(opcodes[item] for item in members)
    int_widths = [int(value) for value in INT_TYPE_RE.findall(text)]
    result.update({
        "local_operand_references": len(LOCAL_RE.findall(text)),
        "global_operand_references": len(GLOBAL_RE.findall(text)),
        "constant_mentions": len(NUMBER_RE.findall(text)),
        "integer_type_mentions": len(int_widths),
        "wide_integer_mentions": sum(value >= 64 for value in int_widths),
        "pointer_type_mentions": len(re.findall(r"\bptr\b", text)),
        "float_type_mentions": sum(len(re.findall(rf"\b{name}\b", text)) for name in ("half", "bfloat", "float", "double", "fp128", "x86_fp80", "ppc_fp128")),
        "vector_type_mentions": len(VECTOR_TYPE_RE.findall(text)),
        "volatile_instruction_count": len(re.findall(r"\bvolatile\b", text)),
        "atomic_instruction_count": len(re.findall(r"\batomic\b", text)) + opcodes["atomicrmw"] + opcodes["cmpxchg"],
        "memory_intrinsic_call_count": len(re.findall(r"@llvm\.(?:memcpy|memmove|memset)", text)),
        "special_value_mentions": sum(len(re.findall(rf"\b{name}\b", text)) for name in ("undef", "poison", "null")),
    })
    return result


def change_summary(
    before: str, after: str, function: str, find_function, logical_instructions,
) -> dict[str, object]:
    pre = tracked_instruction_groups(before, function, find_function, logical_instructions)
    post = tracked_instruction_groups(after, function, find_function, logical_instructions)
    removed: list[tuple[str, str]] = []
    added: list[tuple[str, str]] = []
    modified_pre: list[tuple[str, str]] = []
    modified_post: list[tuple[str, str]] = []
    for stable_id in sorted(set(pre) | set(post)):
        pre_remaining = list(pre.get(stable_id, []))
        post_remaining = list(post.get(stable_id, []))

        # LLVM cloning may copy arbitrary metadata. Match text-identical copies
        # first, then treat the remaining one-to-one pairs as modifications.
        unmatched_pre: list[tuple[str, str]] = []
        for record in pre_remaining:
            try:
                match = post_remaining.index(record)
            except ValueError:
                unmatched_pre.append(record)
            else:
                post_remaining.pop(match)
        pair_count = min(len(unmatched_pre), len(post_remaining))
        modified_pre.extend(unmatched_pre[:pair_count])
        modified_post.extend(post_remaining[:pair_count])
        removed.extend(unmatched_pre[pair_count:])
        added.extend(post_remaining[pair_count:])

    groups = {
        "removed": removed,
        "added": added,
        "modified_pre": modified_pre,
        "modified_post": modified_post,
        "touched_pre": removed + modified_pre,
        "touched_post": added + modified_post,
    }
    features: dict[str, int] = {}
    for prefix, records in groups.items():
        for key, value in summarize_instructions(records).items():
            features[f"{prefix}_{key}"] = value
    return {
        "before_instruction_count": sum(len(records) for records in pre.values()),
        "after_instruction_count": sum(len(records) for records in post.values()),
        "removed_count": len(removed),
        "added_count": len(added),
        "modified_count": len(modified_pre),
        "diff": len(removed) + len(added) + len(modified_pre),
        "features": features,
    }


def flatten_boundary_features(start: dict[str, object], end: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in sorted(set(start) | set(end)):
        start_value = start.get(key, 0)
        end_value = end.get(key, 0)
        result[f"start_{key}"] = start_value
        result[f"end_{key}"] = end_value
        if isinstance(start_value, (int, float)) and isinstance(end_value, (int, float)):
            result[f"delta_{key}"] = end_value - start_value
            result[f"abs_delta_{key}"] = abs(end_value - start_value)
    return result


def cumulative_change_features(passes: list[dict[str, object]], start: int, end: int) -> dict[str, int | float]:
    result: Counter[str] = Counter()
    for entry in passes[start + 1:end + 1]:
        for key, value in entry["change_features"].items():
            if isinstance(value, (int, float)):
                result[f"cumulative_{key}"] += value
    return dict(result)


def write_snapshot(path: Path, text: str) -> tuple[str, str]:
    normalized = text.rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized)
    return normalized, text_sha256(normalized)


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "mp" / "phase1"))
    from phase1_extract import find_function, ir_features_text, logical_instructions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--functions", type=Path, default=root / "benchmarks" / "funcs" / "acyclic.txt")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--opt", type=Path, default=root / "llvm-project" / "build" / "bin" / "opt")
    parser.add_argument("--sample-functions", type=int, default=60)
    parser.add_argument(
        "--exclude-profiles", type=Path,
        help="live profile JSONL whose functions should be omitted from this batch",
    )
    parser.add_argument(
        "--exclude-manifest", type=Path, action="append", default=[],
        help="complete live manifest whose exact segment pairs should be omitted; repeatable",
    )
    parser.add_argument("--limit-functions", type=int, default=0)
    parser.add_argument("--max-segments-per-function", type=int, default=12)
    parser.add_argument("--heuristics", nargs="+", choices=HEURISTICS, default=list(DEFAULT_HEURISTICS))
    parser.add_argument(
        "--sampling-policy",
        choices=("heuristic", "nonheuristic_augmentation"),
        default="heuristic",
        help="candidate-generation policy; augmentation always selects six non-heuristic strata",
    )
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    for name in ("root", "functions", "profiles", "manifest", "snapshots_dir", "opt"):
        setattr(args, name, getattr(args, name).resolve())
    if args.exclude_profiles is not None:
        args.exclude_profiles = args.exclude_profiles.resolve()
    args.exclude_manifest = [path.resolve() for path in args.exclude_manifest]
    if not args.opt.is_file():
        raise SystemExit(f"LLVM opt not found: {args.opt}")

    sampling_frame = sample_functions(
        load_function_paths(args.functions), args.sample_functions, args.seed,
    )
    try:
        excluded = (
            load_excluded_profiles(args.exclude_profiles)
            if args.exclude_profiles is not None else set()
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        excluded_segments, excluded_trace_identities = load_excluded_segments(
            args.exclude_manifest,
        )
    except (ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if args.sampling_policy == "nonheuristic_augmentation" and not args.exclude_manifest:
        raise SystemExit(
            "nonheuristic augmentation requires the existing cohort manifests "
            "via --exclude-manifest"
        )
    effective_sampling_policy = (
        "nonheuristic_augmentation_v1"
        if args.sampling_policy == "nonheuristic_augmentation"
        else "heuristic"
    )
    excluded_in_frame = set(sampling_frame) & excluded
    paths = [path for path in sampling_frame if path not in excluded]
    if args.limit_functions > 0:
        paths = paths[:args.limit_functions]
    args.profiles.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.snapshots_dir.mkdir(parents=True, exist_ok=True)

    existing_profiles = read_jsonl(args.profiles)
    existing_manifest = read_jsonl(args.manifest)
    if args.overwrite:
        existing_profiles = []
        existing_manifest = []
        args.profiles.write_text("")
        args.manifest.write_text("")
    elif (existing_profiles or existing_manifest) and not args.resume:
        raise SystemExit("outputs already contain records; use --resume or choose new filenames")
    elif not args.profiles.exists():
        args.profiles.write_text("")
    if not args.manifest.exists():
        args.manifest.write_text("")

    profile_by_ir: dict[str, dict[str, object]] = {}
    for row in existing_profiles:
        relative_ir = str(row.get("relative_ir", ""))
        if not relative_ir or relative_ir in profile_by_ir or row.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(f"invalid or duplicate live profile for {relative_ir!r}")
        profile_by_ir[relative_ir] = row
    manifest_by_ir: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_segments: set[str] = set()
    for row in existing_manifest:
        segment = str(row.get("segment_id", ""))
        if not segment or segment in seen_segments or row.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(f"invalid or duplicate live segment {segment!r}")
        seen_segments.add(segment)
        manifest_by_ir[str(row["relative_ir"])].append(row)
    extra_manifest_functions = set(manifest_by_ir) - set(profile_by_ir)
    if extra_manifest_functions:
        raise SystemExit("resume manifest contains functions absent from profiles")
    for relative_ir, profile in profile_by_ir.items():
        rows = manifest_by_ir.get(relative_ir, [])
        expected = profile.get("selected_segment_count")
        if expected is None and not rows:
            raise SystemExit("resume files disagree on completed functions")
        if expected is not None and len(rows) != int(expected):
            raise SystemExit(
                f"resume segment count mismatch for {relative_ir}: "
                f"expected {expected}, found {len(rows)}"
            )
        manifest_by_ir.setdefault(relative_ir, [])

    errors: list[dict[str, object]] = []
    for number, relative_ir in enumerate(paths, 1):
        if relative_ir in profile_by_ir:
            print(f"[{number}/{len(paths)}] {relative_ir}: already collected", flush=True)
            continue
        ir_path = args.root / "benchmarks" / "funcs" / relative_ir
        if not ir_path.is_file():
            errors.append({"relative_ir": relative_ir, "error": "IR file not found"})
            continue
        try:
            function = discover_function_name(
                ir_path.read_text(errors="replace"), ir_path.stem,
            )
        except ValueError as error:
            errors.append({"relative_ir": relative_ir, "error": str(error)})
            continue
        command = [
            str(args.opt), "-passes=default<O3>", "--print-before-all",
            "--print-after-all", "--print-module-scope", "-disable-output", str(ir_path),
        ]
        print(f"[{number}/{len(paths)}] tracing {relative_ir}", flush=True)
        run = subprocess.run(command, capture_output=True, text=True)
        pairs = pair_sections(parse_sections(run.stderr))
        if run.returncode != 0 or not pairs:
            errors.append({
                "relative_ir": relative_ir,
                "error": "opt failed or produced no paired leaf-pass dumps",
                "returncode": run.returncode,
                "stderr_tail": run.stderr[-4000:],
            })
            continue
        try:
            occurrences: Counter[str] = Counter()
            passes: list[dict[str, object]] = []
            for index, (name, before, after) in enumerate(pairs):
                changes = change_summary(before, after, function, find_function, logical_instructions)
                occurrence = occurrences[name]
                occurrences[name] += 1
                passes.append({
                    "index": index,
                    "name": name,
                    "occurrence": occurrence,
                    "diff": changes["diff"],
                    "removed": changes["removed_count"],
                    "added": changes["added_count"],
                    "modified": changes["modified_count"],
                    "n_input": changes["before_instruction_count"],
                    "change_features": changes["features"],
                })

            pass_sequence_digest = text_sha256(
                "\0".join(str(row["name"]) for row in passes)
            )
            current_trace_identity = profile_trace_identity({
                "function": function,
                "pipeline": "default<O3>",
                "pass_count": len(passes),
                "pass_sequence_sha256": pass_sequence_digest,
                "passes": passes,
            })
            excluded_trace_identity = excluded_trace_identities.get(relative_ir)
            if (
                excluded_trace_identity is not None
                and excluded_trace_identity != current_trace_identity
            ):
                raise ValueError(
                    "live pass trace differs from exclusion manifest; "
                    "segment indices cannot be safely excluded"
                )
            if args.sampling_policy == "nonheuristic_augmentation":
                local_excluded = {
                    (start, end)
                    for excluded_ir, start, end in excluded_segments
                    if excluded_ir == relative_ir
                }
                selection_details = nonheuristic_augmentation_segments(
                    passes, local_excluded, args.seed, relative_ir,
                )
                candidates = {
                    pair: {"nonheuristic_augmentation"}
                    for pair in selection_details
                }
                chosen = sorted(selection_details)
            else:
                selection_details = {}
                candidates = {
                    pair: selected_by
                    for pair, selected_by in collect_candidates(passes, args.heuristics).items()
                    if (relative_ir, pair[0], pair[1]) not in excluded_segments
                }
                chosen = sorted(select_candidates(
                    candidates, args.max_segments_per_function, args.seed, relative_ir,
                ))
            endpoint_indices = sorted({value for pair in chosen for value in pair})
            snapshot_data: dict[int, tuple[str, str]] = {}
            state_features: dict[int, dict[str, int | float]] = {}
            base = args.snapshots_dir / Path(relative_ir).with_suffix("")
            for index in endpoint_indices:
                snapshot_path = base / f"after-{index:04d}.ll"
                _, digest = write_snapshot(snapshot_path, pairs[index][2])
                snapshot_data[index] = (stored_path(snapshot_path, args.root), digest)
                state_features[index] = ir_features_text(pairs[index][2], function)

            input_features = ir_features_text(pairs[0][1], function)
            profile = {
                "schema_version": SCHEMA_VERSION,
                "profile_method": PROFILE_METHOD,
                "relative_ir": relative_ir,
                "benchmark": Path(relative_ir).parts[0],
                "function": function,
                "pipeline": "default<O3>",
                "pass_count": len(passes),
                "input_features": input_features,
                "pass_sequence_sha256": pass_sequence_digest,
                "selection_policy": effective_sampling_policy,
                "selected_segment_count": len(chosen),
                "passes": passes,
            }
            profile_json = json.dumps(profile, sort_keys=True)
            profile_digest = text_sha256(profile_json + "\n")
            segments: list[dict[str, object]] = []
            for start, end in chosen:
                start_path, start_hash = snapshot_data[start]
                end_path, end_hash = snapshot_data[end]
                net = change_summary(
                    pairs[start][2], pairs[end][2], function, find_function, logical_instructions,
                )
                features = segment_features(passes, start, end)
                features.update(cumulative_change_features(passes, start, end))
                features.update({f"net_{key}": value for key, value in net["features"].items()})
                features.update({
                    "net_removed_count": net["removed_count"],
                    "net_added_count": net["added_count"],
                    "net_modified_count": net["modified_count"],
                    "net_diff": net["diff"],
                })
                segment = {
                    "schema_version": SCHEMA_VERSION,
                    "profile_method": PROFILE_METHOD,
                    "snapshot_method": SNAPSHOT_METHOD,
                    "profile_record_sha256": profile_digest,
                    "segment_id": segment_id(relative_ir, start, end),
                    "benchmark": Path(relative_ir).parts[0],
                    "function": function,
                    "relative_ir": relative_ir,
                    "pass_count": len(passes),
                    "start_index": start,
                    "end_index": end,
                    "index_pair": f"{start}:{end}",
                    "start_pass": passes[start]["name"],
                    "start_occurrence": passes[start]["occurrence"],
                    "end_pass": passes[end]["name"],
                    "end_occurrence": passes[end]["occurrence"],
                    "selected_by": sorted(candidates[(start, end)]),
                    "start_snapshot": start_path,
                    "end_snapshot": end_path,
                    "start_snapshot_sha256": start_hash,
                    "end_snapshot_sha256": end_hash,
                    "input_features": input_features,
                    "features": features,
                    "boundary_features": flatten_boundary_features(
                        state_features[start], state_features[end],
                    ),
                }
                segment.update(selection_details.get((start, end), {
                    "selection_policy": "heuristic",
                }))
                segments.append(segment)
        except Exception as error:
            errors.append({"relative_ir": relative_ir, "error": str(error)})
            continue

        append_jsonl(args.profiles, profile)
        for segment in segments:
            append_jsonl(args.manifest, segment)
        profile_by_ir[relative_ir] = profile
        manifest_by_ir[relative_ir] = segments
        print(
            f"[{number}/{len(paths)}] {relative_ir}: {len(passes)} live passes, "
            f"{len(segments)} segments, {len(endpoint_indices)} snapshots",
            flush=True,
        )

    missing = sorted(set(paths) - set(profile_by_ir))
    extra = sorted(set(profile_by_ir) - set(paths))
    complete = not errors and not missing and not extra
    llvm_commit = (
        tool_output(["git", "rev-parse", "HEAD"], cwd=args.root / "llvm-project")
        if (args.root / "llvm-project" / ".git").exists() else "unknown"
    )
    common_metadata = {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "profile_method": PROFILE_METHOD,
        "snapshot_method": SNAPSHOT_METHOD,
        "functions": str(args.functions),
        "requested_functions": len(paths),
        "completed_functions": len(set(paths) & set(profile_by_ir)),
        "sample_functions": args.sample_functions,
        "sampling_frame_functions": len(sampling_frame),
        "exclude_profiles": str(args.exclude_profiles) if args.exclude_profiles else None,
        "exclude_profiles_sha256": (
            file_sha256(args.exclude_profiles) if args.exclude_profiles else None
        ),
        "excluded_functions": len(excluded_in_frame),
        "exclude_manifests": [str(path) for path in args.exclude_manifest],
        "exclude_manifest_sha256": [file_sha256(path) for path in args.exclude_manifest],
        "excluded_segment_count": len(excluded_segments),
        "sampling_policy": effective_sampling_policy,
        "max_segments_per_function": (
            args.max_segments_per_function
            if args.sampling_policy == "heuristic" else None
        ),
        "augmentation_segments_per_function": (
            6 if args.sampling_policy == "nonheuristic_augmentation" else None
        ),
        "heuristics": (
            list(args.heuristics) if args.sampling_policy == "heuristic" else []
        ),
        "excluded_heuristics": (
            list(DESTIVA_CUTPOINT_HEURISTICS)
            if args.sampling_policy == "nonheuristic_augmentation" else []
        ),
        "seed": args.seed,
        "opt": str(args.opt),
        "opt_sha256": file_sha256(args.opt),
        "opt_version": tool_output([str(args.opt), "--version"]),
        "llvm_git_commit": llvm_commit,
        "destiva_git_commit": tool_output(["git", "rev-parse", "HEAD"], cwd=args.root),
        "tracking_overlay_sha256": file_sha256(
            Path(__file__).with_name("quiet-tracking-output.patch")
        ),
        "pipeline": "default<O3>",
        "missing_functions": missing,
        "unexpected_functions": extra,
        "errors": errors,
    }
    profiles_metadata = dict(common_metadata)
    profiles_metadata.update({
        "record_count": len(profile_by_ir),
        "profiles_sha256": file_sha256(args.profiles),
    })
    manifest_metadata = dict(common_metadata)
    manifest_metadata.update({
        "profiles": str(args.profiles),
        "profiles_sha256": file_sha256(args.profiles),
        "segment_count": sum(len(value) for value in manifest_by_ir.values()),
        "manifest_sha256": file_sha256(args.manifest),
        "snapshots_dir": str(args.snapshots_dir),
    })
    args.profiles.with_suffix(".metadata.json").write_text(
        json.dumps(profiles_metadata, indent=2, sort_keys=True) + "\n"
    )
    args.manifest.with_suffix(".metadata.json").write_text(
        json.dumps(manifest_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"profiles: {args.profiles}")
    print(f"manifest: {args.manifest}")
    if not complete:
        raise SystemExit("live collection is incomplete; see manifest metadata and resume after fixing errors")


if __name__ == "__main__":
    main()
