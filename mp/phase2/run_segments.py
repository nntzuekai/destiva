#!/usr/bin/env python3
"""Validate exact Phase 2 IR snapshot pairs with resume support."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from phase2_common import read_jsonl


SCHEMA_VERSION = 3
MEASUREMENT_METHOD = "live_snapshot_pair_v2"
PROFILE_METHOD = "live_instrumented_llvm_trace_v1"
SNAPSHOT_METHOD = "live_post_pass_ir_v1"
TIME_RE = re.compile(
    r"PHASE2_TIME:\s+function=(\S*)\s+seconds=([0-9.eE+-]+)",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_snapshot(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def command_for(record: dict[str, object], args: argparse.Namespace) -> list[str]:
    command = [
        str(args.alive_tv),
        "--smt-to=0",
        "--time-verify",
        f"--func={record['function']}",
    ]
    if args.mode == "combined":
        command.append("--decompose")
    command.extend([
        str(record["_start_snapshot_path"]),
        str(record["_end_snapshot_path"]),
    ])
    return command


def execution_command(command: Sequence[str], memory_gb: int) -> list[str]:
    if memory_gb <= 0:
        return list(command)
    prlimit = shutil.which("prlimit")
    if not prlimit:
        raise RuntimeError("prlimit is required when --memory-gb is nonzero")
    return [prlimit, f"--as={memory_gb * 1024 ** 3}", "--", *command]


def classify(returncode: int | None, stdout: str, stderr: str, timed_out: bool) -> str:
    text = stdout + "\n" + stderr
    lower = text.lower()
    if timed_out or "error: timeout" in lower:
        return "timeout"
    if "syntactically equal" in lower:
        return "syntactic_skip"
    if "program doesn't type check" in lower or "llvm2alive failed" in lower:
        return "translation_error"
    if "unsupported" in lower and "transformation doesn't verify" not in lower:
        return "unsupported"
    if "transformation doesn't verify" in lower and "(unsound)" in lower:
        return "counterexample"
    if "transformation doesn't verify" in lower:
        return "verification_failure"
    if returncode == 0:
        return "correct"
    return "crash"


def tail(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[-limit:]


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def tool_version(path: Path) -> str:
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def load_live_manifest(manifest_path: Path, root: Path) -> tuple[list[dict[str, object]], str]:
    metadata_path = manifest_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise ValueError(f"manifest metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("profile_method") != PROFILE_METHOD
        or metadata.get("snapshot_method") != SNAPSHOT_METHOD
        or not metadata.get("complete")
        or metadata.get("errors")
    ):
        raise ValueError(f"manifest is not a complete live schema-v3 collection: {metadata_path}")
    rows = read_jsonl(manifest_path)
    seen: set[str] = set()
    hash_cache: dict[Path, str] = {}
    for row in rows:
        segment_id = str(row.get("segment_id", ""))
        if not segment_id or segment_id in seen:
            raise ValueError(f"duplicate or empty segment_id in {manifest_path}: {segment_id!r}")
        seen.add(segment_id)
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("profile_method") != PROFILE_METHOD
            or row.get("snapshot_method") != SNAPSHOT_METHOD
        ):
            raise ValueError(f"unsupported live manifest record for {segment_id}")
        for side in ("start", "end"):
            path = resolve_snapshot(row.get(f"{side}_snapshot", ""), root)
            if not path.is_file():
                raise ValueError(f"missing {side} snapshot for {segment_id}: {path}")
            actual_hash = hash_cache.setdefault(path, file_sha256(path))
            expected_hash = str(row.get(f"{side}_snapshot_sha256", ""))
            if actual_hash != expected_hash:
                raise ValueError(
                    f"{side} snapshot hash mismatch for {segment_id}: {path}"
                )
            row[f"_{side}_snapshot_path"] = path
    actual_digest = file_sha256(manifest_path)
    if metadata.get("manifest_sha256") != actual_digest:
        raise ValueError(f"manifest digest does not match metadata: {manifest_path}")
    if metadata.get("segment_count") != len(rows):
        raise ValueError(f"manifest record count does not match metadata: {manifest_path}")
    return rows, actual_digest


def validate_existing_results(path: Path, manifest_digest: str) -> set[str]:
    done: set[str] = set()
    for record in read_jsonl(path):
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"legacy results cannot be resumed with the exact-snapshot runner: {path}"
            )
        if record.get("measurement_method") != MEASUREMENT_METHOD:
            raise ValueError(f"incompatible measurement method in {path}")
        if record.get("manifest_sha256") != manifest_digest:
            raise ValueError(f"results were produced from a different live manifest: {path}")
        run_id = str(record.get("run_id", ""))
        if not run_id or run_id in done:
            raise ValueError(f"duplicate or empty run_id in {path}: {run_id!r}")
        done.add(run_id)
    return done


def run_one(
    record: dict[str, object], repeat: int, args: argparse.Namespace,
    environment: dict[str, object], manifest_digest: str,
) -> dict[str, object]:
    command = command_for(record, args)
    launch_command = execution_command(command, args.memory_gb)
    started = time.time()
    begin = time.perf_counter()
    timed_out = False
    stdout = ""
    stderr = ""
    returncode: int | None = None
    process = subprocess.Popen(
        launch_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        returncode = process.returncode
    elapsed = time.perf_counter() - begin
    outcome = classify(returncode, stdout, stderr, timed_out)
    telemetry = [
        {"function": function, "seconds": float(seconds)}
        for function, seconds in TIME_RE.findall(stdout + "\n" + stderr)
    ]
    function_times = [
        entry["seconds"] for entry in telemetry
        if entry["function"] == str(record["function"])
    ]
    measurement_valid = outcome == "timeout" or len(function_times) == 1
    run_id = f"{record['segment_id']}:{args.mode}:r{repeat}"
    result = {
        "schema_version": SCHEMA_VERSION,
        "measurement_method": MEASUREMENT_METHOD,
        "manifest_sha256": manifest_digest,
        "run_id": run_id,
        "segment_id": record["segment_id"],
        "mode": args.mode,
        "repeat": repeat,
        "benchmark": record["benchmark"],
        "function": record["function"],
        "relative_ir": record["relative_ir"],
        "start_index": record["start_index"],
        "end_index": record["end_index"],
        "selected_by": record["selected_by"],
        "snapshot_pair": {
            "start": record["start_snapshot"],
            "end": record["end_snapshot"],
            "start_sha256": record["start_snapshot_sha256"],
            "end_sha256": record["end_snapshot_sha256"],
        },
        "outcome": {
            "verdict": outcome,
            "censored": outcome == "timeout",
            "measurement_valid": measurement_valid,
            "wall_time_seconds": round(elapsed, 6),
            "verification_time_seconds": max(function_times) if function_times else None,
            "returncode": returncode,
            "telemetry": telemetry,
        },
        "environment": environment,
        "started_unix": started,
        "command": command,
        "launch_command": launch_command,
    }
    if (
        outcome not in {"correct", "syntactic_skip"}
        or not measurement_valid
        or args.store_success_output
    ):
        result["stdout_tail"] = tail(stdout)
        result["stderr_tail"] = tail(stderr)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alive-tv", type=Path,
                        default=root / "alive2-decomp" / "build" / "alive-tv")
    parser.add_argument("--mode", choices=("temporal", "combined"), default="temporal")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--memory-gb", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="limit pending runs after resume filtering; 0 means no limit")
    parser.add_argument("--execute", action="store_true",
                        help="actually execute; without this flag only commands are previewed")
    parser.add_argument("--store-success-output", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    args.alive_tv = args.alive_tv.resolve()

    try:
        records, manifest_digest = load_live_manifest(args.manifest, args.root)
        if not records:
            raise ValueError(f"manifest is empty: {args.manifest}")
        done = validate_existing_results(args.output, manifest_digest)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    tasks = []
    for record in records:
        for repeat in range(args.repeats):
            run_id = f"{record['segment_id']}:{args.mode}:r{repeat}"
            if run_id not in done:
                tasks.append((record, repeat))
    if args.limit > 0:
        tasks = tasks[:args.limit]

    print(f"manifest segments: {len(records)}")
    print(f"validated exact live snapshot pairs: {len(records)}")
    print(f"already completed: {len(done)}")
    print(f"pending {args.mode} runs: {len(tasks)}")
    if not args.execute:
        for record, _ in tasks[:10]:
            print(shlex.join(execution_command(command_for(record, args), args.memory_gb)))
        if len(tasks) > 10:
            print(f"... {len(tasks) - 10} additional commands")
        print("dry run only; pass --execute on the experiment server")
        return

    if not args.alive_tv.exists():
        raise SystemExit(f"validator not found: {args.alive_tv}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        "git_commit": git_commit(args.root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "alive_tv": str(args.alive_tv),
        "alive_tv_sha256": file_sha256(args.alive_tv),
        "alive_tv_version": tool_version(args.alive_tv),
        "timeout_seconds": args.timeout,
        "memory_gb": args.memory_gb,
        "jobs": args.jobs,
    }

    completed = 0
    with args.output.open("a") as output, concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_one, record, repeat, args, environment, manifest_digest,
            ): (record, repeat)
            for record, repeat in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            record, repeat = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "measurement_method": MEASUREMENT_METHOD,
                    "manifest_sha256": manifest_digest,
                    "run_id": f"{record['segment_id']}:{args.mode}:r{repeat}",
                    "segment_id": record["segment_id"],
                    "mode": args.mode,
                    "repeat": repeat,
                    "benchmark": record["benchmark"],
                    "function": record["function"],
                    "relative_ir": record["relative_ir"],
                    "start_index": record["start_index"],
                    "end_index": record["end_index"],
                    "selected_by": record["selected_by"],
                    "snapshot_pair": {
                        "start": record["start_snapshot"],
                        "end": record["end_snapshot"],
                        "start_sha256": record["start_snapshot_sha256"],
                        "end_sha256": record["end_snapshot_sha256"],
                    },
                    "outcome": {
                        "verdict": "runner_error",
                        "censored": False,
                        "measurement_valid": False,
                    },
                    "runner_error": repr(error),
                    "environment": environment,
                }
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            completed += 1
            verdict = result["outcome"]["verdict"]
            valid = result["outcome"].get("measurement_valid", False)
            print(
                f"[{completed}/{len(tasks)}] {result['segment_id']} "
                f"{verdict} valid={valid}",
                flush=True,
            )


if __name__ == "__main__":
    main()
