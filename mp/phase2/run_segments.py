#!/usr/bin/env python3
"""Run an exact-index Phase 2 segment manifest with resume support."""

from __future__ import annotations

import argparse
import concurrent.futures
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


TIME_RE = re.compile(r"PHASE2_TIME:\s+index=(-?\d+).*?seconds=([0-9.eE+-]+)")


def command_for(record: dict[str, object], args: argparse.Namespace) -> list[str]:
    command = [
        str(args.opt_alive),
        "-tv-smt-to=0",
        f"-tv-check-index-pairs={record['start_index']}:{record['end_index']}",
        "-tv-time-verify",
    ]
    if args.mode == "combined":
        command.append("--decompose")
    command.extend([
        "-O3",
        str(args.root / "benchmarks" / "funcs" / str(record["relative_ir"])),
        "-S", "-o", "/dev/null",
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


def run_one(
    record: dict[str, object], repeat: int, args: argparse.Namespace, environment: dict[str, object],
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
        {"index": int(index), "seconds": float(seconds)}
        for index, seconds in TIME_RE.findall(stdout + "\n" + stderr)
    ]
    end_index = int(record["end_index"])
    endpoint_times = [entry["seconds"] for entry in telemetry if entry["index"] == end_index]
    run_id = f"{record['segment_id']}:{args.mode}:r{repeat}"
    result = {
        "schema_version": 1,
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
        "outcome": {
            "verdict": outcome,
            "censored": outcome == "timeout",
            "wall_time_seconds": round(elapsed, 6),
            "verification_time_seconds": max(endpoint_times) if endpoint_times else None,
            "returncode": returncode,
            "telemetry": telemetry,
        },
        "environment": environment,
        "started_unix": started,
        "command": command,
        "launch_command": launch_command,
    }
    if outcome not in {"correct", "syntactic_skip"} or args.store_success_output:
        result["stdout_tail"] = tail(stdout)
        result["stderr_tail"] = tail(stderr)
    return result


def existing_run_ids(path: Path) -> set[str]:
    return {str(record["run_id"]) for record in read_jsonl(path) if "run_id" in record}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opt-alive", type=Path,
                        default=root / "alive2-decomp" / "build" / "opt-alive-test.sh")
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
    args.opt_alive = args.opt_alive.resolve()

    records = read_jsonl(args.manifest)
    seen_segments = set()
    for record in records:
        segment = str(record["segment_id"])
        if segment in seen_segments:
            raise SystemExit(f"duplicate segment_id in manifest: {segment}")
        seen_segments.add(segment)
    done = existing_run_ids(args.output)
    tasks = []
    for record in records:
        for repeat in range(args.repeats):
            run_id = f"{record['segment_id']}:{args.mode}:r{repeat}"
            if run_id not in done:
                tasks.append((record, repeat))
    if args.limit > 0:
        tasks = tasks[:args.limit]

    print(f"manifest segments: {len(records)}")
    print(f"already completed: {len(done)}")
    print(f"pending {args.mode} runs: {len(tasks)}")
    if not args.execute:
        for record, _ in tasks[:10]:
            print(shlex.join(execution_command(command_for(record, args), args.memory_gb)))
        if len(tasks) > 10:
            print(f"... {len(tasks) - 10} additional commands")
        print("dry run only; pass --execute on the experiment server")
        return

    if not args.opt_alive.exists():
        raise SystemExit(f"validator not found: {args.opt_alive}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        "git_commit": git_commit(args.root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "timeout_seconds": args.timeout,
        "memory_gb": args.memory_gb,
        "jobs": args.jobs,
    }

    completed = 0
    with args.output.open("a") as output, concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, record, repeat, args, environment): (record, repeat)
            for record, repeat in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            record, repeat = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "schema_version": 1,
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
                    "outcome": {"verdict": "runner_error", "censored": False},
                    "runner_error": repr(error),
                    "environment": environment,
                }
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            completed += 1
            verdict = result["outcome"]["verdict"]
            print(f"[{completed}/{len(tasks)}] {result['segment_id']} {verdict}", flush=True)


if __name__ == "__main__":
    main()
