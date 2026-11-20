#!/usr/bin/env python3
"""Focused tests for the live Phase 2 collector and runner plumbing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "mp" / "phase2"
sys.path.insert(0, str(PHASE2))
sys.path.insert(0, str(ROOT))

import tester
from collect_live import (
    change_summary,
    flatten_boundary_features,
    pair_sections,
    parse_sections,
)
from phase2_common import (
    collect_candidates,
    heuristic_segments,
    segment_features,
    select_candidates,
)
from run_segments import (
    MEASUREMENT_METHOD,
    classify,
    command_for,
    load_live_manifest,
    run_one,
    validate_existing_results,
)


def fixture_passes() -> list[dict[str, object]]:
    diffs = [0, 2, 0, 10, 1, 0]
    names = ["Start", "Repeat", "Middle", "Repeat", "Tail", "End"]
    occurrences: dict[str, int] = {}
    result = []
    for index, (name, diff) in enumerate(zip(names, diffs)):
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        result.append({
            "index": index,
            "name": name,
            "occurrence": occurrence,
            "diff": diff,
        })
    return result


def ir(addend: int, include_mul: bool = False) -> str:
    lines = [
        "define i32 @sample(i32 %a) {",
        "entry:",
        f"  %x = add i32 %a, {addend}, !mymd !0",
    ]
    if include_mul:
        lines.append("  %y = mul i32 %x, 2, !mymd !2")
        lines.append("  ret i32 %y, !mymd !1")
    else:
        lines.append("  ret i32 %x, !mymd !1")
    lines.extend([
        "}",
        '!0 = !{!"0"}',
        '!1 = !{!"1"}',
        '!2 = !{!"2"}',
    ])
    return "\n".join(lines) + "\n"


def fake_dump() -> str:
    states = [ir(1), ir(2), ir(2, True), ir(3, True)]
    chunks = []
    for index, name in enumerate(("FirstPass", "SecondPass", "ThirdPass")):
        chunks.extend([
            f"; *** IR Dump Before {name} on [module] ***\n{states[index]}",
            f"; *** IR Dump After {name} on [module] ***\n{states[index + 1]}",
        ])
    return "".join(chunks)


def make_fake_opt(path: Path) -> None:
    program = (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    print('fake opt 1.0')\n"
        "else:\n"
        f"    sys.stderr.write({fake_dump()!r})\n"
    )
    path.write_text(program)
    path.chmod(0o755)


def run_collector(root: Path) -> tuple[Path, Path]:
    funcs = root / "benchmarks" / "funcs" / "demo"
    funcs.mkdir(parents=True)
    (funcs / "sample.ll").write_text(ir(1))
    functions = root / "functions.txt"
    functions.write_text("demo/sample.ll\n")
    fake_opt = root / "opt"
    make_fake_opt(fake_opt)
    profiles = root / "work" / "profiles.jsonl"
    manifest = root / "work" / "manifest.jsonl"
    snapshots = root / "work" / "snapshots"
    run = subprocess.run([
        sys.executable, str(PHASE2 / "collect_live.py"),
        "--root", str(root),
        "--functions", str(functions),
        "--opt", str(fake_opt),
        "--sample-functions", "0",
        "--max-segments-per-function", "12",
        "--profiles", str(profiles),
        "--manifest", str(manifest),
        "--snapshots-dir", str(snapshots),
    ], capture_output=True, text=True)
    if run.returncode != 0:
        raise AssertionError(run.stdout + run.stderr)
    return profiles, manifest


class HeuristicTests(unittest.TestCase):
    def test_matches_existing_tester_heuristics(self) -> None:
        passes = fixture_passes()
        pairs = {
            "mean": list(tester.pairs_mean(passes)),
            "median": list(tester.pairs_median(passes)),
            "top2": list(tester.pairs_top2(passes)),
            "fixed": list(tester.pairs_fixed(passes)),
            "dynamic_four": list(tester.pairs_dynamic_four(passes)),
            "dynamic": list(tester.pairs_dynamic(passes)),
        }
        for heuristic, expected in pairs.items():
            self.assertEqual(heuristic_segments(passes, heuristic), expected, heuristic)

    def test_segment_features_exclude_start_pass_change(self) -> None:
        features = segment_features(fixture_passes(), 1, 4)
        self.assertEqual(features["segment_span"], 3)
        self.assertEqual(features["segment_diff_total"], 11)
        self.assertEqual(features["segment_modified_pass_count"], 2)

    def test_sampling_is_deterministic_and_keeps_clustered4(self) -> None:
        passes = fixture_passes()
        candidates = collect_candidates(
            passes, ["end_to_end", "all_passes", "top2", "dynamic_four"],
        )
        first = select_candidates(candidates, 5, 7, "sample")
        self.assertEqual(first, select_candidates(candidates, 5, 7, "sample"))
        self.assertTrue(set(heuristic_segments(passes, "dynamic_four")).issubset(first))


class LiveCollectorTests(unittest.TestCase):
    def test_dump_pairing_and_instruction_characteristics(self) -> None:
        pairs = pair_sections(parse_sections(fake_dump()))
        self.assertEqual([pair[0] for pair in pairs], ["FirstPass", "SecondPass", "ThirdPass"])
        sys.path.insert(0, str(ROOT / "mp" / "phase1"))
        from phase1_extract import find_function, logical_instructions
        changes = change_summary(
            pairs[1][1], pairs[1][2], "sample", find_function, logical_instructions,
        )
        self.assertEqual(changes["added_count"], 1)
        self.assertEqual(changes["modified_count"], 1)
        self.assertEqual(changes["features"]["added_opcode_mul_count"], 1)
        features = flatten_boundary_features(
            {"ir_instruction_count": 2}, {"ir_instruction_count": 3},
        )
        self.assertEqual(features["delta_ir_instruction_count"], 1)

    def test_collector_needs_no_pass_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, manifest = run_collector(root)
            self.assertFalse(any(root.rglob("*_passes.json")))
            profile = json.loads(profiles.read_text())
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            metadata = json.loads(manifest.with_suffix(".metadata.json").read_text())
            self.assertTrue(metadata["complete"])
            self.assertEqual(profile["pass_count"], 3)
            self.assertEqual([row["name"] for row in profile["passes"]],
                             ["FirstPass", "SecondPass", "ThirdPass"])
            self.assertTrue(rows)
            self.assertIn("input_features", rows[0])
            self.assertIn("boundary_features", rows[0])
            self.assertIn("net_diff", rows[0]["features"])
            for row in rows:
                for side in ("start", "end"):
                    snapshot = root / row[f"{side}_snapshot"]
                    self.assertTrue(snapshot.is_file())
                    self.assertEqual(
                        hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                        row[f"{side}_snapshot_sha256"],
                    )


class RunnerTests(unittest.TestCase):
    def test_command_uses_live_snapshots(self) -> None:
        record = {
            "function": "sample",
            "_start_snapshot_path": Path("/snapshots/start.ll"),
            "_end_snapshot_path": Path("/snapshots/end.ll"),
        }
        args = argparse.Namespace(alive_tv=Path("/tool/alive-tv"), mode="temporal")
        command = command_for(record, args)
        self.assertEqual(command[-2:], ["/snapshots/start.ll", "/snapshots/end.ll"])
        self.assertIn("--time-verify", command)
        self.assertNotIn("-O3", command)
        args.mode = "combined"
        self.assertIn("--decompose", command_for(record, args))

    def test_live_manifest_loader_checks_hashes_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = run_collector(root)
            rows, digest = load_live_manifest(manifest, root)
            self.assertTrue(rows)
            self.assertEqual(len(digest), 64)
            self.assertTrue(rows[0]["_start_snapshot_path"].is_file())

    def test_legacy_results_cannot_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            results.write_text(json.dumps({"schema_version": 2, "run_id": "old"}) + "\n")
            with self.assertRaises(ValueError):
                validate_existing_results(results, "digest")

    def test_run_one_parses_exact_function_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            validator = temp / "alive-tv"
            validator.write_text(
                "#!/bin/sh\n"
                "echo 'Transformation seems to be correct!'\n"
                "echo 'PHASE2_TIME: function=sample seconds=1.25'\n"
            )
            validator.chmod(0o755)
            start, end = temp / "start.ll", temp / "end.ll"
            start.write_text("start\n")
            end.write_text("end\n")
            record = {
                "segment_id": "seg", "benchmark": "demo", "function": "sample",
                "relative_ir": "demo/sample.ll", "start_index": 1, "end_index": 2,
                "selected_by": ["fixed"], "start_snapshot": str(start),
                "end_snapshot": str(end), "start_snapshot_sha256": "a",
                "end_snapshot_sha256": "b", "_start_snapshot_path": start,
                "_end_snapshot_path": end,
            }
            args = argparse.Namespace(
                alive_tv=validator, mode="temporal", memory_gb=0, timeout=5,
                store_success_output=False,
            )
            result = run_one(record, 0, args, {}, "manifest-digest")
            self.assertEqual(result["outcome"]["verdict"], "correct")
            self.assertTrue(result["outcome"]["measurement_valid"])
            self.assertEqual(result["outcome"]["verification_time_seconds"], 1.25)

    def test_verdict_classification_and_instrumentation(self) -> None:
        self.assertEqual(classify(0, "", "", False), "correct")
        self.assertEqual(classify(-9, "", "", True), "timeout")
        self.assertEqual(classify(1, "Transformation doesn't verify! (unsound)", "", False), "counterexample")
        self.assertEqual(classify(1, "ERROR: program doesn't type check!", "", False), "translation_error")
        source = (ROOT / "alive2-decomp" / "tools" / "alive-tv.cpp").read_text()
        self.assertIn("PHASE2_TIME: function=", source)
        self.assertEqual(MEASUREMENT_METHOD, "live_snapshot_pair_v2")


class PipelineGateTests(unittest.TestCase):
    def test_schema_v3_qc_and_merge_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest_path = run_collector(root)
            manifest_rows, manifest_digest = load_live_manifest(manifest_path, root)
            results_path = root / "results.jsonl"
            with results_path.open("w") as stream:
                for segment in manifest_rows:
                    stream.write(json.dumps({
                        "schema_version": 3,
                        "measurement_method": MEASUREMENT_METHOD,
                        "manifest_sha256": manifest_digest,
                        "run_id": f"{segment['segment_id']}:temporal:r0",
                        "segment_id": segment["segment_id"],
                        "mode": "temporal",
                        "repeat": 0,
                        "selected_by": segment["selected_by"],
                        "outcome": {
                            "verdict": "correct", "censored": False,
                            "measurement_valid": True, "wall_time_seconds": 1.2,
                            "verification_time_seconds": 1.0, "returncode": 0,
                        },
                    }, sort_keys=True) + "\n")
            qc_output = root / "qc.md"
            qc = subprocess.run([
                sys.executable, str(PHASE2 / "qc_report.py"),
                "--root", str(root), "--manifest", str(manifest_path),
                "--results", str(results_path), "--output", str(qc_output),
            ], capture_output=True, text=True)
            self.assertEqual(qc.returncode, 0, qc.stdout + qc.stderr)
            self.assertIn("QC status: PASS", qc_output.read_text())

            dataset = root / "dataset.csv"
            merge = subprocess.run([
                sys.executable, str(PHASE2 / "merge_phase2.py"),
                "--root", str(root), "--manifest", str(manifest_path),
                "--results", str(results_path), "--output", str(dataset),
            ], capture_output=True, text=True)
            self.assertEqual(merge.returncode, 0, merge.stdout + merge.stderr)
            self.assertEqual(len(dataset.read_text().splitlines()), len(manifest_rows) + 1)


if __name__ == "__main__":
    unittest.main()
