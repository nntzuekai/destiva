#!/usr/bin/env python3
"""Focused tests for Phase 2 sampling and runner plumbing; no validation runs."""

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
from extract_boundary_features import (
    first_sequence_difference,
    flatten_boundary_features,
    pair_sections,
    parse_sections,
)
from make_manifest import build_manifest
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
    load_exact_boundaries,
    run_one,
    validate_existing_results,
)


def fixture_passes() -> list[dict[str, object]]:
    diffs = [0, 2, 0, 10, 1, 0]
    names = ["Start", "Repeat", "Middle", "Repeat", "Tail", "End"]
    occurrences = {}
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
        self.assertEqual(features["segment_start_occurrence"], 0)
        self.assertEqual(features["segment_end_pass"], "Tail")

    def test_sampling_is_deterministic_and_keeps_clustered4(self) -> None:
        passes = fixture_passes()
        candidates = collect_candidates(
            passes, ["end_to_end", "all_passes", "top2", "dynamic_four"],
        )
        first = select_candidates(candidates, 5, 7, "sample")
        second = select_candidates(candidates, 5, 7, "sample")
        self.assertEqual(first, second)
        clustered = set(heuristic_segments(passes, "dynamic_four"))
        self.assertTrue(clustered.issubset(first))


class ManifestTests(unittest.TestCase):
    def test_small_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            funcs = root / "benchmarks" / "funcs" / "demo"
            funcs.mkdir(parents=True)
            (funcs / "sample.ll").write_text("define i32 @sample() {\n  ret i32 0\n}\n")
            (funcs / "sample_passes.json").write_text(json.dumps({
                "function": "sample",
                "passes": [{"name": row["name"], "diff": row["diff"]} for row in fixture_passes()],
            }))
            functions = root / "functions.txt"
            functions.write_text("demo/sample.ll\n")
            args = argparse.Namespace(
                root=root,
                functions=functions,
                sample_functions=0,
                seed=11,
                heuristics=["end_to_end", "top2", "dynamic_four"],
                max_segments_per_function=8,
            )
            records, metadata = build_manifest(args)
        self.assertGreater(len(records), 0)
        self.assertEqual(metadata["completed_function_count"], 1)
        self.assertEqual(len({row["segment_id"] for row in records}), len(records))
        self.assertTrue(all(row["start_index"] < row["end_index"] for row in records))


class BoundaryFeatureTests(unittest.TestCase):
    def test_dump_pairing_and_feature_deltas(self) -> None:
        dump = """
; *** IR Dump Before FirstPass on [module] ***
define i32 @sample() { ret i32 0 }
; *** IR Dump After FirstPass on [module] ***
define i32 @sample() { ret i32 1 }
; *** IR Dump Before SecondPass on [module] ***
define i32 @sample() { ret i32 1 }
; *** IR Dump After SecondPass on [module] ***
define i32 @sample() { ret i32 2 }
"""
        pairs = pair_sections(parse_sections(dump))
        self.assertEqual([pair[0] for pair in pairs], ["FirstPass", "SecondPass"])
        features = flatten_boundary_features(
            {"ir_instruction_count": 3, "inst_load_count": 1},
            {"ir_instruction_count": 5, "inst_load_count": 0},
        )
        self.assertEqual(features["delta_ir_instruction_count"], 2)
        self.assertEqual(features["abs_delta_inst_load_count"], 1)

    def test_strict_sequence_difference(self) -> None:
        self.assertIsNone(first_sequence_difference(["A", "B"], ["A", "B"]))
        self.assertEqual(
            first_sequence_difference(["A", "B"], ["A", "C"]),
            {"index": 1, "expected": "B", "observed": "C"},
        )
        self.assertEqual(
            first_sequence_difference(["A"], ["A", "B"]),
            {"index": 1, "expected": "<end>", "observed": "B"},
        )

    def test_extractor_materializes_an_aligned_snapshot_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            funcs = root / "benchmarks" / "funcs" / "demo"
            funcs.mkdir(parents=True)
            (funcs / "sample.ll").write_text(
                "define i32 @sample() {\n  ret i32 0\n}\n"
            )
            (funcs / "sample_passes.json").write_text(json.dumps({
                "function": "sample",
                "passes": [{"name": "FirstPass", "diff": 1},
                           {"name": "SecondPass", "diff": 1}],
            }))
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "segment_id": "demo/sample:0:1:test",
                "relative_ir": "demo/sample.ll",
                "relative_passes": "demo/sample_passes.json",
                "function": "sample",
                "start_index": 0,
                "end_index": 1,
                "features": {
                    "segment_start_pass": "FirstPass",
                    "segment_end_pass": "SecondPass",
                },
            }) + "\n")
            fake_opt = root / "opt"
            fake_opt.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"--version\" ]; then echo fake-opt; exit 0; fi\n"
                "cat >&2 <<'EOF'\n"
                "; *** IR Dump Before FirstPass on [module] ***\n"
                "define i32 @sample() {\n  ret i32 0\n}\n"
                "; *** IR Dump After FirstPass on [module] ***\n"
                "define i32 @sample() {\n  ret i32 1\n}\n"
                "; *** IR Dump Before SecondPass on [module] ***\n"
                "define i32 @sample() {\n  ret i32 1\n}\n"
                "; *** IR Dump After SecondPass on [module] ***\n"
                "define i32 @sample() {\n  ret i32 2\n}\n"
                "EOF\n"
            )
            fake_opt.chmod(0o755)
            output = root / "boundaries.jsonl"
            run = subprocess.run([
                sys.executable, str(PHASE2 / "extract_boundary_features.py"),
                "--root", str(root), "--manifest", str(manifest),
                "--output", str(output), "--opt", str(fake_opt),
            ], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            row = json.loads(output.read_text())
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["alignment_error_count"], 0)
            self.assertTrue((root / row["start_snapshot"]).is_file())
            self.assertTrue((root / row["end_snapshot"]).is_file())


class RunnerTests(unittest.TestCase):
    def test_command_uses_exact_snapshots(self) -> None:
        record = {
            "start_index": 12,
            "end_index": 37,
            "relative_ir": "demo/sample.ll",
            "function": "sample",
            "_boundary": {
                "_start_snapshot_path": Path("/snapshots/start.ll"),
                "_end_snapshot_path": Path("/snapshots/end.ll"),
            },
        }
        args = argparse.Namespace(
            alive_tv=Path("/tool/alive-tv"),
            root=Path("/repo"),
            mode="temporal",
        )
        command = command_for(record, args)
        self.assertEqual(command[-2:], ["/snapshots/start.ll", "/snapshots/end.ll"])
        self.assertIn("--time-verify", command)
        self.assertIn("--func=sample", command)
        self.assertNotIn("-O3", command)
        self.assertNotIn("--decompose", command)
        args.mode = "combined"
        self.assertIn("--decompose", command_for(record, args))

    def test_boundary_loader_checks_hashes_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = root / "start.ll"
            end = root / "end.ll"
            start.write_text("start\n")
            end.write_text("end\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            boundary_path = root / "boundaries.jsonl"
            boundary_path.write_text(json.dumps({
                "schema_version": 2,
                "segment_id": "seg",
                "snapshot_method": "post_pass_ir_v1",
                "pass_names_match": True,
                "start_snapshot": "start.ll",
                "end_snapshot": "end.ll",
                "start_snapshot_sha256": digest(start),
                "end_snapshot_sha256": digest(end),
            }) + "\n")
            boundary_path.with_suffix(".metadata.json").write_text(json.dumps({
                "schema_version": 2,
                "complete": True,
                "alignment_error_count": 0,
                "errors": [],
            }))
            boundaries, boundary_digest = load_exact_boundaries(
                [{"segment_id": "seg"}], boundary_path, root,
            )
            self.assertEqual(set(boundaries), {"seg"})
            self.assertEqual(boundaries["seg"]["_start_snapshot_path"], start)
            self.assertEqual(len(boundary_digest), 64)

    def test_legacy_results_cannot_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            results.write_text(json.dumps({"schema_version": 1, "run_id": "old"}) + "\n")
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
            start = temp / "start.ll"
            end = temp / "end.ll"
            start.write_text("start\n")
            end.write_text("end\n")
            record = {
                "segment_id": "seg",
                "benchmark": "demo",
                "function": "sample",
                "relative_ir": "demo/sample.ll",
                "start_index": 1,
                "end_index": 2,
                "selected_by": ["fixed"],
                "_boundary": {
                    "start_snapshot": str(start),
                    "end_snapshot": str(end),
                    "start_snapshot_sha256": "a",
                    "end_snapshot_sha256": "b",
                    "_start_snapshot_path": start,
                    "_end_snapshot_path": end,
                },
            }
            args = argparse.Namespace(
                alive_tv=validator,
                mode="temporal",
                memory_gb=0,
                timeout=5,
                store_success_output=False,
            )
            result = run_one(record, 0, args, {}, "boundary-digest")
            self.assertEqual(result["outcome"]["verdict"], "correct")
            self.assertTrue(result["outcome"]["measurement_valid"])
            self.assertEqual(result["outcome"]["verification_time_seconds"], 1.25)

    def test_verdict_classification(self) -> None:
        self.assertEqual(classify(0, "", "", False), "correct")
        self.assertEqual(classify(-9, "", "", True), "timeout")
        self.assertEqual(classify(1, "Transformation doesn't verify! (unsound)", "", False), "counterexample")
        self.assertEqual(classify(1, "Transformation doesn't verify! (not unsound)", "", False), "verification_failure")
        self.assertEqual(classify(1, "ERROR: program doesn't type check!", "", False), "translation_error")
        self.assertEqual(classify(1, "", "assertion failed", False), "crash")

    def test_standalone_validator_has_phase2_timing(self) -> None:
        source = (ROOT / "alive2-decomp" / "tools" / "alive-tv.cpp").read_text()
        plugin_source = (ROOT / "alive2-decomp" / "tv" / "tv.cpp").read_text()
        self.assertIn("PHASE2_TIME: function=", source)
        self.assertNotIn("tv-check-index-pairs", plugin_source)
        self.assertEqual(MEASUREMENT_METHOD, "exact_snapshot_pair_v1")


class PipelineGateTests(unittest.TestCase):
    def test_schema_v2_qc_and_merge_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.jsonl"
            boundary_path = root / "boundaries.jsonl"
            results_path = root / "results.jsonl"
            manifest = {
                "schema_version": 1,
                "segment_id": "seg",
                "benchmark": "demo",
                "function": "sample",
                "relative_ir": "demo/sample.ll",
                "start_index": 0,
                "end_index": 1,
                "pass_count": 2,
                "selected_by": ["fixed"],
                "features": {"segment_span": 1},
            }
            boundary = {
                "schema_version": 2,
                "segment_id": "seg",
                "snapshot_method": "post_pass_ir_v1",
                "pass_names_match": True,
                "boundary_features": {"delta_ir_instruction_count": 1},
            }
            manifest_path.write_text(json.dumps(manifest) + "\n")
            boundary_path.write_text(json.dumps(boundary) + "\n")
            boundary_path.with_suffix(".metadata.json").write_text(json.dumps({
                "schema_version": 2,
                "complete": True,
                "errors": [],
            }))
            boundary_digest = hashlib.sha256(boundary_path.read_bytes()).hexdigest()
            results_path.write_text(json.dumps({
                "schema_version": 2,
                "measurement_method": MEASUREMENT_METHOD,
                "boundary_features_sha256": boundary_digest,
                "run_id": "seg:temporal:r0",
                "segment_id": "seg",
                "mode": "temporal",
                "repeat": 0,
                "selected_by": ["fixed"],
                "outcome": {
                    "verdict": "correct",
                    "censored": False,
                    "measurement_valid": True,
                    "wall_time_seconds": 1.2,
                    "verification_time_seconds": 1.0,
                    "returncode": 0,
                },
            }) + "\n")
            qc_output = root / "qc.md"
            qc = subprocess.run([
                sys.executable, str(PHASE2 / "qc_report.py"),
                "--manifest", str(manifest_path),
                "--results", str(results_path),
                "--boundary-features", str(boundary_path),
                "--output", str(qc_output),
            ], capture_output=True, text=True)
            self.assertEqual(qc.returncode, 0, qc.stdout + qc.stderr)
            self.assertIn("QC status: PASS", qc_output.read_text())

            dataset = root / "dataset.csv"
            merge = subprocess.run([
                sys.executable, str(PHASE2 / "merge_phase2.py"),
                "--root", str(root),
                "--manifest", str(manifest_path),
                "--results", str(results_path),
                "--boundary-features", str(boundary_path),
                "--output", str(dataset),
                "--no-ir-features",
            ], capture_output=True, text=True)
            self.assertEqual(merge.returncode, 0, merge.stdout + merge.stderr)
            self.assertEqual(len(dataset.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
