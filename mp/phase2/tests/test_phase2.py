#!/usr/bin/env python3
"""Focused tests for Phase 2 sampling and runner plumbing; no validation runs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "mp" / "phase2"
sys.path.insert(0, str(PHASE2))
sys.path.insert(0, str(ROOT))

import tester
from extract_boundary_features import flatten_boundary_features, pair_sections, parse_sections
from make_manifest import build_manifest
from phase2_common import (
    collect_candidates,
    heuristic_segments,
    segment_features,
    select_candidates,
)
from run_segments import classify, command_for


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


class RunnerTests(unittest.TestCase):
    def test_command_uses_exact_indices(self) -> None:
        record = {
            "start_index": 12,
            "end_index": 37,
            "relative_ir": "demo/sample.ll",
        }
        args = argparse.Namespace(
            opt_alive=Path("/tool/opt-alive-test.sh"),
            root=Path("/repo"),
            mode="temporal",
        )
        command = command_for(record, args)
        self.assertIn("-tv-check-index-pairs=12:37", command)
        self.assertIn("-tv-time-verify", command)
        self.assertNotIn("--decompose", command)
        args.mode = "combined"
        self.assertIn("--decompose", command_for(record, args))

    def test_verdict_classification(self) -> None:
        self.assertEqual(classify(0, "", "", False), "correct")
        self.assertEqual(classify(-9, "", "", True), "timeout")
        self.assertEqual(classify(1, "Transformation doesn't verify! (unsound)", "", False), "counterexample")
        self.assertEqual(classify(1, "Transformation doesn't verify! (not unsound)", "", False), "verification_failure")
        self.assertEqual(classify(1, "ERROR: program doesn't type check!", "", False), "translation_error")
        self.assertEqual(classify(1, "", "assertion failed", False), "crash")

    def test_validator_source_has_indexed_option(self) -> None:
        source = (ROOT / "alive2-decomp" / "tv" / "tv.cpp").read_text()
        self.assertIn("tv-check-index-pairs", source)
        self.assertIn("PHASE2_TIME:", source)


if __name__ == "__main__":
    unittest.main()
