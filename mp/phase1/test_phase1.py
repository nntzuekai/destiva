#!/usr/bin/env python3
"""Focused tests for the offline Phase 1 parser and statistics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase1_analyze import cox_fit, rankdata, spearman, standardize
from phase1_extract import ir_features, pass_features


class IRFeatureTests(unittest.TestCase):
    def test_multiline_switch_and_memory_features(self) -> None:
        ir = """
define i32 @sample(ptr %p, i32 %x) {
entry:
  %v = load i32, ptr %p, align 4
  %a = add i32 %v, 1
  switch i32 %a, label %default [
    i32 0, label %zero
    i32 1, label %one
  ]
zero:
  store i32 0, ptr %p, align 4
  br label %default
one:
  %c = call i32 @callee(i32 %a)
  br label %default
default:
  %r = phi i32 [ %a, %entry ], [ 0, %zero ], [ %c, %one ]
  ret i32 %r
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.ll"
            path.write_text(ir)
            features = ir_features(path, "sample")
        self.assertEqual(features["ir_instruction_count"], 9)
        self.assertEqual(features["inst_switch_count"], 1)
        self.assertEqual(features["inst_load_count"], 1)
        self.assertEqual(features["inst_store_count"], 1)
        self.assertEqual(features["inst_phi_count"], 1)
        self.assertEqual(features["direct_call_count"], 1)
        self.assertEqual(features["basic_block_count"], 4)
        self.assertEqual(features["cfg_edge_count"], 5)

    def test_pass_summary(self) -> None:
        data = {
            "function": "sample",
            "passes": [
                {"name": "Start", "diff": 0},
                {"name": "SROAPass", "diff": 2},
                {"name": "InstCombinePass", "diff": 6},
                {"name": "End", "diff": 0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample_passes.json"
            path.write_text(json.dumps(data))
            features = pass_features(path)
        self.assertEqual(features["pass_count"], 4)
        self.assertEqual(features["modified_pass_count"], 2)
        self.assertEqual(features["pass_diff_total_recomputed"], 8)
        self.assertAlmostEqual(features["pass_diff_top1_share"], 0.75)
        self.assertEqual(features["changed_InstCombinePass_count"], 1)


class StatisticalTests(unittest.TestCase):
    def test_rankdata_ties_and_spearman(self) -> None:
        np.testing.assert_allclose(rankdata(np.array([3.0, 1.0, 1.0])), [3.0, 1.5, 1.5])
        self.assertAlmostEqual(spearman(np.arange(10.0), np.arange(10.0)), 1.0)
        self.assertAlmostEqual(spearman(np.arange(10.0), -np.arange(10.0)), -1.0)

    def test_cox_direction_with_censoring(self) -> None:
        rng = np.random.default_rng(7)
        feature = rng.normal(size=700)
        # Larger feature -> longer completion time -> negative completion hazard.
        completion = rng.exponential(scale=np.exp(feature))
        censor_at = 2.0
        events = completion <= censor_at
        times = np.minimum(completion, censor_at)
        x, _ = standardize(feature)
        beta, covariance, converged = cox_fit(times, events, x[:, None])
        self.assertTrue(converged)
        self.assertLess(beta[0], -0.5)
        self.assertGreater(covariance[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
