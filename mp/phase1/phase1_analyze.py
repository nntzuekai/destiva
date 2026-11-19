#!/usr/bin/env python3
"""Analyze the offline DESTIVA Phase 1 function-level feature dataset.

Outputs:
  * report.md                 human-readable findings and limitations
  * correlations.csv          Spearman and partial-Spearman associations
  * cox_associations.csv      timeout-aware Cox proportional-hazards models
  * plots/*.png               summary visualizations

Timeouts are right-censored at ``--timeout`` in the Cox models.  Capped-time
Spearman results are also emitted to match the paper's descriptive convention,
but are clearly labeled and should not be interpreted as uncensored runtimes.
The implementation depends only on NumPy and Matplotlib (already installed by
the artifact's documented analysis environment).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "alive_tv": "End-to-end",
    "all_passes": "Pass-by-pass",
    "pairs_fixed": "Fixed",
    "pairs_mean": "Cutout",
    "pairs_top2": "Top2",
    "pairs_dynamic": "Clustered",
    "pairs_dynamic_four": "Clustered4",
}

BASELINES = {"alive_tv": "E-E", "all_passes": "P-P"}
DEFAULT_COX_METHODS = ("alive_tv", "all_passes", "pairs_dynamic_four")

FEATURE_GROUPS = {
    "size_cfg": [
        "num_instructions", "ir_instruction_count", "function_arguments",
        "basic_block_count", "cfg_edge_count", "cyclomatic_complexity",
        "instructions_per_block", "unique_opcode_count", "opcode_entropy",
    ],
    "instruction_mix": [
        "category_int_arith_count", "category_int_arith_ratio",
        "category_float_arith_count", "category_float_arith_ratio",
        "category_bitwise_count", "category_bitwise_ratio",
        "category_divrem_count", "category_divrem_ratio",
        "category_compare_count", "category_compare_ratio",
        "category_conversion_count", "category_conversion_ratio",
        "category_memory_count", "category_memory_ratio",
        "category_control_count", "category_control_ratio",
        "category_vector_count", "category_vector_ratio",
        "category_aggregate_count", "category_aggregate_ratio",
        "inst_phi_count", "inst_phi_ratio", "inst_select_count", "inst_select_ratio",
        "inst_call_count", "inst_call_ratio", "direct_call_count", "indirect_call_count",
    ],
    "types_operands": [
        "integer_type_mentions", "integer_type_mentions_per_instruction",
        "integer_width_mean", "integer_width_max", "wide_integer_mentions",
        "wide_integer_ratio", "float_type_mentions", "float_type_mentions_per_instruction",
        "pointer_type_mentions", "pointer_mentions_per_instruction",
        "vector_type_mentions", "vector_type_mentions_per_instruction", "vector_lanes_max",
        "ssa_value_reference_count", "ssa_refs_per_instruction",
        "global_reference_count", "global_refs_per_instruction",
        "numeric_literal_mentions", "numeric_literals_per_instruction",
        "zero_literal_mentions", "one_literal_mentions", "special_value_mentions",
        "special_values_per_instruction", "max_ssa_use_count",
        "dependency_edge_count", "dependency_edges_per_instruction",
        "dependency_depth_max", "dependency_depth_mean",
    ],
    "memory": [
        "inst_alloca_count", "inst_alloca_ratio", "inst_load_count", "inst_load_ratio",
        "inst_store_count", "inst_store_ratio", "inst_getelementptr_count",
        "inst_getelementptr_ratio", "atomic_instruction_count", "volatile_instruction_count",
        "memory_intrinsic_call_count",
    ],
    "pass_changes": [
        "total_diff", "pass_count", "modified_pass_count", "unmodified_pass_count",
        "pass_diff_mean_nonzero", "pass_diff_median_nonzero", "pass_diff_max",
        "pass_diff_stddev", "pass_diff_cv", "pass_diff_top1_share",
        "pass_diff_top2_share", "pass_diff_top4_share", "pass_diff_entropy",
        "pass_diff_effective_passes", "passes_above_mean_diff", "passes_above_median_diff",
        "modified_pass_position_mean", "modified_pass_weighted_position",
        "modified_pass_gap_mean", "modified_pass_gap_max", "first_modified_pass_position",
        "last_modified_pass_position",
    ],
}

CONTROL_FEATURES = ("num_instructions", "total_diff", "modified_pass_count", "pass_count")


def read_dataset(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows, fields


def numeric_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        text = row.get(name, "")
        try:
            values.append(float(text))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def string_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([row.get(name, "") for row in rows], dtype=object)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def normal_pvalue(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def correlation_inference(rho: float, n: int) -> tuple[float, float, float]:
    """Return approximate p-value and Fisher-z 95% interval."""
    if not math.isfinite(rho) or n <= 3:
        return float("nan"), float("nan"), float("nan")
    clipped = max(-0.999999, min(0.999999, rho))
    z_score = clipped * math.sqrt(max(1.0, n - 1.0))
    pvalue = normal_pvalue(z_score)
    center = math.atanh(clipped)
    radius = 1.96 / math.sqrt(n - 3)
    return pvalue, math.tanh(center - radius), math.tanh(center + radius)


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), controls])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def benchmark_dummies(benchmarks: np.ndarray) -> tuple[np.ndarray, list[str]]:
    levels = sorted(set(str(value) for value in benchmarks))
    # Drop one level to avoid exact collinearity with the intercept.
    columns = [(benchmarks == level).astype(float) for level in levels[1:]]
    matrix = np.column_stack(columns) if columns else np.empty((len(benchmarks), 0))
    return matrix, [f"benchmark={level}" for level in levels[1:]]


def controls_for(
    rows: list[dict[str, str]], mask: np.ndarray, feature: str, method: str | None,
) -> np.ndarray:
    x = numeric_column(rows, feature)[mask]
    columns: list[np.ndarray] = []
    for control in CONTROL_FEATURES:
        if control == feature:
            continue
        values = numeric_column(rows, control)[mask]
        if not np.all(np.isfinite(values)) or np.std(values) == 0:
            continue
        ranked = rankdata(values)
        # Avoid unstable duplicate controls such as num_instructions and
        # ir_instruction_count.
        if abs(pearson(rankdata(x), ranked)) > 0.995:
            continue
        columns.append(ranked)
    if method and f"{method}_num_pairs" in rows[0]:
        values = numeric_column(rows, f"{method}_num_pairs")[mask]
        if np.all(np.isfinite(values)) and np.std(values) > 0:
            columns.append(rankdata(values))
    dummies, _ = benchmark_dummies(string_column(rows, "benchmark")[mask])
    columns.extend(dummies[:, i] for i in range(dummies.shape[1]))
    return np.column_stack(columns) if columns else np.empty((np.sum(mask), 0))


def adjusted_spearman(
    rows: list[dict[str, str]], feature: str, x: np.ndarray, y: np.ndarray,
    mask: np.ndarray, method: str | None,
) -> float:
    ranked_x = rankdata(x[mask])
    ranked_y = rankdata(y[mask])
    controls = controls_for(rows, mask, feature, method)
    if controls.shape[1] == 0:
        return pearson(ranked_x, ranked_y)
    return pearson(residualize(ranked_x, controls), residualize(ranked_y, controls))


def apply_bh(rows: list[dict[str, object]], group_fields: tuple[str, ...]) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        p = row.get("p_value")
        if isinstance(p, float) and math.isfinite(p):
            groups[tuple(row[field] for field in group_fields)].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: float(row["p_value"]))
        running = 1.0
        for reverse_index in range(len(ordered) - 1, -1, -1):
            rank = reverse_index + 1
            value = min(1.0, float(ordered[reverse_index]["p_value"]) * len(ordered) / rank)
            running = min(running, value)
            ordered[reverse_index]["q_value_bh"] = running


def feature_names(fields: list[str], rows: list[dict[str, str]]) -> list[str]:
    requested = [name for group in FEATURE_GROUPS.values() for name in group]
    selected = []
    for name in requested:
        if name not in fields:
            continue
        values = numeric_column(rows, name)
        finite = values[np.isfinite(values)]
        if len(finite) >= 20 and np.std(finite) > 0:
            selected.append(name)
    return selected


def capped_method_outcome(
    rows: list[dict[str, str]], method: str, timeout: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = string_column(rows, f"{method}_result")
    time = numeric_column(rows, f"{method}_time")
    valid = np.isfinite(time) & np.isin(result, ["correct", "timeout"])
    capped = np.where(result == "timeout", np.minimum(time, timeout), time)
    event = result == "correct"
    return capped, event, valid


def correlation_rows(
    rows: list[dict[str, str]], features: list[str], timeout: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    def add_target(
        target_name: str, kind: str, method: str, y: np.ndarray, valid: np.ndarray,
    ) -> None:
        for feature in features:
            x = numeric_column(rows, feature)
            mask = valid & np.isfinite(x) & np.isfinite(y)
            n = int(np.sum(mask))
            if n < 20 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
                continue
            rho = spearman(x[mask], y[mask])
            partial = adjusted_spearman(rows, feature, x, y, mask, method)
            pvalue, low, high = correlation_inference(partial, n)
            output.append({
                "target": target_name,
                "target_kind": kind,
                "method": method,
                "feature": feature,
                "n": n,
                "spearman_rho": rho,
                "partial_spearman_rho": partial,
                "partial_ci95_low": low,
                "partial_ci95_high": high,
                "p_value": pvalue,
                "q_value_bh": float("nan"),
            })

    outcomes = {}
    for method in METHODS:
        capped, event, valid = capped_method_outcome(rows, method, timeout)
        outcomes[method] = (capped, event, valid)
        add_target(f"{method}_capped_runtime", "capped_runtime", method, capped, valid)
        add_target(f"{method}_solved_runtime", "solved_runtime", method, capped, valid & event)

    for method in METHODS:
        if method in BASELINES:
            continue
        method_time, _, method_valid = outcomes[method]
        for baseline in BASELINES:
            base_time, _, base_valid = outcomes[baseline]
            valid = method_valid & base_valid & (method_time > 0) & (base_time > 0)
            speedup = np.divide(base_time, method_time, out=np.full_like(base_time, np.nan), where=valid)
            add_target(
                f"{method}_speedup_vs_{baseline}", "capped_speedup", method, speedup, valid,
            )

    apply_bh(output, ("target",))
    return output


def should_log_feature(name: str, values: np.ndarray) -> bool:
    if np.nanmin(values) < 0 or np.nanmax(values) <= 10:
        return False
    if any(token in name for token in ("ratio", "share", "position", "entropy", "_cv", "per_instruction")):
        return False
    return any(token in name for token in (
        "count", "mentions", "total", "diff", "edges", "depth", "passes",
        "instructions", "arguments", "complexity", "gap", "max",
    ))


def standardize(values: np.ndarray) -> tuple[np.ndarray, float]:
    std = float(np.std(values))
    return ((values - np.mean(values)) / std, std) if std else (values * 0, 0.0)


def independent_columns(columns: list[np.ndarray], tolerance: float = 1e-5) -> list[np.ndarray]:
    """Drop controls that add effectively no information to earlier controls."""
    kept: list[np.ndarray] = []
    for column in columns:
        if not kept:
            kept.append(column)
            continue
        matrix = np.column_stack(kept)
        residual = residualize(column, matrix)
        if np.std(residual) > tolerance:
            kept.append(column)
    return kept


def cox_fit(
    times: np.ndarray, events: np.ndarray, design: np.ndarray,
    max_iter: int = 30, tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit a Cox model with Breslow tie handling using Newton iterations."""
    order = np.argsort(-times, kind="mergesort")
    times = times[order]
    events = events[order].astype(bool)
    x = design[order]
    p = x.shape[1]
    beta = np.zeros(p, dtype=float)
    converged = False
    information = np.eye(p)

    # For an event at a tied time, the Breslow risk set ends at the last row
    # with that time in descending order.  Precompute those indices once; all
    # risk-set moments can then be obtained with vectorized cumulative sums.
    tie_end = np.empty(len(times), dtype=int)
    i = 0
    while i < len(times):
        j = i + 1
        while j < len(times) and times[j] == times[i]:
            j += 1
        tie_end[i:j] = j - 1
        i = j
    event_indices = np.flatnonzero(events)
    event_risk_ends = tie_end[event_indices]

    for _ in range(max_iter):
        eta = np.clip(x @ beta, -40.0, 40.0)
        weights = np.exp(eta)
        cumulative_w = np.cumsum(weights)
        cumulative_x = np.cumsum(weights[:, None] * x, axis=0)
        outer = x[:, :, None] * x[:, None, :]
        cumulative_xx = np.cumsum(weights[:, None, None] * outer, axis=0)
        risk_w = cumulative_w[event_risk_ends]
        risk_mean = cumulative_x[event_risk_ends] / risk_w[:, None]
        score = np.sum(x[event_indices] - risk_mean, axis=0)
        risk_second = cumulative_xx[event_risk_ends] / risk_w[:, None, None]
        risk_covariance = risk_second - risk_mean[:, :, None] * risk_mean[:, None, :]
        information = np.sum(risk_covariance, axis=0)

        regularized = information + np.eye(p) * 1e-9
        try:
            step = np.linalg.solve(regularized, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(regularized) @ score
        # Damping protects against separation in sparse benchmark controls.
        magnitude = float(np.max(np.abs(step))) if len(step) else 0.0
        if magnitude > 2.0:
            step *= 2.0 / magnitude
        beta += step
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break

    try:
        covariance = np.linalg.inv(information + np.eye(p) * 1e-9)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(information + np.eye(p) * 1e-9)
    return beta, covariance, converged


def cox_rows(
    rows: list[dict[str, str]], features: list[str], timeout: float,
    methods: tuple[str, ...] = DEFAULT_COX_METHODS,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    benchmarks_all = string_column(rows, "benchmark")
    for method in methods:
        times, events, method_valid = capped_method_outcome(rows, method, timeout)
        for feature in features:
            raw = numeric_column(rows, feature)
            mask = method_valid & np.isfinite(raw) & np.isfinite(times) & (times > 0)
            if np.sum(mask) < 40 or np.sum(events[mask]) < 20:
                continue
            x = raw[mask].copy()
            # Extremely sparse characteristics lead to separation and an
            # uninterpretable asymptotic Cox estimate.  They remain present in
            # correlations.csv, but need at least 30 nonzero observations for
            # the timeout-aware model.
            if np.sum(x != 0) < 30:
                continue
            transform = "raw"
            if should_log_feature(feature, x):
                x = np.log1p(x)
                transform = "log1p"
            x, x_std = standardize(x)
            if x_std == 0:
                continue

            controls: list[np.ndarray] = []
            for name in CONTROL_FEATURES:
                if name == feature:
                    continue
                values = numeric_column(rows, name)[mask]
                if not np.all(np.isfinite(values)) or np.std(values) == 0:
                    continue
                values = np.log1p(np.maximum(values, 0))
                values, std = standardize(values)
                if std and abs(pearson(x, values)) <= 0.995:
                    controls.append(values)
            pair_name = f"{method}_num_pairs"
            if pair_name in rows[0]:
                values = numeric_column(rows, pair_name)[mask]
                if np.all(np.isfinite(values)) and np.std(values) > 0:
                    values, _ = standardize(values)
                    controls.append(values)
            dummies, _ = benchmark_dummies(benchmarks_all[mask])
            controls.extend(dummies[:, i] for i in range(dummies.shape[1]))
            controls = independent_columns(controls)
            control_matrix = np.column_stack(controls) if controls else np.empty((len(x), 0))
            # Report the association for one SD of feature variation that is
            # not already explained by the adjustment variables.  This
            # orthogonalization prevents derived quantities such as
            # mean_diff=total_diff/modified_passes from producing enormous,
            # numerically unstable conditional hazard ratios.
            adjusted_x = residualize(x, control_matrix) if controls else x
            adjusted_x, adjusted_std = standardize(adjusted_x)
            if adjusted_std < 1e-5:
                continue
            design = np.column_stack([adjusted_x] + controls)
            beta, covariance, converged = cox_fit(times[mask], events[mask], design)
            coefficient = float(beta[0])
            se = math.sqrt(max(0.0, float(covariance[0, 0])))
            z = coefficient / se if se else float("nan")
            pvalue = normal_pvalue(z) if math.isfinite(z) else float("nan")
            output.append({
                "method": method,
                "feature": feature,
                "n": int(np.sum(mask)),
                "completed": int(np.sum(events[mask])),
                "censored": int(np.sum(~events[mask])),
                "feature_transform": transform,
                "hazard_ratio_per_adjusted_sd": math.exp(max(-40.0, min(40.0, coefficient))),
                "hr_ci95_low": math.exp(max(-40.0, min(40.0, coefficient - 1.96 * se))),
                "hr_ci95_high": math.exp(max(-40.0, min(40.0, coefficient + 1.96 * se))),
                "coefficient": coefficient,
                "standard_error": se,
                "p_value": pvalue,
                "q_value_bh": float("nan"),
                "converged": converged,
            })
    apply_bh(output, ("method",))
    return output


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no output rows for {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def top_rows(
    rows: list[dict[str, object]], target: str, value: str, limit: int = 10,
) -> list[dict[str, object]]:
    selected = [
        row for row in rows
        if row.get("target") == target and math.isfinite(float(row.get(value, float("nan"))))
    ]
    return sorted(selected, key=lambda row: abs(float(row[value])), reverse=True)[:limit]


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def make_heatmap(
    correlations: list[dict[str, object]], output: Path,
) -> list[str]:
    target_by_method = {method: f"{method}_capped_runtime" for method in METHODS}
    candidates = [
        row for row in correlations
        if row["target"] in target_by_method.values()
    ]
    scores: dict[str, float] = defaultdict(float)
    for row in candidates:
        scores[str(row["feature"])] = max(
            scores[str(row["feature"])], abs(float(row["partial_spearman_rho"])),
        )
    features = [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:20]]
    lookup = {(str(row["method"]), str(row["feature"])): float(row["partial_spearman_rho"]) for row in candidates}
    matrix = np.asarray([[lookup.get((method, feature), np.nan) for method in METHODS] for feature in features])
    color_limit = max(0.3, math.ceil(float(np.nanmax(np.abs(matrix))) * 10) / 10)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.34 * len(features))))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-color_limit, vmax=color_limit)
    ax.set_xticks(range(len(METHODS)), [METHODS[m] for m in METHODS], rotation=35, ha="right")
    ax.set_yticks(range(len(features)), features)
    ax.set_title("Partial Spearman correlation with capped runtime\n(positive = longer runtime)")
    fig.colorbar(image, ax=ax, label="partial Spearman ρ")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return features


def make_cox_forest(cox: list[dict[str, object]], output: Path, method: str = "pairs_dynamic_four") -> None:
    selected = [
        row for row in cox
        if row["method"] == method and bool(row["converged"])
        and math.isfinite(float(row["hazard_ratio_per_adjusted_sd"]))
    ]
    selected = sorted(selected, key=lambda row: abs(float(row["coefficient"])), reverse=True)[:15]
    selected.reverse()
    labels = [str(row["feature"]) for row in selected]
    hazard = np.asarray([float(row["hazard_ratio_per_adjusted_sd"]) for row in selected])
    low = np.asarray([float(row["hr_ci95_low"]) for row in selected])
    high = np.asarray([float(row["hr_ci95_high"]) for row in selected])
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(selected))))
    ax.errorbar(hazard, y, xerr=[hazard - low, high - hazard], fmt="o", capsize=3)
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Completion hazard ratio per adjusted feature SD (<1 means slower)")
    ax.set_title("Clustered4 timeout-aware adjusted associations")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_speedup_quartiles(
    rows: list[dict[str, str]], correlations: list[dict[str, object]], output: Path, timeout: float,
) -> list[str]:
    target = "pairs_dynamic_four_speedup_vs_all_passes"
    selected = top_rows(correlations, target, "partial_spearman_rho", limit=6)
    features = [str(row["feature"]) for row in selected]
    method_time, _, method_valid = capped_method_outcome(rows, "pairs_dynamic_four", timeout)
    base_time, _, base_valid = capped_method_outcome(rows, "all_passes", timeout)
    speedup = np.divide(base_time, method_time, out=np.full_like(base_time, np.nan), where=method_time > 0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        x = numeric_column(rows, feature)
        valid = method_valid & base_valid & np.isfinite(x) & np.isfinite(speedup) & (speedup > 0)
        xv = x[valid]
        sv = speedup[valid]
        edges = np.quantile(xv, [0, 0.25, 0.5, 0.75, 1.0])
        quartiles = np.digitize(xv, edges[1:-1], right=True)
        gm = [math.exp(float(np.mean(np.log(sv[quartiles == q])))) for q in range(4)]
        counts = [int(np.sum(quartiles == q)) for q in range(4)]
        ax.bar(range(4), gm, color="#4472c4")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(range(4), [f"Q{q + 1}\nn={counts[q]}" for q in range(4)])
        ax.set_title(feature, fontsize=9)
        ax.set_ylabel("capped geometric-mean speedup")
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    fig.suptitle("Clustered4 speedup over pass-by-pass by feature quartile")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return features


def make_report(
    rows: list[dict[str, str]], features: list[str], correlations: list[dict[str, object]],
    cox: list[dict[str, object]], timeout: float, plots: list[str], output: Path,
) -> None:
    benchmark_counts = Counter(row["benchmark"] for row in rows)
    lines = [
        "# DESTIVA Phase 1: function-level characteristic analysis", "",
        "## Scope", "",
        f"This offline analysis contains **{len(rows)} supported functions** and **{len(features)} candidate features**. "
        "It uses saved validation outcomes and static characteristics of the original LLVM IR; it does not rerun Alive2 or Z3.", "",
        "The unit of observation is a whole function. Therefore, these results can identify characteristics associated with "
        "overall runtime or temporal-decomposition benefit, but cannot identify the cost of an individual validation segment.", "",
        "## Data audit", "",
        markdown_table(["Benchmark", "Functions"], [[name, count] for name, count in sorted(benchmark_counts.items())]), "",
    ]
    verdict_rows = []
    for method, label in METHODS.items():
        counts = Counter(row.get(f"{method}_result", "missing") for row in rows)
        verdict_rows.append([label, counts["correct"], counts["timeout"], counts["incorrect"], counts["missing"]])
    lines += [
        markdown_table(["Method", "Correct", "Timeout", "Incorrect", "Missing"], verdict_rows), "",
        f"Timeout-aware models right-censor timeout observations at **{timeout:g} seconds**. Incorrect results are excluded from "
        "method-specific runtime models rather than being treated as successful completions.", "",
        "## Interpretation guide", "",
        "- A positive runtime correlation means that larger feature values are associated with longer capped runtime.",
        "- A positive speedup correlation means that larger feature values are associated with greater benefit from Clustered4.",
        "- In the Cox models, a hazard ratio below 1 means a lower instantaneous chance of completion—that is, slower or more timeout-prone validation.",
        "- Partial correlations and Cox models control for original instruction count, total modified-instruction count, number of modified passes, pipeline length, benchmark, and segment count where available.",
        "- Reported p-values and confidence intervals are large-sample approximations; BH q-values correct each target/method family for multiple feature tests.", "",
    ]

    sections = [
        ("Clustered4 capped runtime", "pairs_dynamic_four_capped_runtime"),
        ("Clustered4 speedup versus end-to-end", "pairs_dynamic_four_speedup_vs_alive_tv"),
        ("Clustered4 speedup versus pass-by-pass", "pairs_dynamic_four_speedup_vs_all_passes"),
    ]
    for title, target in sections:
        top = top_rows(correlations, target, "partial_spearman_rho", 12)
        table_rows = [
            [
                f"`{row['feature']}`", f"{float(row['spearman_rho']):.3f}",
                f"{float(row['partial_spearman_rho']):.3f}", f"{float(row['q_value_bh']):.3g}", row["n"],
            ]
            for row in top
        ]
        lines += [f"## {title}", "", markdown_table(
            ["Feature", "Spearman ρ", "Adjusted ρ", "BH q", "n"], table_rows,
        ), ""]

    cox_top = [row for row in cox if row["method"] == "pairs_dynamic_four" and bool(row["converged"])]
    cox_top = sorted(cox_top, key=lambda row: abs(float(row["coefficient"])), reverse=True)[:15]
    lines += [
        "## Clustered4 timeout-aware associations", "",
        markdown_table(
            ["Feature", "Transform", "Hazard ratio/adjusted SD", "95% CI", "BH q"],
            [[
                f"`{row['feature']}`", row["feature_transform"],
                f"{float(row['hazard_ratio_per_adjusted_sd']):.3f}",
                f"[{float(row['hr_ci95_low']):.3f}, {float(row['hr_ci95_high']):.3f}]",
                f"{float(row['q_value_bh']):.3g}",
            ] for row in cox_top],
        ), "",
        "## Generated plots", "",
    ]
    lines.extend(f"- [{Path(plot).name}](plots/{Path(plot).name})" for plot in plots)
    lines += [
        "", "## Limitations", "",
        "1. Characteristics come from the original input IR, not from intermediate pass boundaries.",
        "2. Aggregate method time combines several validation segments and driver overhead.",
        "3. The capped-runtime correlations are descriptive and biased by the 600-second ceiling; prefer the Cox results when timeouts are common.",
        "4. Associations are not causal. Benchmark family, pass identity, and unmeasured solver-constraint structure can still confound them.",
        "5. This phase is suitable for selecting promising feature families. Segment-level experiments are required before claiming that a feature improves cut-point placement.", "",
        "## Machine-readable outputs", "",
        "- `correlations.csv`: raw and adjusted Spearman results for solved runtime, capped runtime, and capped speedup.",
        "- `cox_associations.csv`: adjusted timeout-aware models for the selected methods and features.",
        "- `features.csv`: the joined source dataset produced by `phase1_extract.py`.", "",
    ]
    output.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=here / "phase1_output" / "features.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "phase1_output")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--cox-methods", nargs="+", choices=tuple(METHODS),
        default=list(DEFAULT_COX_METHODS),
        help="methods for the slower timeout-aware models (default: E-E, P-P, Clustered4)",
    )
    args = parser.parse_args()

    try:
        rows, fields = read_dataset(args.input)
        features = feature_names(fields, rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = args.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        correlations = correlation_rows(rows, features, args.timeout)
        cox = cox_rows(rows, features, args.timeout, tuple(args.cox_methods))
        write_dict_csv(args.output_dir / "correlations.csv", correlations)
        write_dict_csv(args.output_dir / "cox_associations.csv", cox)

        heatmap = plots_dir / "runtime_partial_correlations.png"
        forest = plots_dir / "clustered4_cox_forest.png"
        quartiles = plots_dir / "clustered4_speedup_quartiles.png"
        make_heatmap(correlations, heatmap)
        make_cox_forest(cox, forest)
        make_speedup_quartiles(rows, correlations, quartiles, args.timeout)
        plots = [str(heatmap), str(forest), str(quartiles)]
        make_report(
            rows, features, correlations, cox, args.timeout, plots,
            args.output_dir / "report.md",
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Analyzed {len(rows)} functions and {len(features)} features", file=sys.stderr)
    print(f"Wrote Phase 1 outputs to {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
