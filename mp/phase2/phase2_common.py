#!/usr/bin/env python3
"""Shared Phase 2 helpers for DESTIVA segment manifests and results."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
HEURISTICS = (
    "end_to_end",
    "all_passes",
    "mean",
    "median",
    "top2",
    "fixed",
    "dynamic_four",
    "dynamic",
)
DEFAULT_HEURISTICS = (
    "end_to_end",
    "mean",
    "median",
    "top2",
    "fixed",
    "dynamic_four",
    "dynamic",
)
DESTIVA_CUTPOINT_HEURISTICS = (
    "mean",
    "median",
    "top2",
    "fixed",
    "dynamic_four",
    "dynamic",
)
AUGMENTATION_STRATA = (
    "adjacent_early",
    "adjacent_middle",
    "adjacent_late",
    "multipass_short",
    "multipass_medium",
    "multipass_long",
)


def load_passes(path: Path) -> tuple[str, list[dict[str, object]]]:
    """Load a *_passes.json file and normalize its modification counts."""
    data = json.loads(path.read_text())
    normalized: list[dict[str, object]] = []
    occurrences: Counter[str] = Counter()
    for index, raw in enumerate(data["passes"]):
        name = str(raw["name"])
        occurrence = occurrences[name]
        occurrences[name] += 1
        if "diff" in raw:
            diff = int(raw.get("diff", 0))
        else:
            diff = sum(int(raw.get(key, 0)) for key in ("removed", "added", "modified"))
        entry = dict(raw)
        entry.update(index=index, name=name, occurrence=occurrence, diff=diff)
        normalized.append(entry)
    return str(data["function"]), normalized


def significant_changes(
    passes: Sequence[Mapping[str, object]], threshold_fn,
) -> list[tuple[int, str, int]]:
    changes = [
        (index, str(entry["name"]), int(entry["diff"]))
        for index, entry in enumerate(passes)
        if index > 0 and int(entry["diff"]) > 0
    ]
    if not changes:
        return []
    threshold = threshold_fn([diff for _, _, diff in changes])
    return sorted((item for item in changes if item[2] > threshold), key=lambda item: item[0])


def _segments_from_centers(n: int, centers: Iterable[int]) -> list[tuple[int, int]]:
    centers = sorted(set(centers))
    if n < 2:
        return []
    if not centers:
        return [(0, n - 1)]
    breakpoints = [0]
    for left, right in zip(centers, centers[1:]):
        breakpoints.append((left + right) // 2)
    breakpoints.append(n - 1)
    breakpoints = sorted(set(breakpoints))
    return [(left, right) for left, right in zip(breakpoints, breakpoints[1:]) if left < right]


def _threshold_segments(
    passes: Sequence[Mapping[str, object]], threshold_fn,
) -> list[tuple[int, int]]:
    n = len(passes)
    changes = significant_changes(passes, threshold_fn)
    breakpoints = sorted(set(
        [0]
        + [point for index, _, _ in changes for point in (index - 1, index)]
        + ([n - 1] if n else [])
    ))
    return [(left, right) for left, right in zip(breakpoints, breakpoints[1:]) if left < right]


def heuristic_segments(
    passes: Sequence[Mapping[str, object]], heuristic: str,
) -> list[tuple[int, int]]:
    """Reproduce the temporal segment choices made by tester.py."""
    if heuristic not in HEURISTICS:
        raise ValueError(f"unknown heuristic: {heuristic}")
    n = len(passes)
    if n < 2:
        return []
    if heuristic == "end_to_end":
        return [(0, n - 1)]
    if heuristic == "all_passes":
        return [(index - 1, index) for index in range(1, n)]
    if heuristic == "mean":
        return _threshold_segments(passes, statistics.mean)
    if heuristic == "median":
        return _threshold_segments(passes, statistics.median)
    if heuristic == "fixed":
        breakpoints = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        return [(left, right) for left, right in zip(breakpoints, breakpoints[1:]) if left < right]

    changed = sorted(
        ((index, int(entry["diff"])) for index, entry in enumerate(passes)
         if index > 0 and int(entry["diff"]) > 0),
        key=lambda item: -item[1],
    )
    if heuristic == "top2":
        chosen = sorted(changed[:2])
        breakpoints = sorted(set(
            [0]
            + [point for index, _ in chosen for point in (index - 1, index)]
            + [n - 1]
        ))
        return [(left, right) for left, right in zip(breakpoints, breakpoints[1:]) if left < right]
    if heuristic == "dynamic_four":
        return _segments_from_centers(n, (index for index, _ in changed[:4]))
    if heuristic == "dynamic":
        return _segments_from_centers(
            n, (index for index, entry in enumerate(passes) if index > 0 and int(entry["diff"]) > 0),
        )
    raise AssertionError("unreachable")


def collect_candidates(
    passes: Sequence[Mapping[str, object]], heuristics: Sequence[str],
) -> dict[tuple[int, int], set[str]]:
    candidates: dict[tuple[int, int], set[str]] = {}
    for heuristic in heuristics:
        for segment in heuristic_segments(passes, heuristic):
            candidates.setdefault(segment, set()).add(heuristic)
    return candidates


def stable_order_key(seed: int, scope: str, value: object) -> str:
    payload = f"{seed}\0{scope}\0{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def nonheuristic_augmentation_segments(
    passes: Sequence[Mapping[str, object]],
    excluded: Iterable[tuple[int, int]],
    seed: int,
    scope: str,
) -> dict[tuple[int, int], dict[str, object]]:
    """Select six diverse pairs outside the complete DESTIVA heuristic union."""
    n = len(passes)
    if n < 2:
        return {}
    forbidden = set(excluded)
    forbidden.update(collect_candidates(passes, DESTIVA_CUTPOINT_HEURISTICS))
    forbidden.add((0, n - 1))  # undecomposed end-to-end baseline

    strata: dict[str, list[tuple[int, int]]] = {
        name: [] for name in AUGMENTATION_STRATA
    }
    for start in range(n - 1):
        for end in range(start + 1, n):
            pair = (start, end)
            if pair in forbidden:
                continue
            span = end - start
            if span == 1:
                region = min(2, (3 * start) // max(1, n - 1))
                stratum = AUGMENTATION_STRATA[region]
            elif span <= 10:
                stratum = "multipass_short"
            elif span <= 40:
                stratum = "multipass_medium"
            else:
                stratum = "multipass_long"
            strata[stratum].append(pair)

    prefix = [0]
    for entry in passes:
        prefix.append(prefix[-1] + int(entry["diff"]))

    selected: dict[tuple[int, int], dict[str, object]] = {}
    tertile_names = ("low", "middle", "high")
    for stratum in AUGMENTATION_STRATA:
        candidates = strata[stratum]
        if not candidates:
            raise ValueError(f"no eligible non-heuristic candidates in {stratum}")

        def diff_total(pair: tuple[int, int]) -> int:
            start, end = pair
            return prefix[end + 1] - prefix[start + 1]

        ranked = sorted(
            candidates,
            key=lambda pair: (
                diff_total(pair), stable_order_key(seed, f"{scope}:{stratum}:rank", pair),
            ),
        )
        buckets: list[list[tuple[int, int]]] = [[], [], []]
        for index, pair in enumerate(ranked):
            bucket = min(2, (3 * index) // len(ranked))
            buckets[bucket].append(pair)
        available = [index for index, values in enumerate(buckets) if values]
        selector = int(stable_order_key(seed, scope, f"{stratum}:tertile")[:16], 16)
        tertile = available[selector % len(available)]
        chosen = min(
            buckets[tertile],
            key=lambda pair: stable_order_key(seed, f"{scope}:{stratum}:choose", pair),
        )
        selected[chosen] = {
            "selection_policy": "nonheuristic_augmentation_v1",
            "sampling_stratum": stratum,
            "sampling_diff_tertile": tertile_names[tertile],
            "sampling_stratum_candidate_count": len(candidates),
        }
    return selected


def select_candidates(
    candidates: Mapping[tuple[int, int], set[str]],
    max_segments: int,
    seed: int,
    scope: str,
) -> list[tuple[int, int]]:
    """Choose a deterministic, heuristic-balanced subset of candidate segments."""
    all_segments = sorted(candidates, key=lambda item: stable_order_key(seed, scope, item))
    if max_segments <= 0 or len(all_segments) <= max_segments:
        return all_segments

    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()

    def add(segment: tuple[int, int]) -> None:
        if segment not in selected_set and len(selected) < max_segments:
            selected.append(segment)
            selected_set.add(segment)

    # Always keep the end-to-end baseline and all Clustered4 segments when present.
    for segment in all_segments:
        if "end_to_end" in candidates[segment]:
            add(segment)
    for segment in all_segments:
        if "dynamic_four" in candidates[segment]:
            add(segment)

    by_heuristic = {
        heuristic: [segment for segment in all_segments if heuristic in candidates[segment]]
        for heuristic in HEURISTICS
    }
    cursor = {heuristic: 0 for heuristic in HEURISTICS}
    while len(selected) < max_segments:
        progressed = False
        for heuristic in HEURISTICS:
            segments = by_heuristic[heuristic]
            while cursor[heuristic] < len(segments):
                segment = segments[cursor[heuristic]]
                cursor[heuristic] += 1
                if segment not in selected_set:
                    add(segment)
                    progressed = True
                    break
            if len(selected) >= max_segments:
                break
        if not progressed:
            break
    for segment in all_segments:
        add(segment)
    return selected


def segment_features(
    passes: Sequence[Mapping[str, object]], start: int, end: int,
) -> dict[str, float | int | str]:
    if not (0 <= start < end < len(passes)):
        raise ValueError(f"invalid segment {start}:{end} for {len(passes)} passes")
    # The interval is between the post-pass states at start and end. Changes
    # caused by passes start+1 through end therefore belong to the segment.
    interval = passes[start + 1:end + 1]
    diffs = [int(entry["diff"]) for entry in interval]
    nonzero = [value for value in diffs if value > 0]
    total = sum(diffs)
    mean = statistics.fmean(nonzero) if nonzero else 0.0
    stdev = statistics.pstdev(nonzero) if len(nonzero) > 1 else 0.0
    entropy = 0.0
    if total > 0:
        entropy = -sum((value / total) * math.log(value / total) for value in nonzero)
    return {
        "segment_span": end - start,
        "segment_start_position": start / max(1, len(passes) - 1),
        "segment_end_position": end / max(1, len(passes) - 1),
        "segment_diff_total": total,
        "segment_diff_max": max(nonzero, default=0),
        "segment_diff_mean_nonzero": mean,
        "segment_diff_stddev": stdev,
        "segment_diff_top1_share": max(nonzero, default=0) / total if total else 0.0,
        "segment_diff_entropy": entropy,
        "segment_modified_pass_count": len(nonzero),
        "segment_unmodified_pass_count": len(diffs) - len(nonzero),
        "segment_start_pass": str(passes[start]["name"]),
        "segment_end_pass": str(passes[end]["name"]),
        "segment_start_occurrence": int(passes[start]["occurrence"]),
        "segment_end_occurrence": int(passes[end]["occurrence"]),
    }


def segment_id(relative_ir: str, start: int, end: int) -> str:
    readable = f"{Path(relative_ir).with_suffix('').as_posix()}:{start}:{end}"
    digest = hashlib.sha256(readable.encode()).hexdigest()[:12]
    return f"{readable}:{digest}"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    if not path.exists():
        return records
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    return records
