# Phase 2: segment-level measurement

Phase 2 creates a dataset in which one observation is one exact temporal
segment of an LLVM optimization pipeline. Candidate segments come from the
same heuristics used by `tester.py` and `runner.sh`; a deterministic sampler
keeps the experiment tractable.

This directory contains the collection pipeline. The expensive validation
runs are intended for the experiment server, not a development machine.

## What is implemented

1. `make_manifest.py` reproduces DESTIVA's temporal heuristics and builds a
   deterministic, heuristic-balanced segment sample.
2. `extract_boundary_features.py` runs LLVM once per selected function and
   extracts instruction, operand, memory, CFG, and dependency features from
   the intermediate IR at both ends of every selected segment.
3. `run_segments.py` validates one exact segment per process, with wall-time
   and memory limits, resumable JSONL output, repeated runs, and explicit
   temporal or combined spatial-temporal modes.
4. `merge_phase2.py` joins outcomes, segment-diff features, boundary features,
   and original-function features into one CSV.
5. `qc_report.py` audits completion, duplicate records, verdict categories,
   internal timing telemetry, benchmark coverage, and heuristic coverage.

The branch also adds `-tv-check-index-pairs=START:END` to
`alive2-decomp/tv/tv.cpp`. Unlike the original name-based option, indexed
pairs remain unambiguous when passes such as `InstCombinePass` occur several
times. `-tv-time-verify` emits `PHASE2_TIME` records when indexed pairs are
used.

## Observation definition

Indices refer to post-pass states in the same zero-based sequence stored by a
function's `_passes.json`. A segment `12:37` validates the transformation from
the state after pass 12 to the state after pass 37. Its modification features
therefore aggregate passes 13 through 37.

Each manifest record includes:

- exact start/end indices, pass names, and name-occurrence numbers;
- all heuristics that selected the segment;
- segment span and normalized positions;
- modification total, maximum, variance, concentration, entropy, and counts;
- a stable `segment_id` used for resume and joins.

Each run record distinguishes `correct`, `syntactic_skip`, `counterexample`
(an unsound transformation), `verification_failure` (not classified as
unsound), `timeout`, `unsupported`, `translation_error`, `crash`, and
`runner_error`. Timeouts are marked as right-censored.

## Server setup

The Dockerfile copies `mp/` into the image and compiles the indexed-pair
validator change. Rebuild the image from the `mp` branch:

```sh
git switch mp
git pull
docker build -t destiva-phase2 .
mkdir -p mp/phase2/work
docker run --rm -it \
  -v "$PWD/mp/phase2/work:/root/destiva/mp/phase2/work" \
  destiva-phase2 bash
cd /root/destiva
```

The bind mount preserves manifests and results after the container exits.
The scripts also work in a native Ubuntu 24.04 build with the same LLVM,
Alive2, and Z3 versions.

## 1. Create a pilot manifest

The following command samples 60 functions across all six benchmarks and at
most 12 segments per function:

```sh
python3 mp/phase2/make_manifest.py \
  --sample-functions 60 \
  --max-segments-per-function 12 \
  --output mp/phase2/work/pilot_manifest.jsonl
```

The default candidate pool contains:

- end-to-end;
- mean and median thresholds;
- Top2;
- Fixed;
- Clustered4 (`dynamic_four`); and
- all-modified-pass clustering (`dynamic`).

The sampler always retains the end-to-end segment and all Clustered4
segments, then fills the budget round-robin across the remaining heuristics.
Segments selected by multiple policies appear once with multiple
`selected_by` labels.

Pass-by-pass is excluded from the default candidate pool because it can add
roughly one segment per pass. Include it explicitly when desired:

```sh
python3 mp/phase2/make_manifest.py \
  --sample-functions 60 \
  --max-segments-per-function 20 \
  --heuristics end_to_end all_passes mean median top2 fixed dynamic_four dynamic \
  --output mp/phase2/work/pilot_with_pp.jsonl
```

Use `--max-segments-per-function 0` to retain the complete union of segments
selected by the requested heuristics. This still does not enumerate arbitrary
O(N²) pass pairs.

## 2. Extract intermediate-boundary features

```sh
python3 mp/phase2/extract_boundary_features.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --output mp/phase2/work/pilot_boundary_features.jsonl
```

LLVM is invoked once per function, not once per segment. Feature extraction
then reuses each requested intermediate state for every segment that shares
that boundary. The metadata file reports pass-count or pass-name alignment
failures; these must be zero before using boundary features.

For a smoke check, add `--limit-functions 2` and use a separate output file.

## 3. Preview validation commands

Execution is opt-in. Without `--execute`, the runner prints commands but does
not invoke LLVM, Alive2, or Z3:

```sh
python3 mp/phase2/run_segments.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --output mp/phase2/work/pilot_temporal_results.jsonl \
  --jobs 8
```

## 4. Collect temporal segment measurements

```sh
python3 mp/phase2/run_segments.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --output mp/phase2/work/pilot_temporal_results.jsonl \
  --timeout 600 \
  --memory-gb 4 \
  --jobs 8 \
  --execute
```

The output is flushed after every completed run. Repeating the same command
resumes collection by skipping existing `run_id` values. Use `--limit 20` for
a small live smoke test and `--repeats 3` for a runtime-stability subset.

To measure spatial plus temporal decomposition, use a separate result file:

```sh
python3 mp/phase2/run_segments.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --output mp/phase2/work/pilot_combined_results.jsonl \
  --mode combined \
  --timeout 600 --memory-gb 4 --jobs 8 --execute
```

Keeping temporal and combined results separate avoids mixing two different
cost populations.

## 5. Audit and merge

```sh
python3 mp/phase2/qc_report.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --results mp/phase2/work/pilot_temporal_results.jsonl \
  --output mp/phase2/work/pilot_qc.md

python3 mp/phase2/merge_phase2.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --results mp/phase2/work/pilot_temporal_results.jsonl \
  --boundary-features mp/phase2/work/pilot_boundary_features.jsonl \
  --output mp/phase2/work/pilot_dataset.csv
```

Do not proceed to modeling if the QC report shows pass alignment failures,
unknown segments, duplicate runs, or no internal `PHASE2_TIME` telemetry.

## Scaling beyond the pilot

Scale in stages rather than immediately applying a 600-second timeout to the
entire corpus:

1. 60 functions × 12 segments, with a small `--limit` live smoke test.
2. The complete pilot after inspecting the first QC report.
3. Several hundred functions with the same sampling budget.
4. All functions only after estimating completion time and timeout rate.

Keep concurrency fixed across comparable runs. Record server load, CPU model,
memory, Git commit, and container image identifier. The runner embeds the Git
commit, hostname, timeout, memory limit, and concurrency in each result.

## Minimal local tests

The tests exercise heuristic equivalence, deterministic sampling, manifest
construction, boundary-dump parsing, indexed command construction, and
verdict classification. They do not run the validator:

```sh
python3 -m unittest discover -s mp/phase2/tests -v
```

## Files

| File | Purpose |
| --- | --- |
| `phase2_common.py` | Heuristics, sampling, IDs, segment features, JSONL helpers. |
| `make_manifest.py` | Deterministic function and segment manifest generation. |
| `extract_boundary_features.py` | Intermediate LLVM IR feature extraction. |
| `run_segments.py` | Resumable, parallel, resource-limited validation runner. |
| `merge_phase2.py` | Join manifest, outcomes, boundary, and function features. |
| `qc_report.py` | Collection integrity and coverage report. |
| `schema.json` | Manifest record schema. |
| `tests/test_phase2.py` | Non-experimental unit tests. |
