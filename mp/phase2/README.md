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
2. `extract_boundary_features.py` first requires the live LLVM pass sequence
   to exactly match each function's `_passes.json`. It then persists the exact
   post-pass IR at every selected boundary and extracts instruction, operand,
   memory, CFG, and dependency features.
3. `run_segments.py` invokes standalone `alive-tv` on each exact start/end IR
   pair, with internal verification timing, wall-time and memory limits,
   resumable JSONL output, repeated runs, and temporal or combined modes.
4. `merge_phase2.py` joins outcomes, segment-diff features, boundary features,
   and original-function features into one CSV.
5. `qc_report.py` audits completion, duplicate records, verdict categories,
   internal timing telemetry, benchmark coverage, and heuristic coverage.

Standalone `alive-tv --time-verify` emits one `PHASE2_TIME` record for the
named function. The collection runner requires that record for every
non-timeout result. This avoids relying on LLVM pass-manager callback indices,
which include adaptor callbacks that are absent from `_passes.json`.

### Incompatible pilot-v1 results

Results created with `-tv-check-index-pairs` are diagnostic only and must not
be resumed or merged. The corrected runner emits schema version 2 with
`measurement_method=exact_snapshot_pair_v1` and refuses legacy result files.
Use new boundary, result, QC, and dataset filenames for corrected runs.

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

### Docker

The Dockerfile copies `mp/` into the image and compiles the standalone timing
instrumentation. Rebuild the image from the `mp` branch:

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

### Native Ubuntu 24.04

Use an Ubuntu 24.04 checkout of the `mp` branch:

```sh
git switch mp
git pull
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build git python3 re2c zlib1g-dev util-linux
```

Build LLVM, Z3, and Alive2 using the native setup in the repository's main
README. LLVM must be the pinned commit documented there. If the project is
already built, rebuild Alive2 so the standalone timing instrumentation is
included:

```sh
cmake --build alive2-decomp/build -j 16
alive2-decomp/build/alive-tv --help 2>&1 | grep time-verify
mkdir -p mp/phase2/work
```

`util-linux` supplies `prlimit`, which the experiment driver uses to impose a
per-process memory limit. By default the scripts look for LLVM at
`llvm-project/build/bin/opt` and standalone Alive2 at
`alive2-decomp/build/alive-tv`; override those paths with `--opt` and
`--alive-tv` if your native builds are elsewhere.

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

## 2. Preflight and materialize exact boundaries

```sh
python3 mp/phase2/extract_boundary_features.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --output mp/phase2/work/pilot_v2_boundary_features.jsonl \
  --snapshots-dir mp/phase2/work/pilot_v2_snapshots
```

LLVM is invoked once per function, not once per segment. Feature extraction
reuses each requested intermediate state for every segment sharing that
boundary. Start/end snapshot paths and SHA-256 hashes are stored in every
boundary record.

This command is a hard preflight gate. It exits nonzero unless the live pass
count and every pass name exactly match `_passes.json`. On success,
`pilot_v2_boundary_features.metadata.json` must report:

- `complete: true`;
- `alignment_error_count: 0`;
- equal `requested_functions`, `aligned_functions`, and `completed_functions`;
- equal `requested_segments` and `segment_count`; and
- an empty `errors` array.

If it fails, do not run validation. Confirm that `llvm-project` is at the
pinned commit from the main README and rebuild LLVM. The metadata identifies
the first differing pass for every affected function.

For a smoke check, add `--limit-functions 2` and use a separate output file.

## 3. Preview validation commands

Execution is opt-in. Without `--execute`, the runner prints commands but does
not invoke LLVM, Alive2, or Z3:

```sh
python3 mp/phase2/run_segments.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --boundary-features mp/phase2/work/pilot_v2_boundary_features.jsonl \
  --output mp/phase2/work/pilot_v2_temporal_results.jsonl \
  --jobs 8
```

## 4. Collect temporal segment measurements

```sh
python3 mp/phase2/run_segments.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --boundary-features mp/phase2/work/pilot_v2_boundary_features.jsonl \
  --output mp/phase2/work/pilot_v2_temporal_results.jsonl \
  --timeout 600 \
  --memory-gb 4 \
  --jobs 8 \
  --execute
```

Before execution, the runner verifies complete manifest coverage and every
snapshot hash. The output is flushed after every completed run. Repeating the
same command resumes collection by skipping existing `run_id` values. Use
`--limit 20` for a small live smoke test and `--repeats 3` for a
runtime-stability subset. A legacy result file causes an immediate error.

To measure spatial plus temporal decomposition, use a separate result file:

```sh
python3 mp/phase2/run_segments.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --boundary-features mp/phase2/work/pilot_v2_boundary_features.jsonl \
  --output mp/phase2/work/pilot_v2_combined_results.jsonl \
  --mode combined \
  --timeout 600 --memory-gb 4 --jobs 8 --execute
```

Keeping temporal and combined results separate avoids mixing two different
cost populations.

## 5. Audit and merge

```sh
python3 mp/phase2/qc_report.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --results mp/phase2/work/pilot_v2_temporal_results.jsonl \
  --boundary-features mp/phase2/work/pilot_v2_boundary_features.jsonl \
  --output mp/phase2/work/pilot_v2_qc.md

python3 mp/phase2/merge_phase2.py \
  --manifest mp/phase2/work/pilot_manifest.jsonl \
  --results mp/phase2/work/pilot_v2_temporal_results.jsonl \
  --boundary-features mp/phase2/work/pilot_v2_boundary_features.jsonl \
  --output mp/phase2/work/pilot_v2_dataset.csv
```

`qc_report.py` exits nonzero unless the report status is `PASS`.
`merge_phase2.py` independently refuses incomplete boundary coverage, legacy
or mixed measurement methods, snapshot-digest mismatches, invalid runs, and
non-timeout runs lacking internal timing.

## Scaling beyond the pilot

Scale in stages rather than immediately applying a 600-second timeout to the
entire corpus:

1. 60 functions × 12 segments, with a small `--limit` live smoke test.
2. The complete pilot after inspecting the first QC report.
3. Several hundred functions with the same sampling budget.
4. All functions only after estimating completion time and timeout rate.

Keep concurrency fixed across comparable runs. Record server load, CPU model,
memory, LLVM/Z3 versions, Git commit, and build type. Also record the container
image identifier for Docker runs, or the operating-system and kernel versions
for native runs. The runner embeds the Git commit, hostname, timeout, memory
limit, and concurrency in each result.

## Minimal local tests

The tests exercise heuristic equivalence, deterministic sampling, manifest
construction, strict pass alignment, snapshot hashes, exact-pair command
construction, legacy-result rejection, and verdict classification. They do
not run the validator:

```sh
python3 -m unittest discover -s mp/phase2/tests -v
```

## Files

| File | Purpose |
| --- | --- |
| `phase2_common.py` | Heuristics, sampling, IDs, segment features, JSONL helpers. |
| `make_manifest.py` | Deterministic function and segment manifest generation. |
| `extract_boundary_features.py` | Strict pass preflight, exact snapshots, and boundary features. |
| `run_segments.py` | Resumable standalone `alive-tv` exact-pair runner. |
| `merge_phase2.py` | Join manifest, outcomes, boundary, and function features. |
| `qc_report.py` | Collection integrity and coverage report. |
| `schema.json` | Manifest record schema. |
| `tests/test_phase2.py` | Non-experimental unit tests. |
