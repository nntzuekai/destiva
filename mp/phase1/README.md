# Phase 1: offline characteristic analysis

This pipeline studies relationships between characteristics of each original
LLVM IR function and the saved runtime of DESTIVA's temporal-decomposition
methods. It uses only artifacts already in the repository:

- `analysis/merged.jsonl` for saved runtimes and verdicts;
- `benchmarks/funcs/**/*.ll` for static IR characteristics; and
- `benchmarks/funcs/**/*_passes.json` for pass-level modification counts.

It does **not** run LLVM optimization, Alive2, Z3, `tester.py`, or `runner.sh`.

## Run it

From the repository root:

```sh
./mp/phase1/run_phase1.sh
```

To select another output directory:

```sh
./mp/phase1/run_phase1.sh /path/to/output
```

The extractor typically takes seconds. The statistical analysis takes a few
minutes on a laptop because it fits adjusted, timeout-aware models for 89
features. Python 3, NumPy, and Matplotlib are required; the artifact's Docker
image already supplies them through its analysis dependencies.

The two stages can also be run independently:

```sh
python3 mp/phase1/phase1_extract.py \
  --output mp/phase1/phase1_output/features.csv

MPLCONFIGDIR=/tmp/destiva-mpl \
python3 mp/phase1/phase1_analyze.py \
  --input mp/phase1/phase1_output/features.csv \
  --output-dir mp/phase1/phase1_output
```

By default, the slower Cox models are fit for end-to-end, pass-by-pass, and
Clustered4. Rank correlations cover all temporal methods. To request Cox
models for additional methods, use, for example:

```sh
python3 mp/phase1/phase1_analyze.py \
  --input mp/phase1/phase1_output/features.csv \
  --output-dir mp/phase1/phase1_output \
  --cox-methods alive_tv all_passes pairs_fixed pairs_mean pairs_top2 \
                pairs_dynamic pairs_dynamic_four
```

## Outputs

| Path | Description |
| --- | --- |
| `features.csv` | One row per supported function, containing static features and saved method outcomes. |
| `features.metadata.json` | Input paths, schema version, row/column counts, and parse errors. |
| `correlations.csv` | Raw and adjusted Spearman associations for solved runtime, capped runtime, and speedup. |
| `cox_associations.csv` | Adjusted proportional-hazards associations with timeouts right-censored at 600 seconds. |
| `report.md` | Data audit, strongest associations, interpretation guide, and limitations. |
| `plots/*.png` | Correlation heatmap, Clustered4 Cox forest plot, and speedup-by-quartile plot. |

## Feature families

The extractor produces the following classes of features:

- size and CFG: instructions, arguments, basic blocks, edges, and cyclomatic complexity;
- instruction mix: integer/floating arithmetic, comparisons, conversions,
  control flow, calls, vectors, aggregates, PHIs, and selects;
- types and operands: integer widths, vector lanes, pointer/type mentions,
  constants, SSA references, use counts, and dependency-depth proxies;
- memory: allocas, loads, stores, GEPs, atomics, volatile operations, and
  memory intrinsics; and
- pass changes: pipeline length, number of modifying passes, modification
  distribution, concentration, entropy, gaps, and positions.

The parser deliberately extracts only inexpensive characteristics that could
plausibly be available to a future cut-point heuristic. It recognizes LLVM IR
instructions directly from the shipped textual `.ll` files and coalesces
multiline instructions such as `switch`.

## Statistical interpretation

The report presents three complementary analyses:

1. **Solved-only Spearman correlation** uses only successful validations.
2. **Capped-time Spearman correlation** includes timeouts as 600 seconds. This
   matches the paper's descriptive convention, but the ceiling biases it.
3. **Cox proportional-hazards analysis** treats timeouts as right-censored.
   A hazard ratio below 1 means that a larger adjusted feature value is
   associated with slower or more timeout-prone completion.

Adjusted analyses control for the original instruction count, total modified
instruction count, number of modifying passes, pipeline length, benchmark,
and number of temporal segments where available. Numeric controls are not a
causal model; they only reduce obvious size and benchmark confounding.

Incorrect results are excluded from method-specific runtime models. The
analysis never treats a failed validation as a successful completion.

## Scope limitation

Every feature describes the original function or its aggregate pass-diff
history, while every timing describes a complete method invocation. Thus this
phase can identify characteristics associated with overall difficulty or the
benefit of Clustered4, but it cannot determine the runtime of an individual
validation segment. That requires the segment-level measurements proposed for
Phase 2.
