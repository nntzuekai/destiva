# DESTIVA Phase 1: function-level characteristic analysis

## Scope

This offline analysis contains **3025 supported functions** and **89 candidate features**. It uses saved validation outcomes and static characteristics of the original LLVM IR; it does not rerun Alive2 or Z3.

The unit of observation is a whole function. Therefore, these results can identify characteristics associated with overall runtime or temporal-decomposition benefit, but cannot identify the cost of an individual validation segment.

## Data audit

| Benchmark | Functions |
| --- | --- |
| bzip2 | 56 |
| fiat | 416 |
| gzip | 37 |
| oggenc | 213 |
| ph7 | 857 |
| sqlite3 | 1446 |

| Method | Correct | Timeout | Incorrect | Missing |
| --- | --- | --- | --- | --- |
| End-to-end | 1580 | 1445 | 0 | 0 |
| Pass-by-pass | 1588 | 1434 | 3 | 0 |
| Fixed | 1704 | 1318 | 3 | 0 |
| Cutout | 1570 | 1451 | 4 | 0 |
| Top2 | 1581 | 1440 | 4 | 0 |
| Clustered | 1340 | 1281 | 404 | 0 |
| Clustered4 | 1431 | 1212 | 382 | 0 |

Timeout-aware models right-censor timeout observations at **600 seconds**. Incorrect results are excluded from method-specific runtime models rather than being treated as successful completions.

## Interpretation guide

- A positive runtime correlation means that larger feature values are associated with longer capped runtime.
- A positive speedup correlation means that larger feature values are associated with greater benefit from Clustered4.
- In the Cox models, a hazard ratio below 1 means a lower instantaneous chance of completion—that is, slower or more timeout-prone validation.
- Partial correlations and Cox models control for original instruction count, total modified-instruction count, number of modified passes, pipeline length, benchmark, and segment count where available.
- Reported p-values and confidence intervals are large-sample approximations; BH q-values correct each target/method family for multiple feature tests.

## Clustered4 capped runtime

| Feature | Spearman ρ | Adjusted ρ | BH q | n |
| --- | --- | --- | --- | --- |
| `pass_diff_max` | 0.771 | 0.230 | 2.76e-30 | 2643 |
| `inst_load_count` | 0.771 | 0.224 | 5.96e-29 | 2643 |
| `modified_pass_position_mean` | 0.072 | -0.217 | 1.68e-27 | 2643 |
| `pass_diff_stddev` | 0.766 | 0.216 | 3.07e-27 | 2643 |
| `modified_pass_count` | 0.609 | -0.208 | 2.4e-25 | 2643 |
| `unmodified_pass_count` | -0.609 | 0.204 | 1.61e-24 | 2643 |
| `inst_load_ratio` | 0.492 | 0.195 | 1.71e-22 | 2643 |
| `pointer_type_mentions` | 0.765 | 0.190 | 2.02e-21 | 2643 |
| `modified_pass_weighted_position` | 0.054 | -0.181 | 1.27e-19 | 2643 |
| `inst_getelementptr_ratio` | 0.226 | 0.169 | 2.86e-17 | 2643 |
| `dependency_edge_count` | 0.756 | 0.164 | 2.69e-16 | 2643 |
| `inst_getelementptr_count` | 0.669 | 0.164 | 2.94e-16 | 2643 |

## Clustered4 speedup versus end-to-end

| Feature | Spearman ρ | Adjusted ρ | BH q | n |
| --- | --- | --- | --- | --- |
| `modified_pass_position_mean` | 0.209 | 0.278 | 1.94e-44 | 2643 |
| `modified_pass_gap_mean` | 0.236 | 0.230 | 9.85e-31 | 2643 |
| `modified_pass_weighted_position` | 0.151 | 0.224 | 4.13e-29 | 2643 |
| `last_modified_pass_position` | 0.098 | 0.198 | 4.85e-23 | 2643 |
| `modified_pass_gap_max` | 0.148 | 0.181 | 2.11e-19 | 2643 |
| `function_arguments` | 0.130 | 0.156 | 1.45e-14 | 2643 |
| `first_modified_pass_position` | 0.160 | 0.150 | 1.6e-13 | 2643 |
| `total_diff` | -0.103 | -0.130 | 3.03e-10 | 2643 |
| `ir_instruction_count` | -0.082 | 0.122 | 4.01e-09 | 2643 |
| `max_ssa_use_count` | -0.157 | -0.121 | 4.01e-09 | 2643 |
| `pass_diff_stddev` | -0.058 | 0.121 | 4.68e-09 | 2643 |
| `inst_alloca_count` | 0.027 | 0.120 | 5.19e-09 | 2643 |

## Clustered4 speedup versus pass-by-pass

| Feature | Spearman ρ | Adjusted ρ | BH q | n |
| --- | --- | --- | --- | --- |
| `modified_pass_position_mean` | 0.108 | 0.244 | 3.55e-34 | 2641 |
| `inst_call_ratio` | 0.232 | 0.240 | 3.16e-33 | 2641 |
| `inst_alloca_ratio` | 0.200 | 0.236 | 2e-32 | 2641 |
| `modified_pass_gap_mean` | 0.219 | 0.225 | 1.57e-29 | 2641 |
| `inst_alloca_count` | 0.009 | 0.194 | 4.71e-22 | 2641 |
| `global_refs_per_instruction` | 0.206 | 0.192 | 9.26e-22 | 2641 |
| `modified_pass_gap_max` | 0.081 | 0.176 | 2.32e-18 | 2641 |
| `last_modified_pass_position` | -0.010 | 0.175 | 2.32e-18 | 2641 |
| `dependency_depth_mean` | -0.181 | -0.170 | 2.46e-17 | 2641 |
| `modified_pass_weighted_position` | 0.055 | 0.159 | 2.8e-15 | 2641 |
| `inst_call_count` | 0.080 | 0.152 | 4.01e-14 | 2641 |
| `integer_type_mentions` | -0.157 | -0.141 | 3.53e-12 | 2641 |

## Clustered4 timeout-aware associations

| Feature | Transform | Hazard ratio/adjusted SD | 95% CI | BH q |
| --- | --- | --- | --- | --- |
| `zero_literal_mentions` | log1p | 0.744 | [0.705, 0.785] | 4.93e-25 |
| `inst_alloca_count` | log1p | 1.310 | [1.226, 1.400] | 2.49e-14 |
| `category_int_arith_count` | log1p | 1.257 | [1.173, 1.348] | 7.12e-10 |
| `category_conversion_count` | log1p | 1.255 | [1.181, 1.333] | 1.37e-12 |
| `inst_alloca_ratio` | raw | 1.254 | [1.191, 1.321] | 2.38e-16 |
| `dependency_depth_max` | log1p | 0.802 | [0.763, 0.842] | 1.17e-16 |
| `modified_pass_count` | log1p | 1.244 | [1.165, 1.328] | 4.39e-10 |
| `modified_pass_gap_mean` | log1p | 1.243 | [1.182, 1.308] | 9.08e-16 |
| `inst_getelementptr_count` | log1p | 0.807 | [0.767, 0.850] | 3.1e-15 |
| `inst_getelementptr_ratio` | raw | 0.815 | [0.774, 0.859] | 1.06e-13 |
| `modified_pass_position_mean` | raw | 1.226 | [1.168, 1.286] | 2.6e-15 |
| `inst_load_ratio` | raw | 0.817 | [0.777, 0.860] | 9.22e-14 |
| `dependency_depth_mean` | raw | 0.823 | [0.783, 0.864] | 9.22e-14 |
| `pass_diff_mean_nonzero` | log1p | 1.202 | [1.129, 1.279] | 3.34e-08 |
| `ir_instruction_count` | log1p | 0.835 | [0.796, 0.876] | 1.05e-12 |

## Generated plots

- [runtime_partial_correlations.png](plots/runtime_partial_correlations.png)
- [clustered4_cox_forest.png](plots/clustered4_cox_forest.png)
- [clustered4_speedup_quartiles.png](plots/clustered4_speedup_quartiles.png)

## Limitations

1. Characteristics come from the original input IR, not from intermediate pass boundaries.
2. Aggregate method time combines several validation segments and driver overhead.
3. The capped-runtime correlations are descriptive and biased by the 600-second ceiling; prefer the Cox results when timeouts are common.
4. Associations are not causal. Benchmark family, pass identity, and unmeasured solver-constraint structure can still confound them.
5. This phase is suitable for selecting promising feature families. Segment-level experiments are required before claiming that a feature improves cut-point placement.

## Machine-readable outputs

- `correlations.csv`: raw and adjusted Spearman results for solved runtime, capped runtime, and capped speedup.
- `cox_associations.csv`: adjusted timeout-aware models for the selected methods and features.
- `features.csv`: the joined source dataset produced by `phase1_extract.py`.
