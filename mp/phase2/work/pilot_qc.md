# DESTIVA Phase 2 quality-control report

## Collection status

- Manifest segments: **720**
- Result records: **720**
- Manifest segments with no result: **0**
- Results absent from the manifest: **0**
- Duplicate run IDs: **0**
- Records with internal verification timing: **254**
- Failed records lacking diagnostic output: **0**

## Verdicts

| Verdict | Runs |
| --- | --- |
| correct | 432 |
| syntactic_skip | 74 |
| timeout | 66 |
| unsupported | 143 |
| verification_failure | 5 |

## Manifest coverage by benchmark

| Benchmark | Segments |
| --- | --- |
| bzip2 | 120 |
| fiat | 120 |
| gzip | 120 |
| oggenc | 120 |
| ph7 | 120 |
| sqlite3 | 120 |

## Heuristic membership

| Heuristic | Manifest segments | Collected records |
| --- | --- | --- |
| dynamic | 190 | 190 |
| dynamic_four | 216 | 216 |
| end_to_end | 60 | 60 |
| fixed | 99 | 99 |
| mean | 212 | 212 |
| median | 247 | 247 |
| top2 | 255 | 255 |

## Checks requiring attention

- No structural collection problems detected.
