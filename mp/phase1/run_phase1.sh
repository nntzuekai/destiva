#!/bin/bash
# Run DESTIVA's offline Phase 1 characteristic analysis.
# Usage: ./mp/phase1/run_phase1.sh [output_directory]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/phase1_output}"

# Matplotlib otherwise tries to create a cache under the invoking user's home.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/destiva-phase1-matplotlib-${UID}}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR"

python3 "$SCRIPT_DIR/phase1_extract.py" \
  --merged "$ROOT_DIR/analysis/merged.jsonl" \
  --funcs-dir "$ROOT_DIR/benchmarks/funcs" \
  --output "$OUTPUT_DIR/features.csv"

python3 "$SCRIPT_DIR/phase1_analyze.py" \
  --input "$OUTPUT_DIR/features.csv" \
  --output-dir "$OUTPUT_DIR"

echo "Phase 1 report: $OUTPUT_DIR/report.md"
