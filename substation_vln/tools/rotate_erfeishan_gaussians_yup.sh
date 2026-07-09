#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$ROOT/substation_vln/data/raw/220kv_erfeishan/gaussian"
DST_DIR="$ROOT/substation_vln/data/processed/220kv_erfeishan/gaussian_yup"
ROTATE_TOOL="$ROOT/external/habitat-gs/tools_gs/rotate_gs.py"

mkdir -p "$DST_DIR"

for i in 0 1 2; do
  input="$SRC_DIR/layer_${i}_point_cloud.ply"
  output="$DST_DIR/layer_${i}_yup.gs.ply"
  echo "Rotating layer_${i}: $input -> $output"
  python "$ROTATE_TOOL" \
    --input "$input" \
    --output "$output" \
    --rx -90
done

echo "Done. Y-up Gaussian files are in: $DST_DIR"
