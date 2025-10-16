#!/bin/bash
# Generate SAT dataset with proper difficulty (k=3, varied sizes n∈[10,30])
# Run in background with nohup

cd "$(dirname "$0")"
source ../.venv/bin/activate

export TOTAL_SAMPLES=2048
export SAT_RATIO=1.0
export SEED=42
export VARY_SIZE=true
export OUTPUT_DIR="../data"
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Starting dataset generation at $(date)"
echo "Total samples: $TOTAL_SAMPLES"
echo "SAT ratio: $SAT_RATIO"
echo "Size variation: $VARY_SIZE (k=3, n∈[10,30])"
echo "Output directory: $OUTPUT_DIR"

python generate_validator_data.py

echo "Dataset generation completed at $(date)"
