#!/bin/bash
# Generate large dataset with 65536 samples (1024 * 64)
# Run in background with nohup

cd "$(dirname "$0")"
source ../.venv/bin/activate

export TOTAL_SAMPLES=65536
export SAT_RATIO=0.75
export SEED=42
export OUTPUT_DIR="../data"
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Starting dataset generation at $(date)"
echo "Total samples: $TOTAL_SAMPLES"
echo "SAT ratio: $SAT_RATIO"
echo "Output directory: $OUTPUT_DIR"

python generate_validator_data.py

echo "Dataset generation completed at $(date)"
