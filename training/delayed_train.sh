#!/bin/bash
# Delayed training script - waits 15 minutes then starts training
# Run in background with nohup

cd "$(dirname "$0")"
source ../.venv/bin/activate

# Set environment variables
export HF_HUB_ENABLE_HF_TRANSFER=0
export DATA_PATH="../data/validator_dataset.json"

DELAY_MINUTES=15
DELAY_SECONDS=$((DELAY_MINUTES * 60))

echo "$(date): Delayed training script started"
echo "Will wait $DELAY_MINUTES minutes before starting training..."
echo "Training will begin at approximately $(date -d "+${DELAY_MINUTES} minutes" 2>/dev/null || date -v +${DELAY_MINUTES}M 2>/dev/null || echo "in $DELAY_MINUTES minutes")"

# Wait for the specified delay
sleep $DELAY_SECONDS

echo "$(date): Delay completed. Starting training now..."
python train_with_validator_logic.py

echo "$(date): Training completed"
