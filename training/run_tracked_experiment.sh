#!/bin/bash

# Complete tracked experiment pipeline
# Usage: ./run_tracked_experiment.sh

set -e

echo "🧪 AFFINE TRACKED TRAINING EXPERIMENT"
echo "======================================"

# Setup
export PYTHONPATH="${PWD}/..:${PYTHONPATH}"
mkdir -p experiments mlruns

# Install MLflow if not present
echo "📦 Installing tracking dependencies..."
source ../training_env/bin/activate
uv pip install mlflow gitpython psutil

# Step 1: Generate baseline evaluation
echo ""
echo "📊 Step 1: Baseline Evaluation"
echo "------------------------------"
python evaluate_with_tracking.py

# Step 2: Run tracked training
echo ""
echo "🏋️ Step 2: Tracked Training"
echo "---------------------------"
python train_tracked.py \
    --experiment_name "affine_qlora_test" \
    --run_name "test_run_$(date +%Y%m%d_%H%M%S)" \
    --max_train_samples 32 \
    --seed 42 \
    --output_dir "./experiments/tracked_test" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --logging_steps 2 \
    --eval_strategy "steps" \
    --eval_steps 5 \
    --save_steps 10 \
    --report_to "none" \
    --no_cuda

# Step 3: Evaluate trained model
echo ""
echo "📈 Step 3: Post-Training Evaluation"
echo "-----------------------------------"
if [ -d "experiments/tracked_test" ]; then
    python -c "
import asyncio
from evaluate_with_tracking import run_full_evaluation

async def main():
    await run_full_evaluation('experiments/tracked_test', 'trained_model', 'post_training_eval')

asyncio.run(main())
"
else
    echo "⚠️ Trained model not found, skipping evaluation"
fi

# Step 4: Launch MLflow UI
echo ""
echo "🎯 EXPERIMENT COMPLETE!"
echo "======================="
echo "Results logged to MLflow. To view:"
echo ""
echo "  mlflow ui --backend-store-uri ./mlruns"
echo ""
echo "Then open: http://localhost:5000"
echo ""
echo "Key metrics tracked:"
echo "  • SAT Accuracy (target validator metric)"
echo "  • ELR Accuracy (math reasoning)"
echo "  • Training Loss (proxy tuning metric)"
echo ""
echo "Files created:"
find experiments/ -name "*.json" 2>/dev/null | head -5 | sed 's/^/  • /'
echo ""
echo "To start MLflow UI now, run:"
echo "  mlflow ui --backend-store-uri ./mlruns &"