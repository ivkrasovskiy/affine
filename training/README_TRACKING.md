# Experiment Tracking & Reproducibility

Simple experiment tracking with **MLflow** (free, local) focusing on 3 key metrics and proper reproducibility.

## 🎯 Key Metrics Tracked

### 1. **Target Validator Metrics** (what Affine validators evaluate):
- **SAT Accuracy**: Boolean satisfiability reasoning (primary target)
- **ELR Accuracy**: Mathematical reasoning (secondary target)

### 2. **Proxy Training Metrics** (tuning indicators):
- **Training Loss**: Model learning progress
- **Validation Loss**: Overfitting detection

### 3. **Meta Metrics**:
- **Overall Accuracy**: Combined performance across environments

## 📊 Data Splits

**Proper train/val/test splits with stratification:**
- **Train**: 70% (for training)
- **Validation**: 15% (for hyperparameter tuning)
- **Test**: 15% (for final evaluation, never seen during training)

Stratified by environment (SAT/ELR) to ensure balanced representation.

## 🔬 Reproducibility Features

### Experiment Metadata Tracked:
```
• Git commit hash & branch
• Python/PyTorch versions  
• System info (CPU, memory)
• All hyperparameters
• Dataset splits & sizes
• Random seeds
• Training duration
```

### File Organization:
```
experiments/
├── run_20241014_123456/
│   ├── model/              # Trained model
│   ├── logs/               # Training logs
│   ├── train_data.json     # Training set
│   ├── val_data.json       # Validation set
│   └── test_data.json      # Test set

mlruns/
├── 0/                      # Default experiment
│   ├── meta.yaml
│   └── <run_id>/
│       ├── metrics/        # All logged metrics
│       ├── params/         # All parameters
│       └── artifacts/      # Model & results
```

## 🚀 Quick Start

### 1. Run Complete Tracked Experiment:
```bash
./run_tracked_experiment.sh
```

### 2. Manual Training with Tracking:
```bash
python train_tracked.py \
    --experiment_name "my_experiment" \
    --run_name "baseline_test" \
    --max_train_samples 100 \
    --seed 42
```

### 3. Evaluate Specific Model:
```bash
python evaluate_with_tracking.py
```

### 4. View Results:
```bash
mlflow ui --backend-store-uri ./mlruns
# Open: http://localhost:5000
```

## 📈 Expected Results Flow

**Before Training (Baseline):**
```
SAT Accuracy:     ~0.100 (10%)
ELR Accuracy:     ~0.050 (5%) 
Overall Accuracy: ~0.080 (8%)
Training Loss:    N/A
```

**After Training (Target):**
```
SAT Accuracy:     ~0.300 (30%) ✅ +20% improvement
ELR Accuracy:     ~0.200 (20%) ✅ +15% improvement  
Overall Accuracy: ~0.270 (27%) ✅ +19% improvement
Training Loss:    ~8.5 → ~3.2  ✅ Decreasing trend
```

## 🔧 MLflow UI Features

### Experiments Dashboard:
- Compare multiple training runs
- Sort by metrics (SAT accuracy, etc.)
- Filter by parameters
- Track training progress

### Individual Run View:
- **Metrics**: Line plots of training/validation loss
- **Parameters**: All hyperparameters logged
- **Artifacts**: Model files, evaluation results
- **System**: Git info, hardware specs

### Model Comparison:
- Side-by-side metric comparison
- Statistical significance tests
- Hyperparameter correlation analysis

## 🎯 Metric Interpretation

### **SAT Accuracy** (Primary Target):
- **Range**: 0.0 - 1.0 (higher = better)
- **Baseline**: ~10% (random guessing with format constraints)
- **Good**: >30% (shows logical reasoning improvement)
- **Excellent**: >50% (strong Boolean logic understanding)

### **ELR Accuracy** (Secondary Target):
- **Range**: 0.0 - 1.0 (higher = better)  
- **Baseline**: ~5% (math problems are harder)
- **Good**: >20% (shows mathematical reasoning)
- **Excellent**: >40% (strong problem-solving skills)

### **Training Loss** (Proxy Metric):
- **Range**: Positive number (lower = better)
- **Baseline**: ~20-25 (untrained)
- **Good**: <10 (model learning patterns)
- **Excellent**: <5 (well-trained, not overfitted)

## ⚡ Performance Tips

### Memory Optimization:
```bash
# For limited VRAM
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
--max_seq_length 1024
```

### Fast Iteration:
```bash
# Quick testing
--max_train_samples 50
--num_train_epochs 1
--logging_steps 1
```

### Production Training:
```bash
# Full training
--max_train_samples 1000
--num_train_epochs 3
--eval_steps 50
```

## 🐛 Troubleshooting

**MLflow UI not accessible:**
```bash
# Check if MLflow is running
ps aux | grep mlflow

# Restart MLflow UI
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0
```

**Missing metrics:**
- Check MLflow callback is enabled
- Verify `report_to=[]` (disables wandb/tensorboard)
- Look for MLflow logs in training output

**Reproducibility issues:**
- Ensure seeds are set consistently
- Check git status is clean
- Verify environment consistency