# SAT Solver Fine-tuning Guide

**Last Updated:** 2025-10-15
**Model:** Marco0/Affine-QQ (Qwen3-2.5B)
**Task:** Boolean SAT problem solving with validator logic

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Dataset Generation](#dataset-generation)
3. [Training Configuration](#training-configuration)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Hardware Requirements](#hardware-requirements)
6. [Lessons Learned](#lessons-learned)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Environment Setup
```bash
cd /workspace/affine/training
source ../.venv/bin/activate

# Install dependencies (if needed)
source $HOME/.local/bin/env uv
uv pip install -r ../requirements.txt

# IMPORTANT: Install adapters FIRST, then downgrade transformers
uv pip install adapters
uv pip install transformers==4.51.0
```

### Generate Dataset
```bash
# Edit generate_large_dataset.sh for your needs:
export TOTAL_SAMPLES=8192        # Recommended: 8K-10K
export SAT_RATIO=1.0             # 100% SAT problems
export VARY_SIZE=true            # Vary problem sizes (critical!)
export SEED=42

# Run generation
bash generate_large_dataset.sh
```

### Start Training
```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
export DATA_PATH="../data/validator_dataset.json"

# Run in background
nohup python train_with_validator_logic.py > training.log 2>&1 &

# Monitor
tail -f training.log
```

### View Results (MLflow UI)
```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:///workspace/affine/training/mlruns --port 5000

# SSH port forward from your laptop:
# ssh -L 5000:localhost:5000 user@server
# Then open: http://localhost:5000
```

---

## Dataset Generation

### SAT Problem Difficulty

**CRITICAL: Use k=3 for proper difficulty!**

The validator library defaults to `k=10` which creates **trivially easy** problems:
- k=10: 99.9% probability any random assignment works (too easy!)
- k=3: 87.5% probability (proper 3-SAT difficulty)

**Problem Size Variation:**
- n (variables): **10-30** (randomized to prevent overfitting)
- k (literals/clause): **3** (fixed, standard 3-SAT)
- m (clauses): **4.26 × n** (phase transition ratio, auto-calculated)

### Configuration in `generate_validator_data.py`

```python
class ValidatorDataGenerator:
    def __init__(self, seed: int = 42, vary_size: bool = True):
        self.k = 3  # 3-SAT (proper difficulty)
        self.n_range = (10, 30) if vary_size else (15, 15)
        # m is auto-calculated as 4.26 × n
```

### Verify Dataset Quality

```python
import json
import random

with open('../data/validator_dataset.json', 'r') as f:
    data = json.load(f)

# Check one sample isn't trivial
sample = data[0]
n = sample['n_vars']
clauses = sample['validator_clauses']

# Test all-false
got_false = {i: False for i in range(1, n+1)}
ok_false = all(any((lit>0)==got_false.get(abs(lit), None) for lit in c) for c in clauses)

# Test all-true
got_true = {i: True for i in range(1, n+1)}
ok_true = all(any((lit>0)==got_true.get(abs(lit), None) for lit in c) for c in clauses)

print(f"All-false works: {ok_false}")  # Should be False
print(f"All-true works: {ok_true}")    # Should be False
print("✓ GOOD" if not (ok_false or ok_true) else "✗ TOO EASY")
```

---

## Training Configuration

### Current Setup (A5000 24GB)

```python
# Model
MODEL_NAME = "Marco0/Affine-QQ"  # Qwen3-2.5B, 4-bit quantized

# LoRA Configuration
LORA_R = 32                      # Rank (controls capacity)
LORA_ALPHA = 64                  # Alpha (usually 2× rank)
LORA_DROPOUT = 0.05

# Target modules (what gets fine-tuned)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",    # Attention
    "gate_proj", "up_proj", "down_proj"         # MLP
]
# Result: 66M trainable params (1.6% of 4B total)

# Training
BATCH_SIZE = 4
GRAD_ACCUM = 4                   # Effective batch: 16
EPOCHS = 3
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.1

# Evaluation
EVAL_STEPS = 50                  # Evaluate every 50 steps
EVAL_SAMPLES = 20                # Eval on 10 problems (//2 in code)
```

### Memory Usage (24GB A5000)
- Model (4-bit): ~8-10GB
- LoRA adapters: ~2-3GB
- Batch size 4: ~10-12GB
- Activations: ~3-4GB
- **Total: ~21-24GB** (maxed out)

---

## Evaluation Metrics

### Metrics Tracked in MLflow

1. **`validator_sat_accuracy`** (every 50 steps)
   - % of problems solved correctly
   - Requires both correct format AND satisfies all clauses

2. **`validator_format_accuracy`** (every 50 steps)
   - % of responses with valid `x1=True, x2=False` format
   - Parses ≥80% of expected variables
   - **Key diagnostic:** If high but solve accuracy low → model learned format but not logic

3. **`baseline_sat_accuracy`** (step 0)
   - Untrained model performance (usually 0%)

4. **`baseline_format_accuracy`** (step 0)
   - Untrained model format compliance

5. **Training loss** (logged frequently)
   - Cross-entropy loss during training
   - Should decrease steadily

### Interpreting Results

**Good Progress:**
- Format accuracy: 0% → 80%+
- Solve accuracy: 0% → 40%+
- Loss: Steady decrease

**Poor Results (what we saw):**
- Format accuracy: 0% → 20%
- Solve accuracy: 0% → 10-20%
- Loss: Plateaus after 100 steps

**Diagnosis:**
- Both low → Model hasn't learned task at all
- Format high, solve low → Learned output format but not solving logic
- Both high → Success!

---

## Hardware Requirements

### Current: A5000 (24GB VRAM)

**Limitations:**
- LoRA rank limited to ~32 (1.6% params)
- Batch size limited to 4
- Can't tune embeddings/lm_head (OOM)
- Training 1024 samples: ~54 minutes

**Verdict:** Marginal for algorithmic tasks like SAT solving

### Recommended: RTX 5090 (32GB VRAM)

**What +8GB enables:**

**Option 1: Max LoRA Capacity**
```python
LORA_R = 128              # 4x increase
LORA_ALPHA = 256
BATCH_SIZE = 4

target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens",       # NEW: helps with x1, x2 notation
    "lm_head"            # NEW: helps with True/False output
]

# Result: ~280M trainable (7%), ~29-31GB memory
```

**Option 2: Balanced**
```python
LORA_R = 96              # 3x increase
LORA_ALPHA = 192
BATCH_SIZE = 6           # 1.5x increase
# + embed_tokens, lm_head

# Result: ~200M trainable (5%), ~30GB memory
```

**Expected improvements:**
- 4-6x more LoRA capacity
- Can tune embeddings (better format learning)
- Can tune LM head (better output generation)
- Larger batch sizes (more stable training)
- **Target accuracy: 60-80%** (vs current 10-20%)

---

## Lessons Learned

### 1. SAT Problem Difficulty Matters

**Problem:**
- Validator defaults to k=10 (99.9% trivial)
- Model can get high accuracy without learning anything

**Solution:**
- Use k=3 (standard 3-SAT)
- Verify problems aren't trivially satisfied by all-true/all-false

### 2. Problem Size Variation is Critical

**Problem:**
- Fixed n=15 causes overfitting to specific problem size
- Model won't generalize to different sizes

**Solution:**
- Randomize n ∈ [10, 30]
- Keep k=3, m=4.26×n constant

### 3. LoRA Capacity vs Task Complexity

**Observation:**
- 1.6% trainable params works for format/style tasks
- Algorithmic reasoning (SAT) needs 5-10% params
- k=3 SAT is MUCH harder than k=10

**Evidence:**
- Loss plateaus after 100 steps → capacity bottleneck
- 10-20% accuracy on hard problems vs 20% on trivial problems

### 4. Dataset Size Requirements

**For SAT solving:**
- 1024 samples: Insufficient (current result: 10-20%)
- 8K-10K samples: Recommended minimum
- Format learning: ~1K samples
- Solving logic: ~10K samples

### 5. Format vs Solve Accuracy

**Key insight:**
Tracking both metrics reveals:
- Whether model learned output format
- Whether model learned solving logic
- Which is the bottleneck

**Action:**
Always track `validator_format_accuracy` separately from `validator_sat_accuracy`

### 6. Dependency Installation Order

**CRITICAL:**
```bash
# CORRECT order:
uv pip install adapters          # First
uv pip install transformers==4.51.0  # Then downgrade

# WRONG (will fail):
uv pip install transformers==4.51.0  # First
uv pip install adapters          # Conflict!
```

**Error if wrong:**
```
Model architectures ['Qwen3ForCausalLM'] are not supported
```

---

## Troubleshooting

### Training Process Issues

**"Training process killed"**
- Cause: OOM (out of memory)
- Fix: Reduce batch size or LoRA rank

**"Loss not decreasing"**
- Check learning rate (try 1e-5 to 5e-4)
- Check gradient norm (should be 0.05-0.2)
- Verify data quality

**"0% accuracy throughout training"**
- Check dataset format (validator_dataset.json vs training_dataset.json)
- Verify validator_solution and validator_clauses present
- Check generation parameters (k=3 not k=10)

### MLflow Issues

**"MLflow shows old experiment"**
- Check backend-store-uri path
- Kill all uvicorn workers: `pkill -9 uvicorn`
- Restart MLflow with correct path

**"Metrics not showing"**
- Check training process is alive: `ps aux | grep train_with`
- Verify mlruns directory has recent updates
- Refresh MLflow UI

### Dataset Issues

**"All problems are too easy"**
- Check k value (should be 3, not 10)
- Run validation script (see Dataset Generation section)
- Regenerate with correct parameters

**"KeyError: validator_solution"**
- Using wrong dataset file (training_dataset.json vs validator_dataset.json)
- Set: `DATA_PATH="../data/validator_dataset.json"`

---

## Next Steps for Better Results

### Immediate (on A5000):

1. **Generate more data:**
   ```bash
   export TOTAL_SAMPLES=10000
   bash generate_large_dataset.sh
   ```

2. **Train longer:**
   ```python
   EPOCHS = 10  # Instead of 3
   ```

3. **Increase evaluation samples:**
   ```python
   EVAL_SAMPLES = 50  # Instead of 20 (more reliable)
   ```

### With RTX 5090 (32GB):

1. **Increase LoRA capacity:**
   ```python
   LORA_R = 128
   LORA_ALPHA = 256
   target_modules += ["embed_tokens", "lm_head"]
   ```

2. **Scale up dataset:**
   ```python
   TOTAL_SAMPLES = 20000  # 2x more
   EPOCHS = 5-10
   ```

3. **Expected results:**
   - Format accuracy: 80-90%
   - Solve accuracy: 60-80%
   - Training time: 3-4 hours

---

## File Reference

**Key Files:**
- `train_with_validator_logic.py` - Main training script
- `generate_validator_data.py` - Dataset generation with validator logic
- `generate_large_dataset.sh` - Batch dataset generation script
- `validator_dataset.json` - Full dataset with validator metadata
- `training_dataset.json` - Simplified (prompt/completion only)

**Generated Outputs:**
- `mlruns/` - MLflow experiment tracking
- `output/model/` - Fine-tuned model checkpoints
- `training.log` - Training logs

**Monitoring:**
```bash
# Training progress
tail -f training.log

# MLflow metrics
mlflow ui --backend-store-uri file:///workspace/affine/training/mlruns --port 5000
```

---

## Summary

**What Works:**
✓ k=3 SAT problems (proper difficulty)
✓ Size variation (n ∈ [10,30])
✓ Validator-based evaluation
✓ Format + Solve accuracy tracking
✓ LoRA on attention + MLP

**What Needs Improvement:**
✗ LoRA capacity (1.6% → 5-7%)
✗ Dataset size (1K → 10K)
✗ Training epochs (3 → 10)
✗ Target modules (add embed_tokens, lm_head)

**Hardware Recommendation:**
- A5000 (24GB): Marginal for SAT, OK for experimentation
- **RTX 5090 (32GB): Recommended for production results**

**Expected Performance:**
- Current (A5000, 1K samples, r=32): **10-20% accuracy**
- Improved (5090, 10K samples, r=128): **60-80% accuracy**

---

*For questions or issues, check the main CLAUDE.md file or training logs.*
