# Affine QLoRA Training Pipeline

A modern parameter-efficient fine-tuning pipeline for Affine reasoning tasks using QLoRA (Quantized LoRA).

## Overview

This pipeline:
1. **Evaluates** the baseline Marco0/Affine-QQ model on SAT and ELR environments
2. **Generates** training data from Affine environments (SAT, ELR) 
3. **Fine-tunes** using QLoRA with 4-bit quantization
4. **Tracks** performance improvements

## Modern PEFT Techniques Used

- **QLoRA**: LoRA + 4-bit quantization for memory efficiency
- **DoRA**: Optional Direction + Magnitude decomposition (set `--use_dora True`)
- **Gradient Checkpointing**: Reduces memory usage
- **AdamW** optimizer with cosine scheduling

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
./run_training.sh
```

## Manual Usage

### 1. Baseline Evaluation
```bash
python evaluate_baseline.py
```
- Tests Marco0/Affine-QQ on 50 SAT + 10 ELR problems  
- Outputs accuracy metrics to `results/`

### 2. Training
```bash
python train_qlora.py \
    --model_name_or_path "Marco0/Affine-QQ" \
    --max_train_samples 500 \
    --num_train_epochs 2 \
    --learning_rate 2e-4
```

### 3. Key Parameters

**Data Generation:**
- `--max_train_samples`: Total training samples (default: 1000)
- SAT problems: 80% of samples
- ELR problems: 20% of samples

**LoRA Config:**
- `--lora_r`: Rank (default: 64)
- `--lora_alpha`: Alpha scaling (default: 16)
- `--lora_dropout`: Dropout rate (default: 0.1)
- `--use_dora`: Enable DoRA decomposition

**Training:**
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Accumulate gradients
- `--learning_rate`: Peak learning rate (2e-4 works well)

## Memory Requirements

- **QLoRA (4-bit)**: ~8-12GB VRAM for 7B models
- **Standard LoRA**: ~20-24GB VRAM
- **CPU fallback**: Possible but very slow

## Expected Results

**Baseline Marco0/Affine-QQ:**
- SAT (simple): ~10-30% accuracy
- ELR: ~5-15% accuracy

**After QLoRA fine-tuning:**
- Expected improvement: +10-20% absolute
- SAT: Better boolean logic reasoning
- ELR: Improved mathematical problem solving

## Output Structure

```
models/
├── affine_qlora_20241014_123456/    # Timestamped model
│   ├── adapter_model.bin            # LoRA weights
│   ├── adapter_config.json          # LoRA config
│   ├── training_config.json         # Full training config
│   └── training_data.json           # Generated training data

results/
├── sat_baseline_20241014_123456.json
├── elr_baseline_20241014_123456.json
└── baseline_summary_20241014_123456.json
```

## Advanced Usage

### Different Model Architectures
```bash
# Try different base models
python train_qlora.py --model_name_or_path "microsoft/DialoGPT-medium"
```

### Hyperparameter Tuning
```bash
# Higher rank LoRA (more parameters)
python train_qlora.py --lora_r 128 --lora_alpha 32

# Enable DoRA 
python train_qlora.py --use_dora True

# More training data
python train_qlora.py --max_train_samples 2000
```

### Multi-Environment Training
The pipeline can be extended to include ABD, DED, and HVM environments by modifying `DatasetGenerator` in `train_qlora.py`.

## Popular Parameter-Efficient Methods (2024/2025)

1. **QLoRA** ✅ (Implemented)
   - LoRA + 4-bit quantization
   - Best memory efficiency
   
2. **DoRA** ✅ (Available via `--use_dora`)
   - Decomposes into direction + magnitude
   - Often better than LoRA
   
3. **AdaLoRA** 
   - Adaptive rank allocation
   - Can be added via `adapters` library

4. **LongLoRA**
   - For extending context length
   - Useful for longer reasoning chains

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `--per_device_train_batch_size` to 1
- Increase `--gradient_accumulation_steps`
- Reduce `--max_seq_length`

**Poor Performance:**
- Increase `--max_train_samples` 
- Try `--use_dora True`
- Adjust `--learning_rate` (try 1e-4 or 5e-4)

**Slow Training:**
- Enable `--fp16` (default)
- Use multiple GPUs with `--ddp_find_unused_parameters False`