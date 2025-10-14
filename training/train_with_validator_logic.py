#!/usr/bin/env python3
"""
Training pipeline using validator data generation logic
Incorporates best practices from colleague's code + exact validator SAT/ELR generation
"""

import os
import json
import asyncio
import logging
import torch
import mlflow
import mlflow.pytorch
import git
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import nullcontext

# Core ML libraries
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForSeq2Seq, TrainerCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from tqdm.auto import tqdm, trange

# Our validator data generator
from generate_validator_data import ValidatorDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== ENVIRONMENT CONFIGURATION ==============
# All hyperparameters via environment variables (from colleague's best practices)

# Model & Data
MODEL_NAME = os.getenv("MODEL_NAME", "Marco0/Affine-QQ")  # Use production model that handles long sequences
OUT_DIR = os.getenv("OUT_DIR", "./experiments/validator_training")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "4096"))  # Increased for full SAT prompts (~3200 tokens)

# Training
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
EPOCHS = float(os.getenv("EPOCHS", "2.0"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))

# LoRA
LORA_R = int(os.getenv("LORA_R", "32"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))

# Evaluation
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "50"))
LOG_STEPS = int(os.getenv("LOG_STEPS", "10"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "100"))
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "20"))

# Generation (for evaluation)
GEN_MAX_NEW = int(os.getenv("GEN_MAX_NEW", "128"))
GEN_TEMP = float(os.getenv("GEN_TEMP", "0.1"))
GEN_DO_SAMPLE = bool(int(os.getenv("GEN_DO_SAMPLE", "0")))

# System
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
SEED = int(os.getenv("SEED", "42"))

# Data generation
DATA_SAMPLES = int(os.getenv("DATA_SAMPLES", "100"))  # Start small for testing

# ============== MLFLOW SETUP ==============

def setup_mlflow_experiment(experiment_name: str = "validator_training") -> str:
    """Setup MLflow experiment"""
    
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
    
    # Create/get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    # Start run
    run_name = f"validator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    
    # Log environment params
    env_params = {
        "model_name": MODEL_NAME,
        "data_samples": DATA_SAMPLES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "lora_r": LORA_R,
        "seed": SEED
    }
    
    mlflow.log_params(env_params)
    
    return mlflow.active_run().info.run_id

# ============== ENHANCED EVALUATION CALLBACK ==============

class ValidatorMetricsCallback(TrainerCallback):
    """Real-time evaluation using validator logic"""
    
    def __init__(self, eval_samples: List[Dict], tokenizer, eval_every: int = EVAL_STEPS):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        
        # Create generator for validation
        self.generator = ValidatorDataGenerator(seed=SEED)
        
        # Prepare evaluation samples
        self.sat_samples = [s for s in eval_samples if s.get("environment") == "SAT"]
        self.elr_samples = [s for s in eval_samples if s.get("environment") == "ELR"]
        
        logger.info(f"Evaluation callback: {len(self.sat_samples)} SAT, {len(self.elr_samples)} ELR")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run validator-based evaluation"""
        
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return control
            
        logger.info(f"🔍 Validator evaluation at step {state.global_step}")
        
        model.eval()
        
        try:
            # Evaluate using validator logic
            sat_accuracy = self._evaluate_sat_with_validator(model)
            elr_accuracy = self._evaluate_elr_with_validator(model)
            overall_accuracy = (sat_accuracy + elr_accuracy) / 2
            
            # Log to MLflow
            step = int(state.global_step)
            mlflow.log_metric("validator_sat_accuracy", sat_accuracy, step=step)
            mlflow.log_metric("validator_elr_accuracy", elr_accuracy, step=step)
            mlflow.log_metric("validator_overall_accuracy", overall_accuracy, step=step)
            
            logger.info(f"Step {step}: SAT={sat_accuracy:.3f}, ELR={elr_accuracy:.3f}, Overall={overall_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
        finally:
            model.train()
            
        return control
    
    def _evaluate_sat_with_validator(self, model) -> float:
        """Evaluate SAT using exact validator validation logic"""
        if not self.sat_samples:
            return 0.0
            
        correct = 0
        total = min(EVAL_SAMPLES // 2, len(self.sat_samples))
        
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 and torch.cuda.is_available() else nullcontext()
        
        for i in range(total):
            sample = self.sat_samples[i]
            prompt = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.inference_mode(), amp_ctx:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=GEN_MAX_NEW,
                        temperature=GEN_TEMP,
                        do_sample=GEN_DO_SAMPLE,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Use validator's exact validation logic
                if self.generator.validate_sat_response(sample, response):
                    correct += 1
                
                # Memory management (from colleague's code)
                del inputs, outputs
                if i % 4 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error evaluating SAT sample {i}: {e}")
        
        return correct / max(total, 1)
    
    def _evaluate_elr_with_validator(self, model) -> float:
        """Evaluate ELR using exact validator validation logic"""
        if not self.elr_samples:
            return 0.0
            
        correct = 0
        total = min(EVAL_SAMPLES // 2, len(self.elr_samples))
        
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 and torch.cuda.is_available() else nullcontext()
        
        for i in range(total):
            sample = self.elr_samples[i]
            prompt = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.inference_mode(), amp_ctx:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=GEN_MAX_NEW,
                        temperature=GEN_TEMP,
                        do_sample=GEN_DO_SAMPLE,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Use validator's exact validation logic
                if self.generator.validate_elr_response(sample, response):
                    correct += 1
                
                # Memory management
                del inputs, outputs
                if i % 4 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error evaluating ELR sample {i}: {e}")
        
        return correct / max(total, 1)

# ============== MLFLOW LOGGING CALLBACK ==============

class MLflowCallback(TrainerCallback):
    """Log training metrics to MLflow"""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=state.global_step)

# ============== TRAINING FUNCTIONS ==============

def load_model_simple(model_name: str = MODEL_NAME):
    """Load model with 4-bit quantization for memory efficiency"""

    logger.info(f"Loading model with 4-bit quantization: {model_name}")

    # Import Qwen3 classes (available in transformers 4.51.3+)
    try:
        from transformers import Qwen3ForCausalLM
        logger.info("Using Qwen3ForCausalLM directly")
    except ImportError:
        logger.warning("Qwen3ForCausalLM not available, falling back to Qwen2")
        from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM

    # Use AutoTokenizer for tokenizer (it auto-detects the right class)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 4-bit quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model using Qwen3ForCausalLM with 4-bit quantization
    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded with LoRA adapters")

    return model, tokenizer

def format_instruction(sample: Dict[str, str]) -> Dict[str, str]:
    """Format sample for instruction following with response masking"""
    prompt = sample["prompt"]
    completion = sample["completion"]
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    full_text = formatted_prompt + completion
    
    # Store the response start position for loss masking
    response_start_len = len(formatted_prompt)
    
    return {
        "input_text": formatted_prompt, 
        "target_text": full_text,
        "response_start": response_start_len
    }

def create_dataset(data: List[Dict[str, str]], tokenizer, max_seq_length: int = MAX_SEQ_LEN):
    """Create tokenized dataset with assistant-only loss masking"""
    
    formatted_data = [format_instruction(sample) for sample in data]
    
    def tokenize_function(examples):
        # Tokenize full text
        targets = tokenizer(
            examples["target_text"], 
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_tensors=None
        )
        
        # Tokenize just the prompt part to find where response starts
        prompts = tokenizer(
            examples["input_text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_tensors=None
        )
        
        # Create labels with masked prompt (only train on response)
        labels = targets["input_ids"].copy()
        prompt_len = len(prompts["input_ids"])
        
        # Mask the prompt tokens (set to -100 so they're ignored in loss)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
            
        # Log masking stats for first sample
        if hasattr(tokenize_function, '_first_log') == False:
            tokenize_function._first_log = True
            total_tokens = len(labels)
            masked_tokens = sum(1 for x in labels if x == -100)
            response_tokens = total_tokens - masked_tokens
            print(f"📊 Assistant-only loss masking: {masked_tokens}/{total_tokens} tokens masked, training on {response_tokens} response tokens")
            
        return {
            "input_ids": targets["input_ids"],
            "attention_mask": targets["attention_mask"],
            "labels": labels
        }
    
    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        desc="Tokenizing data"
    )
    
    return tokenized_dataset

def split_dataset(data: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split dataset with stratification"""
    from sklearn.model_selection import train_test_split
    
    sat_data = [item for item in data if item.get("environment") == "SAT"]
    elr_data = [item for item in data if item.get("environment") == "ELR"]
    
    def split_env_data(env_data, train_r, val_r):
        if len(env_data) < 3:
            return env_data, [], []
        
        train, temp = train_test_split(env_data, train_size=train_r, random_state=SEED)
        
        if len(temp) < 2:
            return train, temp, []
        
        val_size = val_r / (val_r + (1 - train_r - val_r))
        val, test = train_test_split(temp, train_size=val_size, random_state=SEED)
        
        return train, val, test
    
    sat_train, sat_val, sat_test = split_env_data(sat_data, train_ratio, val_ratio)
    elr_train, elr_val, elr_test = split_env_data(elr_data, train_ratio, val_ratio)
    
    train_data = sat_train + elr_train
    val_data = sat_val + elr_val  
    test_data = sat_test + elr_test
    
    logger.info(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data

# ============== MAIN TRAINING FUNCTION ==============

async def main():
    """Main training pipeline with validator logic"""
    
    logger.info("🧪 VALIDATOR-BASED AFFINE TRAINING")
    logger.info("=" * 50)
    
    # Set reproducibility
    torch.manual_seed(SEED)
    
    # Setup MLflow
    run_id = setup_mlflow_experiment("validator_training")
    logger.info(f"MLflow run ID: {run_id}")
    
    try:
        # Create output directory
        output_dir = Path(OUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load pre-generated data from file
        # IMPORTANT: Use validator_dataset.json which contains full metadata including validator_solution
        DATA_PATH = os.getenv("DATA_PATH", "../data/validator_dataset.json")
        logger.info(f"📊 Loading pre-generated data from {DATA_PATH}...")

        with open(DATA_PATH, "r") as f:
            all_data = json.load(f)

        logger.info(f"Loaded {len(all_data)} samples")

        if len(all_data) < 10:
            raise RuntimeError("Insufficient training data in dataset")
        
        # Split dataset
        train_data, val_data, test_data = split_dataset(all_data)
        
        # Log dataset metrics
        mlflow.log_metric("dataset_total", len(all_data))
        mlflow.log_metric("dataset_train", len(train_data))
        mlflow.log_metric("dataset_val", len(val_data))
        mlflow.log_metric("dataset_test", len(test_data))
        
        # Save datasets
        with open(output_dir / "train_data.json", "w") as f:
            json.dump(train_data, f, indent=2)
        with open(output_dir / "val_data.json", "w") as f:
            json.dump(val_data, f, indent=2)
        with open(output_dir / "test_data.json", "w") as f:
            json.dump(test_data, f, indent=2)
        
        # Load model
        logger.info("🤖 Loading model...")
        model, tokenizer = load_model_simple(MODEL_NAME)

        # BASELINE EVALUATION on test set before training
        logger.info("📊 Evaluating BASELINE model on test set...")
        callback = ValidatorMetricsCallback(test_data, tokenizer)
        baseline_sat = callback._evaluate_sat_with_validator(model)
        baseline_elr = callback._evaluate_elr_with_validator(model)
        baseline_overall = (baseline_sat + baseline_elr) / 2 if baseline_elr > 0 else baseline_sat

        logger.info(f"BASELINE - SAT: {baseline_sat:.3f}, ELR: {baseline_elr:.3f}, Overall: {baseline_overall:.3f}")
        mlflow.log_metric("baseline_sat_accuracy", baseline_sat)
        mlflow.log_metric("baseline_elr_accuracy", baseline_elr)
        mlflow.log_metric("baseline_overall_accuracy", baseline_overall)

        # Create datasets
        logger.info("🔨 Creating tokenized datasets...")
        train_dataset = create_dataset(train_data, tokenizer)
        val_dataset = create_dataset(val_data, tokenizer)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir / "model"),
            overwrite_output_dir=True,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            logging_steps=LOG_STEPS,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=2,
            bf16=USE_BF16,  # Use bf16 (model is in bfloat16)
            gradient_checkpointing=True,
            optim="adamw_torch",
            max_grad_norm=1.0,
            remove_unused_columns=False,
            report_to=[]  # Use only MLflow
        )
        
        # Initialize trainer
        logger.info("🏋️ Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[
                MLflowCallback(),
                ValidatorMetricsCallback(test_data, tokenizer)
            ]
        )
        
        # Training
        logger.info("🚀 Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

        # FINAL EVALUATION on test set after training
        logger.info("📊 Evaluating FINE-TUNED model on test set...")
        final_sat = callback._evaluate_sat_with_validator(model)
        final_elr = callback._evaluate_elr_with_validator(model)
        final_overall = (final_sat + final_elr) / 2 if final_elr > 0 else final_sat

        logger.info(f"FINE-TUNED - SAT: {final_sat:.3f}, ELR: {final_elr:.3f}, Overall: {final_overall:.3f}")
        mlflow.log_metric("final_sat_accuracy", final_sat)
        mlflow.log_metric("final_elr_accuracy", final_elr)
        mlflow.log_metric("final_overall_accuracy", final_overall)

        # Calculate improvements
        sat_improvement = final_sat - baseline_sat
        elr_improvement = final_elr - baseline_elr
        overall_improvement = final_overall - baseline_overall

        logger.info("=" * 50)
        logger.info("📈 TRAINING RESULTS COMPARISON")
        logger.info("=" * 50)
        logger.info(f"SAT:     {baseline_sat:.3f} -> {final_sat:.3f} (Δ {sat_improvement:+.3f})")
        logger.info(f"ELR:     {baseline_elr:.3f} -> {final_elr:.3f} (Δ {elr_improvement:+.3f})")
        logger.info(f"Overall: {baseline_overall:.3f} -> {final_overall:.3f} (Δ {overall_improvement:+.3f})")
        logger.info("=" * 50)

        mlflow.log_metric("sat_improvement", sat_improvement)
        mlflow.log_metric("elr_improvement", elr_improvement)
        mlflow.log_metric("overall_improvement", overall_improvement)

        # Log artifacts
        mlflow.log_artifacts(str(output_dir))

        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        mlflow.log_param("training_status", "failed")
        mlflow.log_param("error", str(e))
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    asyncio.run(main())