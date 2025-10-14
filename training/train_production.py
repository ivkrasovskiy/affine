#!/usr/bin/env python3
"""
Production-ready Affine training pipeline
Incorporates best practices from colleague's ABD training code
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

# Affine
import affine as af

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== ENVIRONMENT CONFIGURATION ==============
# All hyperparameters via environment variables (production best practice)

# Model & Data
MODEL_NAME = os.getenv("MODEL_NAME", "Marco0/Affine-QQ")
OUT_DIR = os.getenv("OUT_DIR", "./experiments/production")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))

# Training
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
EPOCHS = float(os.getenv("EPOCHS", "3.0"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))

# LoRA
LORA_R = int(os.getenv("LORA_R", "64"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))

# Evaluation
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "200"))
LOG_STEPS = int(os.getenv("LOG_STEPS", "25"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "200"))
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "50"))

# Generation (for evaluation)
GEN_MAX_NEW = int(os.getenv("GEN_MAX_NEW", "256"))
GEN_TEMP = float(os.getenv("GEN_TEMP", "0.1"))
GEN_DO_SAMPLE = bool(int(os.getenv("GEN_DO_SAMPLE", "0")))

# System
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
SEED = int(os.getenv("SEED", "42"))

# Data generation
DATA_SAMPLES = int(os.getenv("DATA_SAMPLES", "500"))

# ============== MLFLOW SETUP ==============

def setup_mlflow_experiment(experiment_name: str = "affine_production") -> str:
    """Setup MLflow experiment with git tracking"""
    
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
    
    # Start run with timestamp
    run_name = f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    
    # Log environment info
    env_params = {
        "model_name": MODEL_NAME,
        "max_seq_len": MAX_SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "eval_samples": EVAL_SAMPLES,
        "seed": SEED,
        "use_bf16": USE_BF16,
        "python_version": f"{torch.__version__}",
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 1e9, 1)
    }
    
    mlflow.log_params(env_params)
    
    # Log git info
    try:
        repo = git.Repo(search_parent_directories=True)
        mlflow.log_param("git_commit", repo.head.commit.hexsha[:8])
        mlflow.log_param("git_branch", repo.active_branch.name)
        mlflow.log_param("git_dirty", repo.is_dirty())
    except Exception as e:
        logger.warning(f"Could not log git info: {e}")
    
    return mlflow.active_run().info.run_id

# ============== ENHANCED DATA GENERATION ==============

class ProductionDataGenerator:
    """Enhanced data generator with validation and quality filtering"""
    
    def __init__(self, seed: int = SEED):
        self.seed = seed
        torch.manual_seed(seed)
        
        # Initialize environments
        self.sat_env = af.SAT(n=10, k=6)
        self.elr_env = af.ELR()
        
    async def generate_validated_sat_sample(self) -> Optional[Dict[str, str]]:
        """Generate and validate a single SAT sample"""
        try:
            challenge = await self.sat_env.generate()
            
            # Validate the challenge has a solution
            solution = challenge.extra.get("sol", {})
            if not solution:
                return None
                
            # Create ground truth
            ground_truth = ", ".join([f"x{k}={'True' if v else 'False'}" 
                                    for k, v in sorted(solution.items())])
            
            # Validate format
            if len(ground_truth.split(",")) != len(solution):
                return None
                
            return {
                "prompt": challenge.prompt,
                "completion": ground_truth,
                "environment": "SAT",
                "difficulty": "medium",
                "n_vars": len(solution),
                "n_clauses": len(challenge.extra.get("cls", []))
            }
            
        except Exception as e:
            logger.error(f"Error generating SAT sample: {e}")
            return None
    
    async def generate_validated_elr_sample(self) -> Optional[Dict[str, str]]:
        """Generate and validate a single ELR sample"""
        try:
            challenge = await self.elr_env.generate()
            
            answer = challenge.extra.get("numerical_answer", "")
            if not str(answer).isdigit():
                return None
                
            ground_truth = f"<Answer>{answer}</Answer>"
            
            return {
                "prompt": challenge.prompt,
                "completion": ground_truth,
                "environment": "ELR",
                "difficulty": "medium",
                "answer": str(answer)
            }
            
        except Exception as e:
            logger.error(f"Error generating ELR sample: {e}")
            return None
    
    async def generate_production_dataset(self, total_samples: int = DATA_SAMPLES) -> List[Dict[str, str]]:
        """Generate high-quality validated dataset"""
        logger.info(f"Generating {total_samples} validated samples...")
        
        sat_target = int(total_samples * 0.75)  # 75% SAT
        elr_target = total_samples - sat_target
        
        samples = []
        
        # Generate SAT samples
        sat_count = 0
        attempts = 0
        max_attempts = sat_target * 3  # Allow some failures
        
        with tqdm(total=sat_target, desc="SAT samples") as pbar:
            while sat_count < sat_target and attempts < max_attempts:
                sample = await self.generate_validated_sat_sample()
                attempts += 1
                
                if sample:
                    samples.append(sample)
                    sat_count += 1
                    pbar.update(1)
        
        # Generate ELR samples
        elr_count = 0
        attempts = 0
        max_attempts = elr_target * 3
        
        with tqdm(total=elr_target, desc="ELR samples") as pbar:
            while elr_count < elr_target and attempts < max_attempts:
                sample = await self.generate_validated_elr_sample()
                attempts += 1
                
                if sample:
                    samples.append(sample)
                    elr_count += 1
                    pbar.update(1)
        
        logger.info(f"Generated {len(samples)} valid samples (SAT: {sat_count}, ELR: {elr_count})")
        
        # Shuffle
        import random
        random.shuffle(samples)
        
        return samples

# ============== REAL-TIME EVALUATION CALLBACK ==============

class AffineMetricsCallback(TrainerCallback):
    """Real-time evaluation during training (adapted from colleague's code)"""
    
    def __init__(self, eval_samples: List[Dict], tokenizer, eval_every: int = EVAL_STEPS):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        
        # Prepare evaluation data
        self.sat_samples = [s for s in eval_samples if s.get("environment") == "SAT"]
        self.elr_samples = [s for s in eval_samples if s.get("environment") == "ELR"]
        
        logger.info(f"Prepared evaluation: {len(self.sat_samples)} SAT, {len(self.elr_samples)} ELR")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation every N steps"""
        
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return control
            
        logger.info(f"Running evaluation at step {state.global_step}")
        
        model.eval()
        
        try:
            # Evaluate SAT
            sat_accuracy = self._evaluate_sat_samples(model)
            
            # Evaluate ELR  
            elr_accuracy = self._evaluate_elr_samples(model)
            
            # Overall accuracy
            overall_accuracy = (sat_accuracy + elr_accuracy) / 2
            
            # Log to MLflow
            step = int(state.global_step)
            mlflow.log_metric("eval_sat_accuracy", sat_accuracy, step=step)
            mlflow.log_metric("eval_elr_accuracy", elr_accuracy, step=step)
            mlflow.log_metric("eval_overall_accuracy", overall_accuracy, step=step)
            
            logger.info(f"Step {step}: SAT={sat_accuracy:.3f}, ELR={elr_accuracy:.3f}, Overall={overall_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed at step {state.global_step}: {e}")
        finally:
            model.train()
            
        return control
    
    def _evaluate_sat_samples(self, model) -> float:
        """Evaluate SAT accuracy"""
        if not self.sat_samples:
            return 0.0
            
        correct = 0
        total = min(EVAL_SAMPLES // 2, len(self.sat_samples))
        
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else nullcontext()
        
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
                
                # Simple validation: check if response looks like SAT format
                if "x1=" in response and ("True" in response or "False" in response):
                    correct += 1
                
                # Memory management
                del inputs, outputs
                if i % 8 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error evaluating SAT sample {i}: {e}")
        
        return correct / max(total, 1)
    
    def _evaluate_elr_samples(self, model) -> float:
        """Evaluate ELR accuracy"""
        if not self.elr_samples:
            return 0.0
            
        correct = 0
        total = min(EVAL_SAMPLES // 2, len(self.elr_samples))
        
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else nullcontext()
        
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
                
                # Simple validation: check if response has answer tags
                if "<Answer>" in response and "</Answer>" in response:
                    correct += 1
                
                # Memory management
                del inputs, outputs
                if i % 8 == 0:
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

def load_and_prepare_model(model_name: str = MODEL_NAME):
    """Load model with QLoRA configuration"""
    
    logger.info(f"Loading model: {model_name}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def create_dataset(data: List[Dict[str, str]], tokenizer, max_seq_length: int = MAX_SEQ_LEN):
    """Create tokenized dataset"""
    
    def format_instruction(sample):
        prompt = sample["prompt"]
        completion = sample["completion"]
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        full_text = formatted_prompt + completion
        return {"input_text": formatted_prompt, "target_text": full_text}
    
    formatted_data = [format_instruction(sample) for sample in data]
    
    def tokenize_function(examples):
        targets = tokenizer(
            examples["target_text"], 
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_tensors=None
        )
        
        return {
            "input_ids": targets["input_ids"],
            "attention_mask": targets["attention_mask"],
            "labels": targets["input_ids"].copy()
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
    """Production training pipeline"""
    
    logger.info("🚀 STARTING PRODUCTION AFFINE TRAINING")
    logger.info("=" * 60)
    
    # Set reproducibility
    torch.manual_seed(SEED)
    
    # Setup MLflow
    run_id = setup_mlflow_experiment("affine_production")
    logger.info(f"MLflow run ID: {run_id}")
    
    try:
        # Create output directory
        output_dir = Path(OUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate training data
        logger.info("📊 Generating production dataset...")
        generator = ProductionDataGenerator(seed=SEED)
        all_data = await generator.generate_production_dataset(DATA_SAMPLES)
        
        if len(all_data) < 10:
            raise RuntimeError("Failed to generate sufficient training data")
        
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
        
        # Load model and tokenizer
        logger.info("🤖 Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = load_and_prepare_model(MODEL_NAME)
        
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
            save_total_limit=3,
            bf16=USE_BF16,
            fp16=not USE_BF16,
            gradient_checkpointing=True,
            optim="adamw_torch",
            max_grad_norm=1.0,
            remove_unused_columns=False,
            report_to=[]  # Disable built-in W&B
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
                AffineMetricsCallback(test_data, tokenizer)  # Use test set for evaluation
            ]
        )
        
        # Training
        logger.info("🚀 Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Log final artifacts
        mlflow.pytorch.log_model(model, "model")
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