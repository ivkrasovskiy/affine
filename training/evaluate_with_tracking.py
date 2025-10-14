#!/usr/bin/env python3
"""
Evaluation script with MLflow tracking
Compares baseline vs fine-tuned models on the 3 key metrics
"""

import asyncio
import logging
import json
import mlflow
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.asyncio import tqdm
import affine as af

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate models on Affine environments with tracking"""
    
    def __init__(self, model_path: str, model_name: str = "unknown"):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float32,
            device_map=None  # CPU for demo
        )
        
        logger.info(f"Model loaded: {self.model.num_parameters():,} parameters")
    
    async def query(self, prompt: str) -> str:
        """Generate response from model"""
        if self.model is None:
            self.load_model()
            
        try:
            # Format prompt
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""

async def evaluate_sat_accuracy(evaluator: ModelEvaluator, num_samples: int = 20) -> Dict[str, Any]:
    """Evaluate SAT reasoning accuracy"""
    logger.info(f"Evaluating SAT with {num_samples} samples...")
    
    sat_env = af.SAT(n=8, k=5)  # Smaller for faster evaluation
    
    correct = 0
    total = 0
    results = []
    
    for i in tqdm(range(num_samples), desc="SAT Eval"):
        try:
            # Generate challenge
            challenge = await sat_env.generate()
            
            # Get model response  
            response = await evaluator.query(challenge.prompt)
            
            # Evaluate (simplified - check format)
            is_correct = False
            if response and "x1=" in response:
                # Check if it looks like a valid SAT assignment
                if ("True" in response or "False" in response) and "," in response:
                    is_correct = True
            
            results.append({
                "sample_id": i,
                "prompt": challenge.prompt[:100] + "...",
                "response": response[:100] + "..." if len(response) > 100 else response,
                "correct": is_correct
            })
            
            if is_correct:
                correct += 1
            total += 1
            
        except Exception as e:
            logger.error(f"Error in SAT sample {i}: {e}")
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "sat_accuracy": accuracy,
        "sat_correct": correct,
        "sat_total": total,
        "sat_details": results[:5]  # Save first 5 for inspection
    }

async def evaluate_elr_accuracy(evaluator: ModelEvaluator, num_samples: int = 10) -> Dict[str, Any]:
    """Evaluate ELR mathematical reasoning accuracy"""
    logger.info(f"Evaluating ELR with {num_samples} samples...")
    
    elr_env = af.ELR()
    
    correct = 0
    total = 0
    results = []
    
    for i in tqdm(range(num_samples), desc="ELR Eval"):
        try:
            # Generate challenge
            challenge = await elr_env.generate()
            
            # Get model response
            response = await evaluator.query(challenge.prompt)
            
            # Evaluate (simplified - check format)
            is_correct = False
            if response and "<Answer>" in response and "</Answer>" in response:
                # Check if answer tags are present
                is_correct = True
            
            results.append({
                "sample_id": i,
                "prompt": challenge.prompt[:100] + "...",
                "response": response[:100] + "..." if len(response) > 100 else response,
                "correct": is_correct,
                "expected": challenge.extra.get("numerical_answer", "")
            })
            
            if is_correct:
                correct += 1
            total += 1
            
        except Exception as e:
            logger.error(f"Error in ELR sample {i}: {e}")
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "elr_accuracy": accuracy,
        "elr_correct": correct,
        "elr_total": total,
        "elr_details": results[:5]
    }

def calculate_training_metrics(log_dir: Path) -> Dict[str, float]:
    """Extract training metrics from logs"""
    
    # Look for trainer logs
    log_files = list(log_dir.glob("**/trainer_state.json"))
    
    if not log_files:
        logger.warning("No trainer logs found")
        return {}
    
    try:
        with open(log_files[0]) as f:
            trainer_state = json.load(f)
        
        # Extract final training loss
        log_history = trainer_state.get("log_history", [])
        
        if log_history:
            final_train_loss = None
            final_eval_loss = None
            
            for entry in reversed(log_history):
                if "train_loss" in entry and final_train_loss is None:
                    final_train_loss = entry["train_loss"]
                if "eval_loss" in entry and final_eval_loss is None:
                    final_eval_loss = entry["eval_loss"]
                    
                if final_train_loss is not None and final_eval_loss is not None:
                    break
            
            return {
                "final_train_loss": final_train_loss or 0.0,
                "final_eval_loss": final_eval_loss or 0.0,
                "total_training_steps": trainer_state.get("global_step", 0)
            }
    
    except Exception as e:
        logger.error(f"Error reading training logs: {e}")
    
    return {}

async def run_full_evaluation(model_path: str, model_name: str, experiment_name: str = "model_evaluation") -> Dict[str, Any]:
    """Run complete evaluation and log to MLflow"""
    
    # Setup MLflow
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"eval_{model_name}"):
        # Log model info
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("model_name", model_name)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path, model_name)
        
        # Run evaluations
        logger.info("Running evaluations...")
        
        sat_results = await evaluate_sat_accuracy(evaluator, num_samples=15)
        elr_results = await evaluate_elr_accuracy(evaluator, num_samples=8)
        
        # Calculate overall accuracy
        total_correct = sat_results["sat_correct"] + elr_results["elr_correct"]
        total_samples = sat_results["sat_total"] + elr_results["elr_total"]
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Extract training metrics if available
        training_metrics = {}
        model_dir = Path(model_path)
        if (model_dir / "logs").exists():
            training_metrics = calculate_training_metrics(model_dir / "logs")
        
        # Compile all metrics
        all_metrics = {
            **sat_results,
            **elr_results,
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_samples": total_samples,
            **training_metrics
        }
        
        # Log metrics to MLflow
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # Log detailed results as artifacts
        results_summary = {
            "model_name": model_name,
            "model_path": model_path,
            "metrics": all_metrics,
            "sat_examples": sat_results["sat_details"],
            "elr_examples": elr_results["elr_details"]
        }
        
        # Save results
        results_file = f"evaluation_{model_name}.json"
        with open(results_file, "w") as f:
            json.dump(results_summary, f, indent=2)
        
        mlflow.log_artifact(results_file)
        
        # Print summary
        logger.info("=" * 50)
        logger.info(f"EVALUATION RESULTS - {model_name}")
        logger.info("=" * 50)
        logger.info(f"SAT Accuracy:     {sat_results['sat_accuracy']:.3f} ({sat_results['sat_correct']}/{sat_results['sat_total']})")
        logger.info(f"ELR Accuracy:     {elr_results['elr_accuracy']:.3f} ({elr_results['elr_correct']}/{elr_results['elr_total']})")
        logger.info(f"Overall Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_samples})")
        
        if training_metrics:
            if "final_train_loss" in training_metrics:
                logger.info(f"Final Train Loss: {training_metrics['final_train_loss']:.3f}")
            if "final_eval_loss" in training_metrics:
                logger.info(f"Final Eval Loss:  {training_metrics['final_eval_loss']:.3f}")
        
        logger.info("=" * 50)
        
        return all_metrics

async def compare_models(baseline_path: str, finetuned_path: str):
    """Compare baseline vs fine-tuned model"""
    
    logger.info("🔍 COMPARING BASELINE VS FINE-TUNED MODEL")
    logger.info("=" * 60)
    
    # Evaluate baseline
    baseline_metrics = await run_full_evaluation(
        baseline_path, "baseline", "model_comparison"
    )
    
    # Evaluate fine-tuned
    finetuned_metrics = await run_full_evaluation(
        finetuned_path, "finetuned", "model_comparison"
    )
    
    # Calculate improvements
    logger.info("\n📈 IMPROVEMENT ANALYSIS")
    logger.info("-" * 40)
    
    sat_improvement = finetuned_metrics["sat_accuracy"] - baseline_metrics["sat_accuracy"]
    elr_improvement = finetuned_metrics["elr_accuracy"] - baseline_metrics["elr_accuracy"]
    overall_improvement = finetuned_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"]
    
    logger.info(f"SAT Improvement:     {sat_improvement:+.3f}")
    logger.info(f"ELR Improvement:     {elr_improvement:+.3f}")
    logger.info(f"Overall Improvement: {overall_improvement:+.3f}")
    
    # Log comparison metrics
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_metric("sat_improvement", sat_improvement)
        mlflow.log_metric("elr_improvement", elr_improvement)
        mlflow.log_metric("overall_improvement", overall_improvement)
        
        comparison_summary = {
            "baseline": baseline_metrics,
            "finetuned": finetuned_metrics,
            "improvements": {
                "sat": sat_improvement,
                "elr": elr_improvement,
                "overall": overall_improvement
            }
        }
        
        with open("model_comparison.json", "w") as f:
            json.dump(comparison_summary, f, indent=2)
        
        mlflow.log_artifact("model_comparison.json")

async def main():
    """Main evaluation function"""
    
    # For testing, evaluate the simple model we trained
    test_model_path = "test_model"
    
    if Path(test_model_path).exists():
        logger.info("Evaluating test model...")
        await run_full_evaluation(test_model_path, "test_model", "test_evaluation")
    else:
        logger.warning(f"Test model not found at {test_model_path}")
        logger.info("Train a model first with: python train_tracked.py")
    
    logger.info("\n🎯 To view results:")
    logger.info("mlflow ui --backend-store-uri ./mlruns")
    logger.info("Then open: http://localhost:5000")

if __name__ == "__main__":
    asyncio.run(main())