#!/usr/bin/env python3
"""
Evaluate model on full dataset with comprehensive metrics.

This script calculates:
1. Model performance (Format, All-True, SAT accuracy) on train/test/val/full dataset
2. Dataset solvability: What % of problems are satisfiable by all-True solution
3. Model failures vs. theoretical maximum

Run after training completes to get comprehensive statistics.
"""

import os
import json
import re
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MAX_SEQ_LEN = 4096
MAX_NEW_TOKENS = 256


def load_dataset(data_path: str) -> List[Dict]:
    """Load the dataset from JSON"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {data_path}")
    return data


def load_model_and_tokenizer(model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load the trained model and tokenizer"""
    logger.info(f"Loading base model: {base_model}")

    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapters if path provided
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        logger.warning(f"Model path not found: {model_path}. Using base model only.")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def check_dataset_solvability(sample: Dict) -> bool:
    """
    Check if a SAT problem is satisfiable by setting all variables to True.

    Returns:
        True if all-True solution satisfies all clauses, False otherwise
    """
    if sample.get("environment") != "SAT":
        return False

    clauses = sample.get("validator_clauses", [])
    n_vars = sample.get("n_vars", 0)

    if not clauses or not n_vars:
        return False

    # Create all-True assignment
    all_true = {i: True for i in range(1, n_vars + 1)}

    # Check if all clauses are satisfied
    # A clause is satisfied if at least one literal is true
    # Literal > 0 means variable should be True
    # Literal < 0 means variable should be False
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var_idx = abs(lit)
            var_value = all_true.get(var_idx, False)
            # If lit > 0, we need var to be True
            # If lit < 0, we need var to be False
            if (lit > 0) == var_value:
                clause_satisfied = True
                break

        if not clause_satisfied:
            return False

    return True


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate model response for a given prompt"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated tokens (exclude prompt)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return ""


def validate_sat_response(sample: Dict, response: str) -> Tuple[bool, bool, bool]:
    """
    Validate SAT response using exact validator logic.

    Returns:
        (has_valid_format, all_true, sat_correct): tuple of bools
    """
    try:
        clauses = sample.get("validator_clauses", [])
        n_vars = sample.get("n_vars", 0)

        if not clauses or not n_vars:
            return False, False, False

        # Parse response (same as validator logic)
        matches = re.findall(r"x(\d+)=(True|False|1|0)", response)
        parsed_vars = {int(v): val.lower() in ("true", "1") for v, val in matches}

        # Check format: all required variables present (x1 through xN)
        required_vars = set(range(1, n_vars + 1))
        found_vars = set(parsed_vars.keys())
        has_all_required = required_vars.issubset(found_vars)
        has_valid_format = has_all_required and len(found_vars) >= n_vars

        # Check if all required variables are True
        required_values = [parsed_vars.get(i, False) for i in required_vars]
        all_true = all(required_values) if required_values and has_valid_format else False

        # Check if assignment satisfies all clauses (validator logic)
        sat_correct = all(
            any((lit > 0) == parsed_vars.get(abs(lit), None) for lit in c)
            for c in clauses
        )

        return has_valid_format, all_true, sat_correct

    except Exception as e:
        logger.error(f"Error validating response: {e}")
        return False, False, False


def evaluate_dataset(model, tokenizer, samples: List[Dict], dataset_name: str) -> Dict:
    """
    Evaluate model on a dataset split.

    Returns:
        Dictionary with metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {dataset_name} ({len(samples)} samples)")
    logger.info(f"{'='*60}")

    # Dataset solvability metrics
    solvable_count = 0

    # Model performance metrics
    format_correct = 0
    all_true_correct = 0
    sat_correct = 0

    for i, sample in enumerate(samples):
        if sample.get("environment") != "SAT":
            continue

        # Check dataset solvability
        is_solvable = check_dataset_solvability(sample)
        if is_solvable:
            solvable_count += 1

        # Generate model response
        prompt = sample["prompt"]
        response = generate_response(model, tokenizer, prompt)

        # Validate response
        has_format, has_all_true, has_sat = validate_sat_response(sample, response)

        if has_format:
            format_correct += 1
        if has_all_true:
            all_true_correct += 1
        if has_sat:
            sat_correct += 1

        # Log progress every 10%
        if (i + 1) % max(1, len(samples) // 10) == 0:
            progress = (i + 1) / len(samples) * 100
            logger.info(f"  Progress: {progress:.1f}% ({i+1}/{len(samples)})")

    total = len(samples)

    results = {
        "dataset_name": dataset_name,
        "total_samples": total,
        "solvable_by_all_true": solvable_count,
        "solvable_pct": (solvable_count / total * 100) if total > 0 else 0,
        "format_correct": format_correct,
        "format_accuracy": (format_correct / total * 100) if total > 0 else 0,
        "all_true_correct": all_true_correct,
        "all_true_accuracy": (all_true_correct / total * 100) if total > 0 else 0,
        "sat_correct": sat_correct,
        "sat_accuracy": (sat_correct / total * 100) if total > 0 else 0,
        # Model performance vs. theoretical maximum
        "format_vs_max": (format_correct / solvable_count * 100) if solvable_count > 0 else 0,
        "all_true_vs_max": (all_true_correct / solvable_count * 100) if solvable_count > 0 else 0,
        "sat_vs_max": (sat_correct / solvable_count * 100) if solvable_count > 0 else 0,
    }

    return results


def print_results(results: Dict):
    """Print evaluation results in a nice format"""
    print(f"\n{'='*60}")
    print(f"Results for: {results['dataset_name']}")
    print(f"{'='*60}")
    print(f"Total samples: {results['total_samples']}")
    print(f"\nDataset Solvability:")
    print(f"  Solvable by all-True: {results['solvable_by_all_true']} ({results['solvable_pct']:.2f}%)")
    print(f"\nModel Performance (Absolute):")
    print(f"  Format accuracy:   {results['format_correct']:4d} / {results['total_samples']} ({results['format_accuracy']:.2f}%)")
    print(f"  All-True accuracy: {results['all_true_correct']:4d} / {results['total_samples']} ({results['all_true_accuracy']:.2f}%)")
    print(f"  SAT accuracy:      {results['sat_correct']:4d} / {results['total_samples']} ({results['sat_accuracy']:.2f}%)")
    print(f"\nModel Performance vs. Theoretical Maximum (solvable problems only):")
    print(f"  Format vs Max:   {results['format_vs_max']:.2f}%")
    print(f"  All-True vs Max: {results['all_true_vs_max']:.2f}%")
    print(f"  SAT vs Max:      {results['sat_vs_max']:.2f}%")
    print(f"{'='*60}\n")


def main():
    """Main evaluation script"""

    # Configuration from environment variables
    DATA_PATH = os.getenv("DATA_PATH", "../data/validator_dataset.json")
    MODEL_PATH = os.getenv("MODEL_PATH", "./experiments/validator_training")
    BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./evaluation_results.json")

    # Split ratios (should match training script)
    TEST_SPLIT = 0.15
    VAL_SPLIT = 0.15

    logger.info("="*60)
    logger.info("FULL DATASET EVALUATION")
    logger.info("="*60)
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"Output path: {OUTPUT_PATH}")

    # Load dataset
    data = load_dataset(DATA_PATH)
    sat_data = [s for s in data if s.get("environment") == "SAT"]

    # Split data (same logic as training script)
    total = len(sat_data)
    test_size = int(total * TEST_SPLIT)
    val_size = int(total * VAL_SPLIT)
    train_size = total - test_size - val_size

    train_data = sat_data[:train_size]
    test_data = sat_data[train_size:train_size + test_size]
    val_data = sat_data[train_size + test_size:]

    logger.info(f"\nDataset splits:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Test:  {len(test_data)} samples")
    logger.info(f"  Val:   {len(val_data)} samples")
    logger.info(f"  Total: {len(sat_data)} samples")

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, BASE_MODEL)

    # Evaluate on each split
    all_results = {}

    # Train set
    train_results = evaluate_dataset(model, tokenizer, train_data, "Train")
    print_results(train_results)
    all_results["train"] = train_results

    # Test set
    test_results = evaluate_dataset(model, tokenizer, test_data, "Test")
    print_results(test_results)
    all_results["test"] = test_results

    # Val set
    val_results = evaluate_dataset(model, tokenizer, val_data, "Val")
    print_results(val_results)
    all_results["val"] = val_results

    # Full dataset
    full_results = evaluate_dataset(model, tokenizer, sat_data, "Full Dataset")
    print_results(full_results)
    all_results["full"] = full_results

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n✅ Evaluation complete!")
    logger.info(f"📁 Results saved to: {OUTPUT_PATH}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset solvability: {full_results['solvable_pct']:.2f}% problems solvable by all-True")
    print(f"\nModel performance on full dataset:")
    print(f"  Format:   {full_results['format_accuracy']:.2f}%")
    print(f"  All-True: {full_results['all_true_accuracy']:.2f}%")
    print(f"  SAT:      {full_results['sat_accuracy']:.2f}%")
    print(f"\nModel vs. Theoretical Maximum:")
    print(f"  Format:   {full_results['format_vs_max']:.2f}% of solvable problems")
    print(f"  All-True: {full_results['all_true_vs_max']:.2f}% of solvable problems")
    print(f"  SAT:      {full_results['sat_vs_max']:.2f}% of solvable problems")
    print("="*60)


if __name__ == "__main__":
    main()
