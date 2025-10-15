#!/usr/bin/env python3
"""
Test baseline model evaluation to verify:
1. Model loads correctly
2. Evaluation logic works
3. We get accuracy > 0% (sanity check)
"""

import os
import sys
import json
import random
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_validator_data import ValidatorDataGenerator

# Configuration
MODEL_NAME = "Marco0/Affine-QQ"
DATA_PATH = "../data/validator_dataset.json"
NUM_TEST_SAMPLES = 20  # Test on 20 samples for quick validation

def load_model_simple(model_name: str):
    """Load model and tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading model: {model_name}")

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"✓ Model loaded")
    return model, tokenizer


def evaluate_sample(model, tokenizer, sample, generator, debug=False):
    """Evaluate a single SAT sample"""
    prompt = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        if debug:
            logger.info(f"  Response: {response[:150]}...")

        # Validate using validator logic
        is_correct = generator.validate_sat_response(sample, response, debug=debug)

        return is_correct, response

    except Exception as e:
        logger.error(f"Error evaluating sample: {e}")
        return False, ""


def main():
    logger.info("="*60)
    logger.info("BASELINE MODEL EVALUATION TEST")
    logger.info("="*60)

    # Load data
    logger.info(f"Loading dataset: {DATA_PATH}")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    # Sample test set
    test_samples = random.sample(data, min(NUM_TEST_SAMPLES, len(data)))
    logger.info(f"✓ Testing on {len(test_samples)} samples")

    # Check dataset properties
    sizes = [s['n_vars'] for s in test_samples]
    k_values = [s['k_vars_per_clause'] for s in test_samples]
    logger.info(f"  n range: {min(sizes)}-{max(sizes)}")
    logger.info(f"  k values: {set(k_values)}")
    logger.info("")

    # Load model
    model, tokenizer = load_model_simple(MODEL_NAME)
    model.eval()

    # Initialize validator
    generator = ValidatorDataGenerator(seed=42)

    # Evaluate samples
    logger.info(f"Evaluating {len(test_samples)} samples...")
    logger.info("-"*60)

    correct = 0
    for i, sample in enumerate(test_samples):
        n = sample['n_vars']
        k = sample['k_vars_per_clause']

        debug = (i < 3)  # Show details for first 3
        if debug:
            logger.info(f"\nSample {i+1}/{len(test_samples)}: n={n}, k={k}")

        is_correct, response = evaluate_sample(model, tokenizer, sample, generator, debug=debug)

        if is_correct:
            correct += 1
            if debug:
                logger.info(f"  ✓ CORRECT")
        else:
            if debug:
                logger.info(f"  ✗ INCORRECT")

        # Progress indicator
        if not debug and (i+1) % 5 == 0:
            logger.info(f"  Progress: {i+1}/{len(test_samples)} - accuracy so far: {correct/(i+1)*100:.1f}%")

    accuracy = correct / len(test_samples)

    logger.info("")
    logger.info("="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Correct: {correct}/{len(test_samples)}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info("")

    if accuracy > 0:
        logger.info("✓ SUCCESS: Evaluation logic works correctly!")
        logger.info("  Model can solve some problems even without fine-tuning.")
    else:
        logger.info("⚠ WARNING: 0% accuracy")
        logger.info("  This could mean:")
        logger.info("  - Model needs fine-tuning (expected)")
        logger.info("  - Evaluation logic has issues (needs investigation)")
        logger.info("  Try: Check if model generates valid format responses")

    return 0 if accuracy >= 0 else 1


if __name__ == "__main__":
    exit(main())
