#!/usr/bin/env python3
"""Quick test: Check 3 samples and show actual model outputs"""

import json
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from generate_validator_data import ValidatorDataGenerator

# Load model
print("Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "Marco0/Affine-QQ",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Marco0/Affine-QQ", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

print("✓ Model loaded\n")

# Load data
with open('../data/validator_dataset.json', 'r') as f:
    data = json.load(f)

generator = ValidatorDataGenerator()

# Test 3 samples
print("Testing 3 samples:")
print("="*70)

correct = 0
for i in range(3):
    sample = data[i]
    prompt = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"

    print(f"\nSample {i+1}: n={sample['n_vars']}, k={sample['k_vars_per_clause']}")
    print(f"Expected: {sample['completion'][:80]}...")

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True,
                                pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f"Generated: {response[:80]}...")

    # Validate
    is_correct = generator.validate_sat_response(sample, response)
    print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

    if is_correct:
        correct += 1

print(f"\n{'='*70}")
print(f"Accuracy: {correct}/3 = {correct/3*100:.1f}%")
print(f"\n{'✓ SUCCESS: Evaluation works!' if correct > 0 else '⚠ Model needs training (0% is expected for base model)'}")
