#!/usr/bin/env python3
"""Diagnose the zero loss bug"""

import json
import torch
from transformers import AutoTokenizer

# Load the model tokenizer
MODEL_NAME = "Marco0/Affine-QQ"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load training data
with open('./experiments/validator_training/train_data.json', 'r') as f:
    train_data = json.load(f)

print(f"📊 Training data: {len(train_data)} samples\n")

# Check first few samples
print("="*60)
print("CHECKING COMPLETION DIVERSITY")
print("="*60)
completions = [s['completion'] for s in train_data[:10] if s.get('environment') == 'SAT']
unique_completions = set(completions)
print(f"First 10 SAT samples have {len(unique_completions)} unique completions")
if len(unique_completions) <= 3:
    for i, comp in enumerate(unique_completions):
        print(f"  {i+1}. {comp}")

# Check completion format
print("\n" + "="*60)
print("SAMPLE FORMATTING")
print("="*60)
sample = train_data[0]
prompt = sample['prompt']
completion = sample['completion']

formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
full_text = formatted_prompt + completion

print(f"Prompt length: {len(prompt)}")
print(f"Completion length: {len(completion)}")
print(f"Completion: {completion}")
print(f"\nFull formatted text (first 200 chars):")
print(full_text[:200])
print(f"\nFull formatted text (last 200 chars):")
print(full_text[-200:])

# Tokenize and check labels
print("\n" + "="*60)
print("TOKENIZATION CHECK")
print("="*60)

# Tokenize full text
targets = tokenizer(full_text, truncation=True, max_length=4096, return_tensors=None)
print(f"Total tokens: {len(targets['input_ids'])}")

# Tokenize just prompt
prompts = tokenizer(formatted_prompt, truncation=True, max_length=4096, return_tensors=None)
print(f"Prompt tokens: {len(prompts['input_ids'])}")

# Create labels (mask prompt)
labels = targets["input_ids"].copy()
prompt_len = len(prompts["input_ids"])

for i in range(min(prompt_len, len(labels))):
    labels[i] = -100

response_tokens = sum(1 for x in labels if x != -100)
masked_tokens = sum(1 for x in labels if x == -100)

print(f"Masked (prompt) tokens: {masked_tokens}")
print(f"Training (response) tokens: {response_tokens}")
print(f"\nResponse token IDs: {[t for t in labels if t != -100][:20]}")

# Decode response tokens
response_token_ids = [t for t in labels if t != -100]
decoded_response = tokenizer.decode(response_token_ids, skip_special_tokens=True)
print(f"\nDecoded response: {decoded_response}")

# Check if all responses are identical
print("\n" + "="*60)
print("CHECKING ALL TRAINING SAMPLES")
print("="*60)
all_response_tokens = []
for i, sample in enumerate(train_data[:100]):  # Check first 100
    if sample.get('environment') != 'SAT':
        continue

    formatted = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n{sample['completion']}"
    tokens = tokenizer(formatted, truncation=True, max_length=4096, return_tensors=None)
    prompt_tokens = tokenizer(f"### Instruction:\n{sample['prompt']}\n\n### Response:\n",
                              truncation=True, max_length=4096, return_tensors=None)

    response_start = len(prompt_tokens['input_ids'])
    response_token_ids = tuple(tokens['input_ids'][response_start:])
    all_response_tokens.append(response_token_ids)

unique_responses = set(all_response_tokens)
print(f"First 100 SAT samples have {len(unique_responses)} unique token sequences")
print(f"Most common length: {max(set([len(r) for r in all_response_tokens]),
                                   key=[len(r) for r in all_response_tokens].count)}")

if len(unique_responses) == 1:
    print("\n⚠️  WARNING: ALL RESPONSES ARE IDENTICAL!")
    print("This explains the zero loss - model is just learning to always output the same tokens.")
