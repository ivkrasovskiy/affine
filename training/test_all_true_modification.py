#!/usr/bin/env python3
"""Test script to verify all-True modifications work correctly"""

import json
import sys
sys.path.append('.')

from train_with_validator_logic import force_all_true_solutions

# Load test dataset
print("📂 Loading test dataset...")
with open("./data/validator_dataset.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# Show original data
print("\n📊 BEFORE force_all_true_solutions:")
sample = data[0]
print(f"Environment: {sample['environment']}")
print(f"Original completion: {sample['completion']}")
print(f"Original solution: {sample['validator_solution']}")
print(f"N vars: {sample['n_vars']}")

# Apply transformation
print("\n🔄 Applying force_all_true_solutions...")
modified_data = force_all_true_solutions(data)

# Show modified data
print("\n📊 AFTER force_all_true_solutions:")
modified_sample = modified_data[0]
print(f"Modified completion: {modified_sample['completion']}")
print(f"Modified solution: {modified_sample['validator_solution']}")

# Verify format
print("\n✅ VERIFICATION:")
n_vars = modified_sample['n_vars']
expected_completion = ", ".join([f"x{k}=True" for k in range(1, n_vars + 1)])
expected_solution = {i: True for i in range(1, n_vars + 1)}

print(f"Expected completion matches: {modified_sample['completion'] == expected_completion}")
print(f"Expected solution matches: {modified_sample['validator_solution'] == expected_solution}")

# Count how many have been modified
sat_samples = [s for s in modified_data if s.get('environment') == 'SAT']
all_true_samples = sum(1 for s in sat_samples if all(s['validator_solution'].values()))
print(f"\nSAT samples with all-True solutions: {all_true_samples}/{len(sat_samples)} ({all_true_samples/len(sat_samples)*100:.1f}%)")

print("\n✅ Test complete!")
