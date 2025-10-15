#!/usr/bin/env python3
"""Test SAT validation logic standalone"""

import json
import sys
import re

def validate_sat_response(challenge_data: dict, response: str, debug: bool = True) -> bool:
    """Validate SAT response using EXACT validator evaluation logic"""
    try:
        solution = challenge_data["validator_solution"]
        clauses = challenge_data["validator_clauses"]

        if debug:
            print(f"\n=== VALIDATION DEBUG ===")
            print(f"Response: {response[:150]}...")
            print(f"Expected solution dict (first 5): {dict(list(solution.items())[:5])}")
            print(f"Number of clauses: {len(clauses)}")
            print(f"First clause: {clauses[0]}")

        # Parse response exactly like validator does
        matches = re.findall(r"x(\d+)=(True|False|1|0)", response)
        got = {int(v): val.lower() in ("true","1")
               for v, val in matches}

        if debug:
            print(f"\nRegex matches found: {len(matches)}")
            print(f"Parsed {len(got)} variables: {dict(list(got.items())[:5])}")

        # Check if assignment satisfies all clauses (validator logic)
        failed_clauses = []
        for i, c in enumerate(clauses):
            clause_result = any((lit>0)==got.get(abs(lit), None) for lit in c)
            if not clause_result and debug and i < 3:
                print(f"\nClause {i} FAILED: {c}")
                for lit in c:
                    var_val = got.get(abs(lit), "MISSING")
                    expected = (lit > 0)
                    print(f"  Literal {lit}: need {expected}, got {var_val}")
            if not clause_result:
                failed_clauses.append(i)

        ok = len(failed_clauses) == 0

        if debug:
            print(f"\nFailed clauses: {len(failed_clauses)} / {len(clauses)}")
            print(f"Validation result: {ok}")

        return ok

    except Exception as e:
        print(f"Error validating SAT response: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Load first sample from dataset
    with open('../data/validator_dataset.json', 'r') as f:
        data = json.load(f)

    sample = data[0]

    print("Testing with CORRECT completion from dataset:")
    print("=" * 60)
    result = validate_sat_response(sample, sample['completion'], debug=True)
    print(f"\n✓ CORRECT completion validated: {result}")

    print("\n\nTesting with INCORRECT completion (flipped x1):")
    print("=" * 60)
    wrong_completion = sample['completion'].replace("x1=True", "x1=False")
    result = validate_sat_response(sample, wrong_completion, debug=True)
    print(f"\n✗ WRONG completion validated: {result}")

    print("\n\nTesting with INCOMPLETE completion (only first 5 vars):")
    print("=" * 60)
    incomplete = ", ".join(sample['completion'].split(", ")[:5])
    result = validate_sat_response(sample, incomplete, debug=True)
    print(f"\n⚠ INCOMPLETE completion validated: {result}")
