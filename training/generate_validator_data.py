#!/usr/bin/env python3
"""
Data generation using EXACT validator logic
Reuses the SAT and ELR generation from affine validators
"""

import os
import json
import asyncio
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm.asyncio import tqdm
import affine as af

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidatorDataGenerator:
    """Generate training data using exact validator logic"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Import the actual environment classes
        from affine.envs.sat import SAT
        from affine.envs.elr import ELR
        
        # Initialize environments with SAME parameters as validators
        self.sat_env = SAT(n=15, k=10, m=None)  # Default validator params
        self.elr_env = ELR()
        
        logger.info(f"Initialized SAT env: n={self.sat_env.n}, k={self.sat_env.k}, m={self.sat_env.m}")
    
    async def generate_sat_sample(self) -> Optional[Dict[str, Any]]:
        """Generate SAT sample using EXACT validator logic"""
        try:
            # Use the validator's generate method directly
            challenge = await self.sat_env.generate()
            
            # Extract the known solution from validator
            solution = challenge.extra["sol"]
            clauses = challenge.extra["cls"]
            
            # Create ground truth in the EXACT format validators expect
            ground_truth = ", ".join([f"x{k}={'True' if v else 'False'}" 
                                    for k, v in sorted(solution.items())])
            
            return {
                "prompt": challenge.prompt,
                "completion": ground_truth,
                "environment": "SAT",
                "validator_solution": solution,
                "validator_clauses": clauses,
                "n_vars": self.sat_env.n,
                "k_vars_per_clause": self.sat_env.k,
                "n_clauses": self.sat_env.m
            }
            
        except Exception as e:
            logger.error(f"Error generating SAT sample: {e}")
            return None
    
    async def generate_elr_sample(self) -> Optional[Dict[str, Any]]:
        """Generate ELR sample using EXACT validator logic"""
        try:
            # Use the validator's generate method directly
            challenge = await self.elr_env.generate()
            
            # Extract answer in validator format
            answer = challenge.extra.get("numerical_answer", "")
            if not answer:
                return None
                
            # Ground truth in EXACT format validators expect
            ground_truth = f"<Answer>{answer}</Answer>"
            
            return {
                "prompt": challenge.prompt,
                "completion": ground_truth,
                "environment": "ELR",
                "numerical_answer": answer,
                "problem_type": "mathematical"
            }
            
        except Exception as e:
            logger.error(f"Error generating ELR sample: {e}")
            return None
    
    def validate_sat_response(self, challenge_data: Dict, response: str) -> bool:
        """Validate SAT response using EXACT validator evaluation logic"""
        try:
            # Recreate the validation logic from sat.py line 30-34
            import re
            
            solution = challenge_data["validator_solution"]
            clauses = challenge_data["validator_clauses"]
            
            # Parse response exactly like validator does
            got = {int(v): val.lower() in ("true","1")
                   for v, val in re.findall(r"x(\d+)=(True|False|1|0)", response)}
            
            # Check if assignment satisfies all clauses (validator logic)
            ok = all(any((lit>0)==got.get(abs(lit), None) for lit in c) for c in clauses)
            
            return ok
            
        except Exception as e:
            logger.error(f"Error validating SAT response: {e}")
            return False
    
    def validate_elr_response(self, challenge_data: Dict, response: str) -> bool:
        """Validate ELR response using validator format"""
        try:
            expected_answer = str(challenge_data["numerical_answer"])
            
            # Extract answer from tags
            import re
            matches = re.findall(r"<Answer>(.*?)</Answer>", response, re.IGNORECASE)
            if not matches:
                return False
                
            got_answer = matches[0].strip()
            return got_answer == expected_answer
            
        except Exception as e:
            logger.error(f"Error validating ELR response: {e}")
            return False
    
    async def generate_dataset(self, 
                             total_samples: int = 500,
                             sat_ratio: float = 0.75) -> List[Dict[str, Any]]:
        """Generate dataset using validator logic"""
        
        sat_target = int(total_samples * sat_ratio)
        elr_target = total_samples - sat_target
        
        logger.info(f"Generating {total_samples} samples (SAT: {sat_target}, ELR: {elr_target})")
        
        samples = []
        
        # Generate SAT samples
        logger.info("Generating SAT samples...")
        sat_progress = tqdm(total=sat_target, desc="SAT")
        sat_count = 0
        
        while sat_count < sat_target:
            sample = await self.generate_sat_sample()
            if sample:
                # Validate the sample works
                if self.validate_sat_response(sample, sample["completion"]):
                    samples.append(sample)
                    sat_count += 1
                    sat_progress.update(1)
                else:
                    logger.warning("Generated SAT sample failed validation")
        
        sat_progress.close()
        
        # Generate ELR samples only if needed
        if elr_target > 0:
            logger.info("Generating ELR samples...")
            elr_progress = tqdm(total=elr_target, desc="ELR")
            elr_count = 0
            
            while elr_count < elr_target:
                sample = await self.generate_elr_sample()
                if sample:
                    # Validate the sample works
                    if self.validate_elr_response(sample, sample["completion"]):
                        samples.append(sample)
                        elr_count += 1
                        elr_progress.update(1)
                    else:
                        logger.warning("Generated ELR sample failed validation")
            
            elr_progress.close()
        else:
            logger.info("Skipping ELR samples (SAT_RATIO=1.0)")
        
        # Shuffle to mix environments
        random.shuffle(samples)
        
        logger.info(f"Successfully generated {len(samples)} validated samples")
        return samples

def save_dataset(samples: List[Dict], output_dir: str = "./data"):
    """Save dataset in multiple formats"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save full dataset
    with open(output_path / "validator_dataset.json", "w") as f:
        json.dump(samples, f, indent=2)
    
    # Save training format (prompt/completion only)
    training_samples = []
    for sample in samples:
        training_samples.append({
            "prompt": sample["prompt"],
            "completion": sample["completion"],
            "environment": sample["environment"]
        })
    
    with open(output_path / "training_dataset.json", "w") as f:
        json.dump(training_samples, f, indent=2)
    
    # Save statistics
    stats = {
        "total_samples": len(samples),
        "sat_samples": len([s for s in samples if s["environment"] == "SAT"]),
        "elr_samples": len([s for s in samples if s["environment"] == "ELR"]),
        "sat_parameters": {
            "n_vars": samples[0]["n_vars"] if samples and samples[0]["environment"] == "SAT" else None,
            "k_vars_per_clause": samples[0]["k_vars_per_clause"] if samples and samples[0]["environment"] == "SAT" else None,
            "n_clauses": samples[0]["n_clauses"] if samples and samples[0]["environment"] == "SAT" else None
        }
    }
    
    with open(output_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved dataset to {output_path}")
    logger.info(f"  - Full dataset: validator_dataset.json ({len(samples)} samples)")
    logger.info(f"  - Training format: training_dataset.json")
    logger.info(f"  - Statistics: dataset_stats.json")
    
    return output_path

async def main():
    """Generate validator dataset"""
    
    # Configuration
    TOTAL_SAMPLES = int(os.getenv("TOTAL_SAMPLES", "200"))  # Start smaller for testing
    SAT_RATIO = float(os.getenv("SAT_RATIO", "0.75"))
    SEED = int(os.getenv("SEED", "42"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data")
    
    logger.info("🧪 VALIDATOR DATA GENERATION")
    logger.info("=" * 40)
    logger.info(f"Total samples: {TOTAL_SAMPLES}")
    logger.info(f"SAT ratio: {SAT_RATIO}")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    # Generate data
    generator = ValidatorDataGenerator(seed=SEED)
    samples = await generator.generate_dataset(TOTAL_SAMPLES, SAT_RATIO)
    
    if not samples:
        logger.error("❌ Failed to generate any samples")
        return 1
    
    # Save dataset
    output_path = save_dataset(samples, OUTPUT_DIR)
    
    # Show examples
    logger.info("\n📊 SAMPLE EXAMPLES")
    logger.info("-" * 30)
    
    sat_samples = [s for s in samples if s["environment"] == "SAT"]
    elr_samples = [s for s in samples if s["environment"] == "ELR"]
    
    if sat_samples:
        sat_example = sat_samples[0]
        logger.info("SAT Example:")
        logger.info(f"  Prompt: {sat_example['prompt'][:80]}...")
        logger.info(f"  Answer: {sat_example['completion']}")
        logger.info(f"  Vars: {sat_example['n_vars']}, Clauses: {sat_example['n_clauses']}")
    
    if elr_samples:
        elr_example = elr_samples[0]
        logger.info("ELR Example:")
        logger.info(f"  Prompt: {elr_example['prompt'][:80]}...")
        logger.info(f"  Answer: {elr_example['completion']}")
    
    logger.info(f"\n✅ Dataset generation complete!")
    logger.info(f"📁 Files saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))