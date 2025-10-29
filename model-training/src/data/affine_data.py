"""Data collection for Affine tasks (SAT, ABD, DED)"""

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from affine.utils import R2BufferedDataset


@dataclass
class AffineSample:
    """Represents a training sample for Affine tasks"""
    prompt: str
    response: str
    environment: str
    difficulty: float
    metadata: Dict[str, Any]


class AffineDataCollector:
    """Collects and processes data for Affine tasks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get("dataset", "satpalsr/rl-python")
        self.buffer_size = config.get("buffer_size", 5)
        self.max_batch = config.get("max_batch", 5)
        self.num_samples = config.get("num_samples", 10000)
        self.environments = config.get("environments", ["affine:sat", "affine:abd", "affine:ded"])

    async def collect_sat_samples(self, num_samples: int) -> List[AffineSample]:
        """Collect SAT solving samples"""
        samples = []

        for i in range(num_samples):
            # Generate k-SAT problem
            n_vars = random.randint(10, 20)
            k = random.randint(3, 5)
            n_clauses = int(n_vars * 4.2)  # Near phase transition

            clauses = []
            for _ in range(n_clauses):
                clause = []
                vars_in_clause = random.sample(range(1, n_vars + 1), k)
                for var in vars_in_clause:
                    lit = var if random.random() > 0.5 else -var
                    clause.append(lit)
                clauses.append(clause)

            # Generate satisfying assignment
            assignment = {i: random.choice([True, False]) for i in range(1, n_vars + 1)}

            # Format prompt
            clauses_str = " ∧ ".join([
                "(" + " ∨ ".join([f"{'¬' if lit < 0 else ''}x{abs(lit)}" for lit in clause]) + ")"
                for clause in clauses
            ])

            prompt = f"""Solve the following {k}-SAT problem with {n_vars} variables:

{clauses_str}

Provide a satisfying assignment in the format: x1=True, x2=False, x3=True, ...
If the formula is unsatisfiable, respond with "UNSAT"."""

            # Format response
            response = ", ".join([f"x{i}={assignment[i]}" for i in range(1, n_vars + 1)])

            samples.append(AffineSample(
                prompt=prompt,
                response=response,
                environment="affine:sat",
                difficulty=n_clauses / n_vars,
                metadata={"n_vars": n_vars, "k": k, "n_clauses": n_clauses}
            ))

        return samples

    async def collect_abd_samples(self, num_samples: int) -> List[AffineSample]:
        """Collect abduction (reverse engineering) samples"""
        samples = []

        # Initialize R2 dataset
        dataset = R2BufferedDataset(
            self.dataset_name,
            buffer_size=self.buffer_size,
            max_batch=self.max_batch
        )

        i = 0
        async for item in dataset:
            if i >= num_samples:
                break
            i += 1

            code = item.get("code", "")
            inputs = item.get("inputs", [])
            outputs = item.get("outputs", [])

            if not code or not inputs or not outputs:
                continue

            # Select random input-output pair
            idx = random.randint(0, min(len(inputs), len(outputs)) - 1)
            test_input = inputs[idx]
            expected_output = outputs[idx]

            prompt = f"""Given the following Python program and its expected output, determine what input would produce this output.

Program:
```python
{code}
```

Expected Output:
{expected_output}

Provide the input in the format:
<INPUT>
[input lines here]
</INPUT>"""

            response = f"""<INPUT>
{test_input}
</INPUT>"""

            samples.append(AffineSample(
                prompt=prompt,
                response=response,
                environment="affine:abd",
                difficulty=len(code.split('\n')) / 50.0,
                metadata={"code_length": len(code), "num_lines": len(code.split('\n'))}
            ))

        return samples

    async def collect_ded_samples(self, num_samples: int) -> List[AffineSample]:
        """Collect deduction (code generation) samples"""
        samples = []

        # Initialize R2 dataset
        dataset = R2BufferedDataset(
            self.dataset_name,
            buffer_size=self.buffer_size,
            max_batch=self.max_batch
        )

        i = 0
        async for item in dataset:
            if i >= num_samples:
                break
            i += 1

            problem = item.get("problem", "")
            code = item.get("code", "")
            inputs = item.get("inputs", [])
            outputs = item.get("outputs", [])

            if not problem or not code:
                continue

            # Create example test cases
            test_cases = []
            for inp, out in zip(inputs[:3], outputs[:3]):
                test_cases.append(f"Input: {inp}\nOutput: {out}")

            examples = "\n\n".join(test_cases) if test_cases else ""

            # Build the example test cases section
            example_section = ""
            if examples:
                newline = "\n"
                example_section = f"Example test cases:{newline}{examples}"

            prompt = f"""Write a Python program that solves the following problem:

{problem}

{example_section}

Write a complete program that reads from standard input and writes to standard output."""

            response = f"""```python
{code}
```"""

            samples.append(AffineSample(
                prompt=prompt,
                response=response,
                environment="affine:ded",
                difficulty=len(code.split('\n')) / 50.0,
                metadata={"code_length": len(code), "num_test_cases": len(test_cases)}
            ))

        return samples

    async def collect_all_samples(self) -> Dict[str, List[AffineSample]]:
        """Collect samples for all Affine environments"""
        all_samples = {}
        samples_per_env = self.num_samples // len(self.environments)

        for env in self.environments:
            print(f"Collecting {samples_per_env} samples for {env}...")

            if env == "affine:sat":
                samples = await self.collect_sat_samples(samples_per_env)
            elif env == "affine:abd":
                samples = await self.collect_abd_samples(samples_per_env)
            elif env == "affine:ded":
                samples = await self.collect_ded_samples(samples_per_env)
            else:
                print(f"Unknown environment: {env}")
                continue

            all_samples[env] = samples
            print(f"Collected {len(samples)} samples for {env}")

        return all_samples

    def save_samples(self, samples: Dict[str, List[AffineSample]], output_dir: Path):
        """Save collected samples to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for env, env_samples in samples.items():
            env_name = env.replace(":", "_")
            output_file = output_dir / f"{env_name}_samples.jsonl"

            with open(output_file, 'w') as f:
                for sample in env_samples:
                    json_line = {
                        "prompt": sample.prompt,
                        "response": sample.response,
                        "environment": sample.environment,
                        "difficulty": sample.difficulty,
                        "metadata": sample.metadata
                    }
                    f.write(json.dumps(json_line) + '\n')

            print(f"Saved {len(env_samples)} samples to {output_file}")

    def load_samples(self, input_dir: Path) -> Dict[str, List[AffineSample]]:
        """Load samples from disk"""
        all_samples = {}

        for env in self.environments:
            env_name = env.replace(":", "_")
            input_file = input_dir / f"{env_name}_samples.jsonl"

            if not input_file.exists():
                print(f"Warning: {input_file} not found")
                continue

            samples = []
            with open(input_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(AffineSample(
                        prompt=data["prompt"],
                        response=data["response"],
                        environment=data["environment"],
                        difficulty=data["difficulty"],
                        metadata=data["metadata"]
                    ))

            all_samples[env] = samples
            print(f"Loaded {len(samples)} samples from {input_file}")

        return all_samples


async def main():
    """Example usage"""
    config = {
        "dataset": "satpalsr/rl-python",
        "buffer_size": 5,
        "max_batch": 5,
        "num_samples": 1000,
        "environments": ["affine:sat", "affine:abd", "affine:ded"]
    }

    collector = AffineDataCollector(config)
    samples = await collector.collect_all_samples()

    output_dir = Path("data_cache/affine")
    collector.save_samples(samples, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
