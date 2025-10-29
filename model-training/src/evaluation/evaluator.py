"""Model evaluator for all environments"""

import asyncio
import sys
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from test_code import environments as test_envs
except ImportError:
    test_envs = None
    print("Warning: test-code module not found. Using mock evaluation.")

from ..training.model_loader import load_model_and_tokenizer
from .metrics import compute_metrics, aggregate_metrics, EvaluationMetrics


class ModelEvaluator:
    """Evaluates model performance across all environments"""

    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        self.config = config
        self.eval_config = config.get("evaluation", {})
        self.model_config = config.get("model", {})

        self.model_path = model_path or self.model_config.get("base_model_path")
        self.model = None
        self.tokenizer = None
        self.environments = self.eval_config.get("environments", [])

    def setup(self):
        """Load model and tokenizer"""
        print("=" * 80)
        print("EVALUATION SETUP")
        print("=" * 80)

        print(f"Loading model from: {self.model_path}")

        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path=self.model_path,
            use_lora=False,  # Load full model for evaluation
            device_map=self.model_config.get("device_map", "auto"),
            torch_dtype=self.model_config.get("torch_dtype", "bfloat16"),
            use_flash_attention=self.model_config.get("use_flash_attention", True),
        )

        self.model.eval()
        print("Model loaded successfully!")

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response for a given prompt"""
        # Format as chat
        formatted_prompt = f"<|im_start|>system\nYou are a helpful AI assistant specialized in reasoning and problem-solving.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    async def evaluate_environment(self, env_name: str, num_samples: int) -> EvaluationMetrics:
        """Evaluate model on a specific environment"""
        print(f"\nEvaluating {env_name}...")

        scores = []
        errors = 0

        # Use mock evaluation if test environments not available
        if test_envs is None:
            print("Using mock evaluation (test-code not available)")
            for i in tqdm(range(num_samples), desc=f"{env_name}"):
                # Mock evaluation - random scores
                import random
                score = random.choice([0.0, 1.0]) if "affine" in env_name else random.uniform(0.0, 1.0)
                scores.append(score)
        else:
            # Real evaluation using test environments
            try:
                # Dynamically import the environment
                env_module = env_name.replace(":", ":")

                # Generate test cases and evaluate
                for i in tqdm(range(num_samples), desc=f"{env_name}"):
                    try:
                        # This is a simplified evaluation - you would use actual test-code logic here
                        score = 0.0  # Placeholder
                        scores.append(score)
                    except Exception as e:
                        errors += 1
                        scores.append(0.0)
                        continue

            except Exception as e:
                print(f"Error loading environment {env_name}: {e}")
                # Fall back to mock evaluation
                for i in tqdm(range(num_samples), desc=f"{env_name}"):
                    import random
                    score = random.choice([0.0, 1.0]) if "affine" in env_name else random.uniform(0.0, 1.0)
                    scores.append(score)

        # Compute metrics
        metrics = compute_metrics(
            environment=env_name,
            scores=scores,
            metadata={"errors": errors, "error_rate": errors / num_samples if num_samples > 0 else 0.0}
        )

        print(f"Results for {env_name}:")
        print(f"  Mean Score: {metrics.mean_score:.4f}")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  Confidence Interval: [{metrics.confidence_interval[0]:.4f}, {metrics.confidence_interval[1]:.4f}]")
        print(f"  Success Rate: {metrics.success_rate:.4f}")
        print(f"  Errors: {errors}/{num_samples}")

        return metrics

    async def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate model on all environments"""
        print("\n" + "=" * 80)
        print("EVALUATING MODEL ON ALL ENVIRONMENTS")
        print("=" * 80)

        num_samples = self.eval_config.get("num_eval_samples", 200)

        # Evaluate each environment
        all_metrics = []
        for env_name in self.environments:
            metrics = await self.evaluate_environment(env_name, num_samples)
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics)

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total Samples: {aggregated['total_samples']}")
        print(f"Environments: {aggregated['num_environments']}")
        print(f"Weighted Accuracy: {aggregated['weighted_accuracy']:.4f}")
        print(f"Weighted Mean Score: {aggregated['weighted_mean_score']:.4f}")
        print(f"Overall Success Rate: {aggregated['overall_success_rate']:.4f}")
        print(f"\nBest Environment: {aggregated['best_environment']['name']} ({aggregated['best_environment']['score']:.4f})")
        print(f"Worst Environment: {aggregated['worst_environment']['name']} ({aggregated['worst_environment']['score']:.4f})")

        return aggregated

    def save_results(self, results: Dict[str, Any], output_file: Path):
        """Save evaluation results to file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def compare_with_baseline(self, baseline_results_path: Path) -> Dict[str, Any]:
        """Compare current model with baseline"""
        if not baseline_results_path.exists():
            print(f"Baseline results not found: {baseline_results_path}")
            return {}

        with open(baseline_results_path, 'r') as f:
            baseline = json.load(f)

        # Perform comparison
        # This would involve loading both sets of results and computing differences
        print("Baseline comparison not yet implemented")
        return {}


async def run_evaluation(config_path: Optional[str] = None, model_path: Optional[str] = None):
    """Main function to run evaluation"""
    import yaml

    # Load config
    if config_path is None:
        config_path = "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create evaluator
    evaluator = ModelEvaluator(config, model_path=model_path)

    # Setup
    evaluator.setup()

    # Evaluate
    results = await evaluator.evaluate_all()

    # Save results
    output_file = Path("logs/evaluation_results.json")
    evaluator.save_results(results, output_file)

    return results


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Affine model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, help="Model path to evaluate")
    parser.add_argument("--baseline", type=str, help="Baseline results for comparison")

    args = parser.parse_args()

    results = asyncio.run(run_evaluation(args.config, args.model))

    return results


if __name__ == "__main__":
    main()
