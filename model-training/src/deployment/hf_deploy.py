"""HuggingFace deployment utilities"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, create_repo, upload_folder
import yaml


class HuggingFaceDeployer:
    """Deploy models to HuggingFace Hub"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hf_config = config.get("huggingface", {})
        self.api = HfApi()

        # Get HF token from environment
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.token:
            print("Warning: No HuggingFace token found in environment variables")
            print("Set HF_TOKEN or HUGGINGFACE_TOKEN to enable deployment")

    def create_model_card(self, model_path: Path, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Create model card content"""
        model_name = self.hf_config.get("repo_name", "affine-model")

        card_content = f"""---
license: apache-2.0
language:
- en
tags:
- reinforcement-learning
- reasoning
- qwen3
- affine
datasets:
- satpalsr/rl-python
metrics:
- accuracy
pipeline_tag: text-generation
---

# {model_name}

Fine-tuned version of Affine-0004 (Qwen3-4B) optimized for reasoning and interactive agent tasks.

## Model Description

This model is fine-tuned on 8 diverse environments testing:
- **Logical Reasoning**: SAT solving, abduction, deduction
- **Interactive Agents**: WebShop, AlfWorld, BabyAI, SciWorld, TextCraft

The model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and supports:
- Extended context length (256K tokens)
- Flash Attention 2 for efficient inference
- BFloat16 precision

## Training Details

### Base Model
- Architecture: Qwen3ForCausalLM
- Parameters: 4B
- Context Length: 262,144 tokens

### Fine-tuning
- Method: LoRA + PPO (Proximal Policy Optimization)
- LoRA Rank: 64
- LoRA Alpha: 128
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Data
- Affine Tasks: 10,000+ samples from satpalsr/rl-python
- AgentGym Tasks: 5,000+ episodes across 5 environments

### Hyperparameters
- Learning Rate: 2e-5
- Batch Size: 16 (effective)
- Epochs: 3
- Optimizer: AdamW
- Scheduler: Cosine with warmup

"""

        if metrics:
            card_content += f"""## Performance Metrics

"""
            if "weighted_accuracy" in metrics:
                card_content += f"- **Overall Accuracy**: {metrics['weighted_accuracy']:.4f}\n"
            if "weighted_mean_score" in metrics:
                card_content += f"- **Mean Score**: {metrics['weighted_mean_score']:.4f}\n"
            if "overall_success_rate" in metrics:
                card_content += f"- **Success Rate**: {metrics['overall_success_rate']:.4f}\n"

            card_content += "\n### Per-Environment Performance\n\n"
            card_content += "| Environment | Accuracy | Mean Score | Confidence Interval |\n"
            card_content += "|-------------|----------|------------|--------------------|\n"

            if "per_environment" in metrics:
                for env_name, env_metrics in metrics["per_environment"].items():
                    ci = env_metrics.get("confidence_interval", {})
                    ci_lower = ci.get("lower", 0.0)
                    ci_upper = ci.get("upper", 0.0)
                    card_content += f"| {env_name} | {env_metrics.get('accuracy', 0.0):.4f} | {env_metrics.get('mean_score', 0.0):.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] |\n"

        card_content += """
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "YOUR_USERNAME/""" + model_name + """",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "YOUR_USERNAME/""" + model_name + """",
    trust_remote_code=True
)

# Generate response
prompt = "Solve the following SAT problem: (x1 ∨ x2) ∧ (¬x1 ∨ x3)"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Evaluation

The model is evaluated on the Affine subnet using the following criteria:
- Minimum 200 samples per environment
- Pareto dominance scoring across all environments
- Bayesian confidence intervals (80% confidence level)

## Limitations and Bias

- The model is optimized for reasoning and agent tasks
- Performance may vary on out-of-distribution tasks
- Inherits biases from the base Qwen3 model and training data

## Citation

```bibtex
@misc{affine-model-training,
  title={Fine-tuned Affine Model for Reasoning and Interactive Agents},
  year={2024},
  url={https://huggingface.co/YOUR_USERNAME/""" + model_name + """}
}
```

## License

Apache 2.0

## Acknowledgments

- Base model: [Affine-0004](https://huggingface.co/kiwikiw/Affine-0004)
- Training framework: Transformers, PEFT, TRL
- Evaluation: Affine Subnet (Bittensor)
"""

        return card_content

    def prepare_for_upload(self, model_path: Path, metrics: Optional[Dict[str, Any]] = None):
        """Prepare model directory for upload"""
        model_path = Path(model_path)

        print(f"Preparing model at {model_path} for upload...")

        # Create model card
        if self.hf_config.get("create_model_card", True):
            card_content = self.create_model_card(model_path, metrics)
            card_path = model_path / "README.md"

            with open(card_path, 'w') as f:
                f.write(card_content)

            print(f"Model card created: {card_path}")

        # Create .gitattributes for LFS
        gitattributes_content = """*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
"""
        gitattributes_path = model_path / ".gitattributes"
        with open(gitattributes_path, 'w') as f:
            f.write(gitattributes_content)

        print("Model prepared for upload")

    def upload_to_hub(
        self,
        model_path: Path,
        repo_name: Optional[str] = None,
        private: Optional[bool] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Upload model to HuggingFace Hub

        Args:
            model_path: Path to model directory
            repo_name: Repository name (username/model-name)
            private: Whether to make repo private
            commit_message: Commit message

        Returns:
            URL of uploaded model
        """
        if not self.token:
            raise ValueError("HuggingFace token not found. Set HF_TOKEN or HUGGINGFACE_TOKEN environment variable")

        model_path = Path(model_path)
        repo_name = repo_name or self.hf_config.get("repo_name", "affine-model")
        private = private if private is not None else self.hf_config.get("private", False)
        commit_message = commit_message or self.hf_config.get("commit_message", "Upload fine-tuned model")

        print(f"\n{'=' * 80}")
        print(f"UPLOADING MODEL TO HUGGINGFACE HUB")
        print(f"{'=' * 80}\n")
        print(f"Model Path: {model_path}")
        print(f"Repository: {repo_name}")
        print(f"Private: {private}")

        try:
            # Create repository if it doesn't exist
            print(f"\nCreating repository: {repo_name}")
            repo_url = create_repo(
                repo_id=repo_name,
                private=private,
                token=self.token,
                exist_ok=True
            )
            print(f"Repository created/verified: {repo_url}")

            # Upload folder
            print(f"\nUploading files from {model_path}...")
            upload_folder(
                folder_path=str(model_path),
                repo_id=repo_name,
                repo_type="model",
                token=self.token,
                commit_message=commit_message,
            )

            model_url = f"https://huggingface.co/{repo_name}"
            print(f"\n{'=' * 80}")
            print(f"MODEL UPLOADED SUCCESSFULLY")
            print(f"{'=' * 80}")
            print(f"Model URL: {model_url}")

            return model_url

        except Exception as e:
            print(f"\nError uploading model: {e}")
            raise

    def deploy(self, model_path: Path, metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Complete deployment workflow

        Args:
            model_path: Path to model directory
            metrics: Evaluation metrics to include in model card

        Returns:
            URL of deployed model
        """
        # Prepare for upload
        self.prepare_for_upload(model_path, metrics)

        # Upload to hub
        model_url = self.upload_to_hub(model_path)

        return model_url


def deploy_model(
    config_path: str = "config.yaml",
    model_path: Optional[str] = None,
    metrics_path: Optional[str] = None
):
    """
    Deploy model to HuggingFace Hub

    Args:
        config_path: Path to configuration file
        model_path: Path to model directory
        metrics_path: Path to evaluation metrics JSON
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load metrics if provided
    metrics = None
    if metrics_path:
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    # Get model path
    if model_path is None:
        model_path = Path(config["training"]["save_dir"]) / "sft_final"

    # Create deployer
    deployer = HuggingFaceDeployer(config)

    # Deploy
    model_url = deployer.deploy(Path(model_path), metrics)

    return model_url


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy model to HuggingFace Hub")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, required=True, help="Model directory path")
    parser.add_argument("--metrics", type=str, help="Evaluation metrics JSON path")

    args = parser.parse_args()

    model_url = deploy_model(args.config, args.model, args.metrics)

    print(f"\nDeployment complete! Model available at: {model_url}")


if __name__ == "__main__":
    main()
