"""Supervised Fine-Tuning trainer for Affine tasks"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
import wandb

from .model_loader import load_model_and_tokenizer, save_model, count_parameters
from data.dataset import TrainingDataset


class SupervisedFineTuner:
    """Supervised fine-tuning trainer"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.data_config = config.get("data", {}).get("affine", {})
        self.tracking_config = config.get("tracking", {})

        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """Initialize model, tokenizer, and datasets"""
        print("=" * 80)
        print("SUPERVISED FINE-TUNING SETUP")
        print("=" * 80)

        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path=self.model_config["base_model_path"],
            use_lora=self.training_config.get("use_lora", True),
            lora_config={
                "r": self.training_config.get("lora_r", 64),
                "lora_alpha": self.training_config.get("lora_alpha", 128),
                "lora_dropout": self.training_config.get("lora_dropout", 0.05),
                "target_modules": self.training_config.get("lora_target_modules", []),
            },
            device_map=self.model_config.get("device_map", "auto"),
            torch_dtype=self.model_config.get("torch_dtype", "bfloat16"),
            use_flash_attention=self.model_config.get("use_flash_attention", True),
        )

        # Print parameter counts
        param_info = count_parameters(self.model)
        print(f"\nModel Parameters:")
        print(f"  Total: {param_info['total_parameters']:,}")
        print(f"  Trainable: {param_info['trainable_parameters']:,}")
        print(f"  Trainable %: {param_info['trainable_percentage']:.2f}%")

        # Load datasets
        print("\nLoading datasets...")
        environments = self.data_config.get("environments", [])

        # Load train split
        train_data_dir = Path("data_cache/affine")
        self.train_dataset = TrainingDataset(
            data_dir=train_data_dir,
            environments=environments,
            tokenizer=self.tokenizer,
            max_length=self.model_config.get("context_length", 4096),
        )

        # Create validation split (10% of data)
        total_size = len(self.train_dataset)
        val_size = int(total_size * 0.1)
        train_size = total_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get("seed", 42))
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")

        # Initialize W&B if configured
        if self.tracking_config.get("use_wandb", False):
            wandb.init(
                project=self.tracking_config.get("wandb_project", "affine-training"),
                entity=self.tracking_config.get("wandb_entity"),
                name=self.tracking_config.get("experiment_name", "sft-training"),
                config=self.config
            )

    def train(self):
        """Run training"""
        print("\n" + "=" * 80)
        print("STARTING SUPERVISED FINE-TUNING")
        print("=" * 80 + "\n")

        # Create output directory
        output_dir = Path(self.training_config.get("output_dir", "checkpoints"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.training_config.get("num_epochs", 3),
            per_device_train_batch_size=self.training_config.get("batch_size", 2),
            per_device_eval_batch_size=self.training_config.get("eval_batch_size", 2),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 8),
            learning_rate=self.training_config.get("learning_rate", 2e-5),
            warmup_steps=self.training_config.get("warmup_steps", 100),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            logging_steps=self.training_config.get("logging_steps", 10),
            eval_steps=self.training_config.get("eval_steps", 500),
            save_steps=self.training_config.get("save_steps", 1000),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            fp16=self.training_config.get("fp16", False),
            bf16=self.training_config.get("bf16", True),
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", True),
            lr_scheduler_type=self.training_config.get("scheduler_type", "cosine"),
            report_to="wandb" if self.tracking_config.get("use_wandb", False) else "none",
            logging_dir=str(output_dir / "logs"),
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            ddp_find_unused_parameters=False if self.training_config.get("use_lora", True) else None,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )

        # Train
        print("Starting training...")
        train_result = trainer.train()

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Save final model
        save_dir = Path(self.training_config.get("save_dir", "models")) / "sft_final"
        save_model(self.model, self.tokenizer, save_dir, save_full_model=True)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Final model saved to: {save_dir}")

        return metrics

    def evaluate(self):
        """Run evaluation"""
        if self.model is None or self.val_dataset is None:
            raise RuntimeError("Must call setup() before evaluate()")

        print("\n" + "=" * 80)
        print("RUNNING EVALUATION")
        print("=" * 80 + "\n")

        # Create output directory
        output_dir = Path(self.training_config.get("output_dir", "checkpoints"))

        # Configure training arguments (for eval)
        eval_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_eval_batch_size=self.training_config.get("eval_batch_size", 2),
            fp16=self.training_config.get("fp16", False),
            bf16=self.training_config.get("bf16", True),
            report_to="none",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )

        # Evaluate
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        print(f"\nEvaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        return metrics


def run_sft_training(config_path: Optional[str] = None):
    """Main function to run SFT training"""
    import yaml

    # Load config
    if config_path is None:
        config_path = "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = SupervisedFineTuner(config)

    # Setup
    trainer.setup()

    # Train
    metrics = trainer.train()

    # Evaluate
    eval_metrics = trainer.evaluate()

    return metrics, eval_metrics


if __name__ == "__main__":
    run_sft_training()
