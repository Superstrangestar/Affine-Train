#!/usr/bin/env python3
"""Script to run supervised fine-tuning"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.sft_trainer import SupervisedFineTuner
from utils.config import load_config
from utils.logger import setup_logger


def main():
    """Main SFT training workflow"""
    # Setup logger
    logger = setup_logger("sft-training", Path("logs"))
    logger.info("Starting supervised fine-tuning...")

    # Load config
    config = load_config("config.yaml")

    # Create trainer
    trainer = SupervisedFineTuner(config)

    try:
        # Setup
        logger.info("Setting up trainer...")
        trainer.setup()

        # Train
        logger.info("Starting training...")
        train_metrics = trainer.train()

        # Evaluate
        logger.info("Running evaluation...")
        eval_metrics = trainer.evaluate()

        logger.info("\n" + "=" * 80)
        logger.info("SFT TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Train Loss: {train_metrics.get('train_loss', 'N/A')}")
        logger.info(f"Eval Loss: {eval_metrics.get('eval_loss', 'N/A')}")

    except Exception as e:
        logger.error(f"Error during SFT training: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()
