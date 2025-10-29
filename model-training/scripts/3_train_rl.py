#!/usr/bin/env python3
"""Script to run RL training"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.rl_trainer import RLTrainer
from utils.config import load_config
from utils.logger import setup_logger


def main():
    """Main RL training workflow"""
    # Setup logger
    logger = setup_logger("rl-training", Path("logs"))
    logger.info("Starting RL training...")

    # Load config
    config = load_config("config.yaml")

    # Update model path to use SFT output
    sft_model_path = Path(config["training"]["save_dir"]) / "sft_final"
    if sft_model_path.exists():
        logger.info(f"Using SFT model from: {sft_model_path}")
        config["model"]["base_model_path"] = str(sft_model_path)
    else:
        logger.warning(f"SFT model not found at {sft_model_path}, using base model")

    # Create trainer
    trainer = RLTrainer(config)

    try:
        # Setup
        logger.info("Setting up RL trainer...")
        trainer.setup()

        # Train
        logger.info("Starting RL training...")
        metrics = trainer.train()

        logger.info("\n" + "=" * 80)
        logger.info("RL TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Best Reward: {metrics.get('best_reward', 'N/A')}")

    except Exception as e:
        logger.error(f"Error during RL training: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()
