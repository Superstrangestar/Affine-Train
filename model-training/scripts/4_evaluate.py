#!/usr/bin/env python3
"""Script to evaluate trained model"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.evaluator import ModelEvaluator
from utils.config import load_config
from utils.logger import setup_logger


async def main():
    """Main evaluation workflow"""
    # Setup logger
    logger = setup_logger("evaluation", Path("logs"))
    logger.info("Starting model evaluation...")

    # Load config
    config = load_config("config.yaml")

    # Determine model path
    # Try RL model first, fall back to SFT, then base
    model_paths = [
        Path(config["training"]["save_dir"]) / "rl_final",
        Path(config["training"]["save_dir"]) / "sft_final",
        config["model"]["base_model_path"]
    ]

    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = str(path)
            logger.info(f"Using model from: {model_path}")
            break

    if model_path is None:
        logger.error("No trained model found!")
        return

    # Create evaluator
    evaluator = ModelEvaluator(config, model_path=model_path)

    try:
        # Setup
        logger.info("Setting up evaluator...")
        evaluator.setup()

        # Evaluate
        logger.info("Running evaluation...")
        results = await evaluator.evaluate_all()

        # Save results
        output_file = Path("logs/evaluation_results.json")
        evaluator.save_results(results, output_file)

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    asyncio.run(main())
