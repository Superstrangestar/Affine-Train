#!/usr/bin/env python3
"""Script to deploy model to HuggingFace Hub"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deployment.hf_deploy import HuggingFaceDeployer
from utils.config import load_config
from utils.logger import setup_logger


def main():
    """Main deployment workflow"""
    # Setup logger
    logger = setup_logger("deployment", Path("logs"))
    logger.info("Starting model deployment...")

    # Load config
    config = load_config("config.yaml")

    # Determine model path
    model_paths = [
        Path(config["training"]["save_dir"]) / "rl_final",
        Path(config["training"]["save_dir"]) / "sft_final",
    ]

    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            logger.info(f"Using model from: {model_path}")
            break

    if model_path is None:
        logger.error("No trained model found!")
        return

    # Load evaluation metrics if available
    metrics = None
    metrics_path = Path("logs/evaluation_results.json")
    if metrics_path.exists():
        logger.info(f"Loading evaluation metrics from: {metrics_path}")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    # Create deployer
    deployer = HuggingFaceDeployer(config)

    try:
        # Deploy
        logger.info("Deploying model to HuggingFace Hub...")
        model_url = deployer.deploy(model_path, metrics)

        logger.info("\n" + "=" * 80)
        logger.info("DEPLOYMENT COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Model URL: {model_url}")

    except Exception as e:
        logger.error(f"Error during deployment: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()
