#!/usr/bin/env python3
"""Script to run the complete training pipeline"""

import asyncio
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger


async def run_step(name: str, script: str, logger):
    """Run a pipeline step"""
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP: {name}")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=Path(__file__).parent.parent,
            check=True,
            capture_output=False
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ {name} completed in {duration:.2f}s")
        return True

    except subprocess.CalledProcessError as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"✗ {name} failed after {duration:.2f}s")
        logger.error(f"Error: {e}")
        return False


async def main():
    """Run complete training pipeline"""
    # Setup logger
    logger = setup_logger("pipeline", Path("logs"))

    logger.info("=" * 80)
    logger.info("AFFINE MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pipeline_start = datetime.now()

    # Define pipeline steps
    steps = [
        ("Data Collection", "scripts/1_collect_data.py"),
        ("Supervised Fine-Tuning", "scripts/2_train_sft.py"),
        ("RL Training", "scripts/3_train_rl.py"),
        ("Evaluation", "scripts/4_evaluate.py"),
        ("Deployment", "scripts/5_deploy.py"),
    ]

    # Run each step
    results = {}
    for name, script in steps:
        success = await run_step(name, script, logger)
        results[name] = success

        if not success:
            logger.error(f"Pipeline stopped due to failure in: {name}")
            break

    # Summary
    pipeline_duration = (datetime.now() - pipeline_start).total_seconds()

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Duration: {pipeline_duration / 60:.2f} minutes")
    logger.info("\nStep Results:")

    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {status}: {name}")

    if all(results.values()):
        logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        logger.error("\n❌ PIPELINE FAILED")


if __name__ == "__main__":
    asyncio.run(main())
