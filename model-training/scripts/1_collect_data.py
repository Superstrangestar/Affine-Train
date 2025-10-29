#!/usr/bin/env python3
"""Script to collect training data from all environments"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.affine_data import AffineDataCollector
from data.agentgym_data import AgentGymDataCollector
from utils.config import load_config
from utils.logger import setup_logger


async def main():
    """Main data collection workflow"""
    # Setup logger
    logger = setup_logger("data-collection", Path("logs"))
    logger.info("Starting data collection...")

    # Load config
    config = load_config("config.yaml")

    # Collect Affine data
    logger.info("=" * 80)
    logger.info("COLLECTING AFFINE DATA")
    logger.info("=" * 80)

    affine_config = config.get("data", {}).get("affine", {})
    affine_collector = AffineDataCollector(affine_config)

    try:
        affine_samples = await affine_collector.collect_all_samples()
        output_dir = Path("data_cache/affine")
        affine_collector.save_samples(affine_samples, output_dir)
        logger.info(f"Affine data collected: {sum(len(s) for s in affine_samples.values())} samples")
    except Exception as e:
        logger.error(f"Error collecting Affine data: {e}")
        logger.exception(e)

    # Collect AgentGym data
    logger.info("\n" + "=" * 80)
    logger.info("COLLECTING AGENTGYM DATA")
    logger.info("=" * 80)

    agentgym_config = config.get("data", {}).get("agentgym", {})
    agentgym_collector = AgentGymDataCollector(agentgym_config)

    try:
        agentgym_episodes = await agentgym_collector.collect_all_episodes()
        output_dir = Path("data_cache/agentgym")
        agentgym_collector.save_episodes(agentgym_episodes, output_dir)
        logger.info(f"AgentGym data collected: {sum(len(e) for e in agentgym_episodes.values())} episodes")
    except Exception as e:
        logger.error(f"Error collecting AgentGym data: {e}")
        logger.exception(e)

    logger.info("\n" + "=" * 80)
    logger.info("DATA COLLECTION COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
