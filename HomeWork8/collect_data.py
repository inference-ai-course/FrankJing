"""
Data collection script for academic papers.
Handles the first step of the pipeline: collecting papers from arXiv.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from utils import setup_logging, set_random_seeds, format_timestamp
from data_collector import PaperCollector


def collect_papers(num_papers: int = 10, output_file: str = None) -> str:
    """
    Collect academic papers from arXiv.

    Args:
        num_papers: Number of papers to collect
        output_file: Path to save collected papers (optional)

    Returns:
        Path to the saved papers file
    """
    logger = setup_logging("INFO")
    set_random_seeds(42)

    # Create directories
    Path("data").mkdir(parents=True, exist_ok=True)

    # Initialize paper collector
    paper_collector = PaperCollector()

    try:
        logger.info(f"Collecting {num_papers} papers from arXiv")
        papers = paper_collector.collect_sample_papers(num_papers)
        logger.info(f"Successfully collected {len(papers)} papers")

        # Save papers to file
        if output_file is None:
            timestamp = format_timestamp()
            output_file = f"data/papers/collected_papers_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Papers saved to {output_file}")

        # Print summary
        print(f"Data Collection Complete!")
        print(f"Papers collected: {len(papers)}")
        print(f"Output file: {output_file}")

        return output_file

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point for data collection script."""
    parser = argparse.ArgumentParser(description="Collect academic papers from arXiv")

    parser.add_argument("--num-papers", type=int, default=10,
                       help="Number of papers to collect (default: 10)")
    parser.add_argument("--output-file", type=str,
                       help="Output file path for collected papers")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")

    args = parser.parse_args()

    # Collect papers
    output_file = collect_papers(
        num_papers=args.num_papers,
        output_file=args.output_file
    )

    print(f"\nTo proceed with the pipeline, run:")
    print(f"python main.py --input-papers {output_file}")


if __name__ == "__main__":
    main()