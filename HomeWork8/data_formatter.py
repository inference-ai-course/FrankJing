"""
Data formatting module for multimodal summarization pipeline.
Handles formatting data for different stages: summarization, annotation, and reward training.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from utils import setup_logging, save_jsonl, load_jsonl, format_timestamp


class DataFormatter:
    """Formats data for different stages of the pipeline."""

    def __init__(self):
        self.logger = setup_logging("INFO")

    def format_for_summarization(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format paper data for summarization."""
        return {
            'paper_id': paper_data['paper_id'],
            'title': paper_data['title'],
            'content': self._get_paper_content(paper_data),
            'metadata': {
                'authors': paper_data.get('authors', []),
                'published_date': paper_data.get('published_date', ''),
                'source': paper_data.get('source', ''),
                'text_length': paper_data.get('text_length', 0),
                'num_images': paper_data.get('num_images', 0)
            }
        }

    def _get_paper_content(self, paper_data: Dict[str, Any]) -> str:
        """Get the main content for summarization."""
        sections = paper_data.get('sections', {})

        # Combine sections in order of preference
        content_parts = []

        if sections.get('abstract'):
            content_parts.append(f"Abstract: {sections['abstract']}")

        if sections.get('introduction'):
            content_parts.append(f"Introduction: {sections['introduction']}")

        if sections.get('methodology'):
            content_parts.append(f"Methodology: {sections['methodology']}")

        if sections.get('results'):
            content_parts.append(f"Results: {sections['results']}")

        if sections.get('conclusion'):
            content_parts.append(f"Conclusion: {sections['conclusion']}")

        # If no sections found, use full text
        if not content_parts:
            content_parts.append(paper_data.get('full_text', ''))

        return '\n\n'.join(content_parts)

    def format_for_annotation(self, paper_data: Dict[str, Any],
                            summaries: List[str]) -> Dict[str, Any]:
        """Format data for human annotation."""
        return {
            'paper_id': paper_data['paper_id'],
            'title': paper_data['title'],
            'abstract': paper_data.get('abstract', ''),
            'summaries': summaries,
            'metadata': paper_data.get('metadata', {})
        }

    def create_reward_training_data(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert annotations to reward model training format.
        Creates dataset in JSONL format with 'chosen' and 'rejected' fields.
        """
        training_data = []

        for annotation in annotations:
            if annotation.get('preferred_index', 0) == 0:  # Skip ties
                continue

            if annotation['preferred_index'] == 1:
                chosen = annotation['summary_1']['text']
                rejected = annotation['summary_2']['text']
            else:
                chosen = annotation['summary_2']['text']
                rejected = annotation['summary_1']['text']

            training_data.append({
                'chosen': chosen,
                'rejected': rejected,
                'paper_id': annotation['paper_id'],
                'annotator_id': annotation['annotator_id'],
                'timestamp': annotation['timestamp']
            })

        self.logger.info(f"Created {len(training_data)} reward training examples")
        return training_data

    def save_reward_training_data(self, training_data: List[Dict[str, Any]],
                                output_path: Optional[str] = None) -> str:
        """Save reward training data in JSONL format."""
        if output_path is None:
            timestamp = format_timestamp()
            output_path = f"data/reward_training_{timestamp}.jsonl"

        save_jsonl(training_data, output_path)
        self.logger.info(f"Saved {len(training_data)} training examples to {output_path}")
        return output_path

    def load_reward_training_data(self, input_path: str) -> List[Dict[str, Any]]:
        """Load reward training data from JSONL format."""
        return load_jsonl(input_path)

    def format_summary_pairs(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format summaries into pairs for comparison."""
        pairs = []

        # Group summaries by paper_id
        summaries_by_paper = {}
        for summary in summaries:
            paper_id = summary['paper_id']
            if paper_id not in summaries_by_paper:
                summaries_by_paper[paper_id] = []
            summaries_by_paper[paper_id].append(summary)

        # Create pairs for each paper
        for paper_id, paper_summaries in summaries_by_paper.items():
            if len(paper_summaries) >= 2:
                # Take first two summaries as a pair
                pairs.append({
                    'paper_id': paper_id,
                    'summary_1': paper_summaries[0],
                    'summary_2': paper_summaries[1],
                    'pair_id': f"{paper_id}_pair_1"
                })

        self.logger.info(f"Created {len(pairs)} summary pairs")
        return pairs

    def validate_training_data(self, training_data: List[Dict[str, Any]]) -> bool:
        """Validate that training data has required fields for reward modeling."""
        required_fields = ['chosen', 'rejected']

        for i, example in enumerate(training_data):
            for field in required_fields:
                if field not in example:
                    self.logger.error(f"Training example {i} missing required field: {field}")
                    return False

                if not isinstance(example[field], str) or not example[field].strip():
                    self.logger.error(f"Training example {i} has invalid {field} field")
                    return False

        self.logger.info(f"Validated {len(training_data)} training examples")
        return True

    def format_evaluation_data(self, papers: List[Dict[str, Any]],
                             summaries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Format data for evaluation with references and predictions."""
        references = []
        predictions = []

        for paper in papers:
            # Use abstract as reference
            references.append(paper.get('abstract', ''))

        for summary in summaries:
            predictions.append(summary.get('text', ''))

        return {
            'references': references,
            'predictions': predictions
        }