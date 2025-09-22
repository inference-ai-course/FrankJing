"""
Main execution script for multimodal summarization and reward modeling project.
Orchestrates the pipeline from summary generation to evaluation (post-data collection).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils import setup_logging, set_random_seeds, format_timestamp
# Data collection moved to collect_data.py
from data_formatter import DataFormatter
from summary_generator import SummaryGenerator, MultimodalSummaryGenerator
from annotation_interface import CommandLineAnnotationInterface, WebAnnotationInterface, BatchAnnotationInterface
from reward_trainer import RewardModelTrainer, RewardModelEvaluator
from evaluator import SummaryEvaluator, ComparativeAnalyzer

class MultimodalSummarizationPipeline:
    """Main pipeline for multimodal summarization and reward modeling (post-data collection)."""

    def __init__(self):
        self.logger = setup_logging("INFO")

        # Set random seeds
        set_random_seeds(42)

        # Create directories
        self._create_directories()

        # Initialize components (no PaperCollector - data collection is separate)
        self.data_formatter = DataFormatter()
        self.summary_generator = SummaryGenerator()
        self.multimodal_generator = MultimodalSummaryGenerator()
        self.annotation_interface = CommandLineAnnotationInterface()
        self.reward_trainer = RewardModelTrainer()
        self.evaluator = SummaryEvaluator()
        self.analyzer = ComparativeAnalyzer()

        self.logger.info("Post-collection pipeline initialized")
    
    def _create_directories(self):
        """Create necessary directories for the project."""
        directories = [
            "data/papers",
            "data/summaries", 
            "data/annotations",
            "outputs",
            "models",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(self, papers: List[Dict[str, Any]], eval_papers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the complete post-collection pipeline."""
        self.logger.info("Starting post-collection pipeline execution")

        results = {
            "pipeline_start": format_timestamp(),
            "config": {
                "num_papers": len(papers),
                "num_eval_papers": len(eval_papers) if eval_papers else 0
            }
        }

        try:
            # Step 1: Verify papers input
            if not papers:
                raise ValueError("No papers provided. Please run collect_data.py first.")

            results["papers_loaded"] = len(papers)
            self.logger.info(f"Loaded {len(papers)} papers")
            
            # Step 2: Generate summaries
            self.logger.info("Step 2: Generating summaries")
            summaries = self.summary_generator.generate_summaries_for_papers(
                papers, num_summaries_per_paper=2
            )
            results["summaries_generated"] = len(summaries)
            self.logger.info(f"Generated {len(summaries)} summaries")
            
            # Step 3: Create summary pairs for annotation
            self.logger.info("Step 3: Creating summary pairs")
            summary_pairs = self.summary_generator.create_summary_pairs(summaries)
            results["summary_pairs"] = len(summary_pairs)
            self.logger.info(f"Created {len(summary_pairs)} summary pairs")
            
            # Step 4: Human annotation
            self.logger.info("Step 4: Collecting human annotations")
            annotations = self.annotation_interface.annotate_summary_pairs(
                summary_pairs, annotator_id="human"
            )
            results["annotations_collected"] = len(annotations)
            self.logger.info(f"Collected {len(annotations)} annotations")
            
            # Step 5: Prepare reward training data
            self.logger.info("Step 5: Preparing reward training data")
            training_data = self.data_formatter.create_reward_training_data(annotations)
            results["training_examples"] = len(training_data)
            self.logger.info(f"Prepared {len(training_data)} training examples")
            
            # Step 6: Train reward model
            self.logger.info("Step 6: Training reward model")
            train_dataset, val_dataset = self.reward_trainer.prepare_dataset(training_data)
            self.reward_trainer.train(train_dataset, val_dataset, "models/reward_model")
            results["reward_model_trained"] = True
            self.logger.info("Reward model training completed")
            
            # Step 7: Evaluate on evaluation papers
            self.logger.info("Step 7: Evaluating on evaluation papers")
            if eval_papers:
                eval_summaries = self.summary_generator.generate_summaries_for_papers(
                    eval_papers, num_summaries_per_paper=1
                )
            else:
                # Use a subset of training papers for evaluation if no eval_papers provided
                eval_papers = papers[:min(5, len(papers))]
                eval_summaries = summaries[:min(5, len(summaries))]
            
            # Create references (using abstracts as ground truth)
            references = [paper.get('abstract', '') for paper in eval_papers]
            predictions = [summary['text'] for summary in eval_summaries]
            
            # Comprehensive evaluation
            evaluation_results = self.evaluator.evaluate_summaries(
                predictions, references, self.reward_trainer
            )
            results["evaluation_results"] = evaluation_results
            self.logger.info("Evaluation completed")
            
            # Step 8: Generate reports
            self.logger.info("Step 8: Generating reports")
            self._generate_reports(results)
            
            results["pipeline_end"] = format_timestamp()
            results["success"] = True
            
            self.logger.info("Full pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def load_papers_from_file(self, papers_file: str) -> List[Dict[str, Any]]:
        """Load papers from a JSON file."""
        self.logger.info(f"Loading papers from {papers_file}")
        import json
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        self.logger.info(f"Loaded {len(papers)} papers")
        return papers
    
    def run_summarization_only(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run only the summarization step."""
        self.logger.info("Running summarization only")
        summaries = self.summary_generator.generate_summaries_for_papers(papers)
        self.logger.info(f"Generated {len(summaries)} summaries")
        return summaries
    
    def run_annotation_only(self, summary_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run only the annotation step."""
        self.logger.info("Running annotation only")
        annotations = self.annotation_interface.annotate_summary_pairs(summary_pairs)
        self.logger.info(f"Collected {len(annotations)} annotations")
        return annotations
    
    def run_reward_training_only(self, training_data: List[Dict[str, Any]]) -> None:
        """Run only the reward model training step."""
        self.logger.info("Running reward model training only")
        train_dataset, val_dataset = self.reward_trainer.prepare_dataset(training_data)
        self.reward_trainer.train(train_dataset, val_dataset, "models/reward_model")
        self.logger.info("Reward model training completed")
    
    def run_evaluation_only(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Run only the evaluation step."""
        self.logger.info("Running evaluation only")
        results = self.evaluator.evaluate_summaries(
            predictions, references, self.reward_trainer
        )
        self.logger.info("Evaluation completed")
        return results
    
    def _generate_reports(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive reports."""
        try:
            # Create reports directory
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = format_timestamp()
            
            # Save results
            results_file = reports_dir / f"pipeline_results_{timestamp}.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Generate evaluation report
            if "evaluation_results" in results:
                eval_report = self.evaluator.generate_evaluation_report(
                    results["evaluation_results"]
                )
                report_file = reports_dir / f"evaluation_report_{timestamp}.txt"
                with open(report_file, 'w') as f:
                    f.write(eval_report)
                
                self.logger.info(f"Evaluation report saved to {report_file}")
            
            # Generate summary
            summary = self._generate_pipeline_summary(results)
            summary_file = reports_dir / f"pipeline_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            self.logger.info(f"Pipeline summary saved to {summary_file}")
            self.logger.info(f"All reports saved to {reports_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of the pipeline execution."""
        summary = []
        summary.append("=" * 80)
        summary.append("MULTIMODAL SUMMARIZATION PIPELINE SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Pipeline Start: {results.get('pipeline_start', 'Unknown')}")
        summary.append(f"Pipeline End: {results.get('pipeline_end', 'Unknown')}")
        summary.append(f"Success: {results.get('success', False)}")
        summary.append("")
        
        # Configuration
        config = results.get('config', {})
        summary.append("CONFIGURATION:")
        summary.append("-" * 15)
        summary.append(f"Number of papers: {config.get('num_papers', 0)}")
        summary.append(f"Number of evaluation papers: {config.get('num_eval_papers', 0)}")
        summary.append("")
        
        # Results
        summary.append("RESULTS:")
        summary.append("-" * 10)
        summary.append(f"Papers collected: {results.get('papers_collected', 0)}")
        summary.append(f"Summaries generated: {results.get('summaries_generated', 0)}")
        summary.append(f"Summary pairs created: {results.get('summary_pairs', 0)}")
        summary.append(f"Annotations collected: {results.get('annotations_collected', 0)}")
        summary.append(f"Training examples: {results.get('training_examples', 0)}")
        summary.append(f"Reward model trained: {results.get('reward_model_trained', False)}")
        summary.append("")
        
        # Evaluation summary
        if "evaluation_results" in results:
            eval_results = results["evaluation_results"]
            summary.append("EVALUATION SUMMARY:")
            summary.append("-" * 20)
            
            if "rouge_scores" in eval_results:
                rouge = eval_results["rouge_scores"]
                summary.append("ROUGE Scores:")
                for metric, score in rouge.items():
                    if isinstance(score, (int, float)):
                        summary.append(f"  {metric}: {score:.4f}")
            
            if "bertscore_scores" in eval_results:
                bert = eval_results["bertscore_scores"]
                summary.append("BERTScore:")
                summary.append(f"  F1: {bert.get('f1', 0):.4f}")
            
            if "reward_scores" in eval_results and eval_results["reward_scores"]:
                import numpy as np
                scores = eval_results["reward_scores"]
                summary.append("Reward Model Scores:")
                summary.append(f"  Mean: {np.mean(scores):.4f}")
                summary.append(f"  Std: {np.std(scores):.4f}")
        
        # Error information
        if "error" in results:
            summary.append("")
            summary.append("ERROR:")
            summary.append("-" * 8)
            summary.append(results["error"])
        
        summary.append("=" * 80)
        
        return "\n".join(summary)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multimodal Summarization and Reward Modeling Pipeline (Post-Collection)")

    # Pipeline options
    parser.add_argument("--mode", choices=["full", "summarize", "annotate", "train", "evaluate"],
                       default="full", help="Pipeline mode to run")

    # Input/Output options
    parser.add_argument("--input-papers", type=str, required=True,
                       help="JSON file containing collected papers (from collect_data.py)")
    parser.add_argument("--eval-papers", type=str,
                       help="JSON file containing evaluation papers (optional)")
    parser.add_argument("--input-file", type=str, help="Input file for specific modes")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    
    # Model options
    parser.add_argument("--llama-model", type=str, 
                       default="meta-llama/Llama-3.1-7B-Instruct",
                       help="LLaMA model to use")
    parser.add_argument("--reward-model", type=str, 
                       default="microsoft/deberta-v3-base",
                       help="Reward model to use")
    
    # Training options
    parser.add_argument("--batch-size", type=int, default=8, 
                       help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, 
                       help="Learning rate")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--fp16", action="store_true", 
                       help="Use FP16 for training")
    
    args = parser.parse_args()
    
    # Configuration is now hardcoded in individual modules
    
    # Initialize pipeline
    pipeline = MultimodalSummarizationPipeline()
    
    try:
        # Load papers from input file
        papers = pipeline.load_papers_from_file(args.input_papers)

        # Load evaluation papers if provided
        eval_papers = None
        if args.eval_papers:
            eval_papers = pipeline.load_papers_from_file(args.eval_papers)

        if args.mode == "full":
            results = pipeline.run_full_pipeline(papers, eval_papers)
            print(f"Pipeline completed successfully: {results['success']}")
            
        elif args.mode == "summarize":
            summaries = pipeline.run_summarization_only(papers)
            print(f"Generated {len(summaries)} summaries")
            
        elif args.mode == "annotate":
            if not args.input_file:
                print("Error: --input-file required for annotate mode")
                sys.exit(1)
            # Load summary pairs and run annotation
            import json
            with open(args.input_file, 'r') as f:
                summary_pairs = json.load(f)
            annotations = pipeline.run_annotation_only(summary_pairs)
            print(f"Collected {len(annotations)} annotations")
            
        elif args.mode == "train":
            if not args.input_file:
                print("Error: --input-file required for train mode")
                sys.exit(1)
            # Load training data and run training
            import json
            with open(args.input_file, 'r') as f:
                training_data = json.load(f)
            pipeline.run_reward_training_only(training_data)
            print("Reward model training completed")
            
        elif args.mode == "evaluate":
            if not args.input_file:
                print("Error: --input-file required for evaluate mode")
                sys.exit(1)
            # Load predictions and references and run evaluation
            import json
            with open(args.input_file, 'r') as f:
                data = json.load(f)
            predictions = data.get('predictions', [])
            references = data.get('references', [])
            results = pipeline.run_evaluation_only(predictions, references)
            print("Evaluation completed")
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
