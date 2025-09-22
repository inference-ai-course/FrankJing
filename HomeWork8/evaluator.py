"""
Evaluation module for comparing ROUGE, BERTScore, and reward model scores.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

from utils import setup_logging, format_timestamp, save_json, load_json

class SummaryEvaluator:
    """Evaluates summaries using multiple metrics and compares their performance."""
    
    def __init__(self):
        self.logger = setup_logging("INFO")
        
        # Initialize metrics
        self.rouge = None
        self.bertscore = None
        self._load_metrics()
    
    def _load_metrics(self):
        """Load evaluation metrics."""
        try:
            from evaluate import load
            
            self.rouge = load("rouge")
            self.bertscore = load("bertscore")
            
            self.logger.info("Evaluation metrics loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading evaluation metrics: {e}")
            raise
    
    def compute_rouge_scores(self, predictions: List[str], 
                           references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores for summaries."""
        try:
            self.logger.info(f"Computing ROUGE scores for {len(predictions)} summaries")
            
            rouge_scores = self.rouge.compute(
                predictions=predictions,
                references=references
            )
            
            # Extract individual metrics
            result = {}
            rouge_metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
            for metric in rouge_metrics:
                if metric in rouge_scores:
                    result[metric] = rouge_scores[metric]
            
            self.logger.info(f"ROUGE scores computed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error computing ROUGE scores: {e}")
            return {}
    
    def compute_bertscore_scores(self, predictions: List[str], 
                                references: List[str]) -> Dict[str, Any]:
        """Compute BERTScore for summaries."""
        try:
            self.logger.info(f"Computing BERTScore for {len(predictions)} summaries")
            
            bertscore_scores = self.bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli"
            )
            
            # Compute average scores
            result = {
                "precision": np.mean(bertscore_scores["precision"]),
                "recall": np.mean(bertscore_scores["recall"]),
                "f1": np.mean(bertscore_scores["f1"]),
                "individual_scores": {
                    "precision": bertscore_scores["precision"],
                    "recall": bertscore_scores["recall"],
                    "f1": bertscore_scores["f1"]
                }
            }
            
            self.logger.info(f"BERTScore computed: P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1']:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error computing BERTScore: {e}")
            return {}
    
    def compute_reward_scores(self, summaries: List[str], 
                            reward_model) -> List[float]:
        """Compute reward model scores for summaries."""
        try:
            self.logger.info(f"Computing reward scores for {len(summaries)} summaries")
            
            scores = reward_model.score_summaries(summaries)
            
            self.logger.info(f"Reward scores computed: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
            return scores
            
        except Exception as e:
            self.logger.error(f"Error computing reward scores: {e}")
            return [0.0] * len(summaries)
    
    def evaluate_summaries(self, predictions: List[str], references: List[str], 
                          reward_model=None) -> Dict[str, Any]:
        """Comprehensive evaluation of summaries using all available metrics."""
        self.logger.info("Starting comprehensive summary evaluation")
        
        results = {
            "timestamp": format_timestamp(),
            "num_summaries": len(predictions),
            "rouge_scores": {},
            "bertscore_scores": {},
            "reward_scores": [],
            "correlations": {},
            "rankings": {}
        }
        
        # Compute ROUGE scores
        results["rouge_scores"] = self.compute_rouge_scores(predictions, references)
        
        # Compute BERTScore
        results["bertscore_scores"] = self.compute_bertscore_scores(predictions, references)
        
        # Compute reward scores if model provided
        if reward_model:
            results["reward_scores"] = self.compute_reward_scores(predictions, reward_model)
        
        # Compute correlations
        if results["reward_scores"]:
            results["correlations"] = self._compute_metric_correlations(
                results["rouge_scores"],
                results["bertscore_scores"],
                results["reward_scores"]
            )
        
        # Compute rankings
        results["rankings"] = self._compute_rankings(
            results["rouge_scores"],
            results["bertscore_scores"],
            results["reward_scores"]
        )
        
        self.logger.info("Summary evaluation completed")
        return results
    
    def _compute_metric_correlations(self, rouge_scores: Dict[str, float], 
                                   bertscore_scores: Dict[str, Any], 
                                   reward_scores: List[float]) -> Dict[str, float]:
        """Compute correlations between different metrics."""
        try:
            from scipy.stats import pearsonr, spearmanr
            
            correlations = {}
            
            # ROUGE correlations
            for metric, score in rouge_scores.items():
                if isinstance(score, list) and len(score) == len(reward_scores):
                    pearson_corr, _ = pearsonr(score, reward_scores)
                    spearman_corr, _ = spearmanr(score, reward_scores)
                    correlations[f"rouge_{metric}_pearson"] = pearson_corr
                    correlations[f"rouge_{metric}_spearman"] = spearman_corr
            
            # BERTScore correlations
            if "individual_scores" in bertscore_scores:
                for metric in ["precision", "recall", "f1"]:
                    scores = bertscore_scores["individual_scores"][metric]
                    if len(scores) == len(reward_scores):
                        pearson_corr, _ = pearsonr(scores, reward_scores)
                        spearman_corr, _ = spearmanr(scores, reward_scores)
                        correlations[f"bertscore_{metric}_pearson"] = pearson_corr
                        correlations[f"bertscore_{metric}_spearman"] = spearman_corr
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error computing correlations: {e}")
            return {}
    
    def _compute_rankings(self, rouge_scores: Dict[str, float], 
                         bertscore_scores: Dict[str, Any], 
                         reward_scores: List[float]) -> Dict[str, List[int]]:
        """Compute rankings based on different metrics."""
        rankings = {}
        
        # ROUGE rankings
        for metric, scores in rouge_scores.items():
            if isinstance(scores, list):
                rankings[f"rouge_{metric}"] = self._get_ranking(scores)
        
        # BERTScore rankings
        if "individual_scores" in bertscore_scores:
            for metric in ["precision", "recall", "f1"]:
                scores = bertscore_scores["individual_scores"][metric]
                rankings[f"bertscore_{metric}"] = self._get_ranking(scores)
        
        # Reward model rankings
        if reward_scores:
            rankings["reward_model"] = self._get_ranking(reward_scores)
        
        return rankings
    
    def _get_ranking(self, scores: List[float]) -> List[int]:
        """Get ranking of items based on scores (higher is better)."""
        # Create list of (score, index) pairs
        indexed_scores = [(score, i) for i, score in enumerate(scores)]
        
        # Sort by score in descending order
        indexed_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Create ranking array
        ranking = [0] * len(scores)
        for rank, (_, original_index) in enumerate(indexed_scores):
            ranking[original_index] = rank + 1
        
        return ranking
    
    def compare_rankings(self, rankings: Dict[str, List[int]]) -> Dict[str, float]:
        """Compare rankings between different metrics."""
        try:
            from scipy.stats import kendalltau, spearmanr
            
            comparison_results = {}
            
            # Get all metric names
            metrics = list(rankings.keys())
            
            # Compare each pair of metrics
            for i, metric1 in enumerate(metrics):
                for metric2 in metrics[i+1:]:
                    ranking1 = rankings[metric1]
                    ranking2 = rankings[metric2]
                    
                    if len(ranking1) == len(ranking2):
                        # Kendall's tau
                        tau, _ = kendalltau(ranking1, ranking2)
                        comparison_results[f"{metric1}_vs_{metric2}_kendall"] = tau
                        
                        # Spearman correlation
                        rho, _ = spearmanr(ranking1, ranking2)
                        comparison_results[f"{metric1}_vs_{metric2}_spearman"] = rho
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing rankings: {e}")
            return {}
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("SUMMARY EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {evaluation_results.get('timestamp', 'Unknown')}")
        report.append(f"Number of summaries: {evaluation_results.get('num_summaries', 0)}")
        report.append("")
        
        # ROUGE scores
        if "rouge_scores" in evaluation_results:
            report.append("ROUGE SCORES:")
            report.append("-" * 20)
            for metric, score in evaluation_results["rouge_scores"].items():
                if isinstance(score, (int, float)):
                    report.append(f"{metric}: {score:.4f}")
                else:
                    report.append(f"{metric}: {score}")
            report.append("")
        
        # BERTScore
        if "bertscore_scores" in evaluation_results:
            report.append("BERTSCORE:")
            report.append("-" * 15)
            bert = evaluation_results["bertscore_scores"]
            if "precision" in bert:
                report.append(f"Precision: {bert['precision']:.4f}")
                report.append(f"Recall: {bert['recall']:.4f}")
                report.append(f"F1: {bert['f1']:.4f}")
            report.append("")
        
        # Reward scores
        if "reward_scores" in evaluation_results and evaluation_results["reward_scores"]:
            report.append("REWARD MODEL SCORES:")
            report.append("-" * 25)
            scores = evaluation_results["reward_scores"]
            report.append(f"Mean: {np.mean(scores):.4f}")
            report.append(f"Std: {np.std(scores):.4f}")
            report.append(f"Min: {np.min(scores):.4f}")
            report.append(f"Max: {np.max(scores):.4f}")
            report.append("")
        
        # Correlations
        if "correlations" in evaluation_results and evaluation_results["correlations"]:
            report.append("METRIC CORRELATIONS:")
            report.append("-" * 30)
            for metric, corr in evaluation_results["correlations"].items():
                report.append(f"{metric}: {corr:.4f}")
            report.append("")
        
        # Ranking comparisons
        if "rankings" in evaluation_results:
            ranking_comparisons = self.compare_rankings(evaluation_results["rankings"])
            if ranking_comparisons:
                report.append("RANKING COMPARISONS:")
                report.append("-" * 25)
                for comparison, value in ranking_comparisons.items():
                    report.append(f"{comparison}: {value:.4f}")
                report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save evaluation results to file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            
            save_json(serializable_results, filepath)
            self.logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

class ComparativeAnalyzer:
    """Analyzes and compares different summarization approaches."""
    
    def __init__(self):
        self.logger = setup_logging("INFO")
    
    def analyze_summary_quality(self, summaries: List[Dict[str, Any]], 
                              references: List[str]) -> Dict[str, Any]:
        """Analyze quality aspects of summaries."""
        analysis = {
            "length_analysis": self._analyze_lengths(summaries),
            "diversity_analysis": self._analyze_diversity(summaries),
            "coherence_analysis": self._analyze_coherence(summaries),
            "coverage_analysis": self._analyze_coverage(summaries, references)
        }
        
        return analysis
    
    def _analyze_lengths(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze length characteristics of summaries."""
        lengths = [len(summary.get('text', '').split()) for summary in summaries]
        
        return {
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "length_distribution": lengths
        }
    
    def _analyze_diversity(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze diversity of summaries."""
        # Group by strategy
        strategy_groups = {}
        for summary in summaries:
            strategy = summary.get('strategy', 'unknown')
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(summary['text'])
        
        # Compute diversity metrics
        diversity_metrics = {}
        for strategy, texts in strategy_groups.items():
            if len(texts) > 1:
                # Simple diversity metric based on unique words
                all_words = set()
                for text in texts:
                    all_words.update(text.lower().split())
                
                unique_word_ratio = len(all_words) / sum(len(text.split()) for text in texts)
                diversity_metrics[strategy] = unique_word_ratio
        
        return {
            "strategy_distribution": {k: len(v) for k, v in strategy_groups.items()},
            "diversity_metrics": diversity_metrics
        }
    
    def _analyze_coherence(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze coherence of summaries (simplified)."""
        coherence_scores = []
        
        for summary in summaries:
            text = summary.get('text', '')
            # Simple coherence metric based on sentence structure
            sentences = text.split('.')
            if len(sentences) > 1:
                # Check for proper sentence endings and transitions
                coherence_score = min(1.0, len([s for s in sentences if s.strip()]) / len(sentences))
            else:
                coherence_score = 0.5
            
            coherence_scores.append(coherence_score)
        
        return {
            "mean_coherence": np.mean(coherence_scores),
            "coherence_scores": coherence_scores
        }
    
    def _analyze_coverage(self, summaries: List[Dict[str, Any]], 
                         references: List[str]) -> Dict[str, Any]:
        """Analyze coverage of key information."""
        coverage_scores = []
        
        for i, summary in enumerate(summaries):
            if i < len(references):
                # Simple coverage based on word overlap
                summary_words = set(summary.get('text', '').lower().split())
                reference_words = set(references[i].lower().split())
                
                if reference_words:
                    coverage = len(summary_words.intersection(reference_words)) / len(reference_words)
                else:
                    coverage = 0.0
                
                coverage_scores.append(coverage)
        
        return {
            "mean_coverage": np.mean(coverage_scores) if coverage_scores else 0.0,
            "coverage_scores": coverage_scores
        }
    
    def generate_comparative_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comparative analysis report."""
        report = []
        report.append("=" * 80)
        report.append("COMPARATIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {format_timestamp()}")
        report.append("")
        
        # Length analysis
        if "length_analysis" in analysis_results:
            length = analysis_results["length_analysis"]
            report.append("LENGTH ANALYSIS:")
            report.append("-" * 20)
            report.append(f"Mean length: {length['mean_length']:.1f} words")
            report.append(f"Std deviation: {length['std_length']:.1f} words")
            report.append(f"Range: {length['min_length']:.0f} - {length['max_length']:.0f} words")
            report.append("")
        
        # Diversity analysis
        if "diversity_analysis" in analysis_results:
            diversity = analysis_results["diversity_analysis"]
            report.append("DIVERSITY ANALYSIS:")
            report.append("-" * 22)
            report.append("Strategy distribution:")
            for strategy, count in diversity["strategy_distribution"].items():
                report.append(f"  {strategy}: {count} summaries")
            report.append("")
        
        # Coherence analysis
        if "coherence_analysis" in analysis_results:
            coherence = analysis_results["coherence_analysis"]
            report.append("COHERENCE ANALYSIS:")
            report.append("-" * 23)
            report.append(f"Mean coherence: {coherence['mean_coherence']:.3f}")
            report.append("")
        
        # Coverage analysis
        if "coverage_analysis" in analysis_results:
            coverage = analysis_results["coverage_analysis"]
            report.append("COVERAGE ANALYSIS:")
            report.append("-" * 21)
            report.append(f"Mean coverage: {coverage['mean_coverage']:.3f}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
