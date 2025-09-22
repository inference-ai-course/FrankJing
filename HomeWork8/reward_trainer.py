"""
Reward model training pipeline using DeBERTa-v3 and TRL.
"""

import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
from trl import RewardTrainer, RewardConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import (
    setup_logging, 
    get_device, 
    log_model_info, 
    save_checkpoint, 
    load_checkpoint,
    format_timestamp,
    set_random_seeds
)

class RewardModelTrainer:
    """Trains a reward model on preference data using DeBERTa-v3."""
    
    def __init__(self):
        self.logger = setup_logging("INFO")
        self.device = get_device()
        
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Model configuration
        self.reward_model_name = "microsoft/deberta-v3-base"
        self.reward_model_num_labels = 1
        self.batch_size = 8
        self.num_epochs = 3
        self.learning_rate = 5e-5
        self.warmup_steps = 100
        self.logging_steps = 10
        self.save_steps = 500
        self.eval_steps = 500
        self.fp16 = True
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        self._load_model()
        
    def _load_model(self):
        """Load DeBERTa-v3 model and tokenizer."""
        try:
            self.logger.info(f"Loading reward model: {self.reward_model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.reward_model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_model_name,
                num_labels=self.reward_model_num_labels,
                trust_remote_code=True
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            log_model_info(self.model, self.logger)
            self.logger.info("Reward model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading reward model: {e}")
            raise
    
    def prepare_dataset(self, training_data: List[Dict[str, Any]], 
                       validation_split: float = 0.1) -> Tuple[Dataset, Dataset]:
        """Prepare dataset for training."""
        self.logger.info(f"Preparing dataset with {len(training_data)} examples")
        
        # Split into train and validation
        np.random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - validation_split))
        
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        self.logger.info(f"Train examples: {len(train_data)}, Validation examples: {len(val_data)}")
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            self._tokenize_function, 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            self._tokenize_function, 
            batched=True, 
            remove_columns=val_dataset.column_names
        )
        
        return train_dataset, val_dataset
    
    def _tokenize_function(self, examples):
        """Tokenize examples for reward model training."""
        # Tokenize chosen and rejected texts
        chosen_tokens = self.tokenizer(
            examples["chosen"],
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            examples["rejected"],
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create input_ids and attention_mask for both
        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"]
        }
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              output_dir: str = "reward_model") -> None:
        """Train the reward model."""
        try:
            self.logger.info("Starting reward model training")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Training arguments
            training_args = RewardConfig(
                output_dir=output_dir,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                warmup_steps=self.warmup_steps,
                logging_steps=self.logging_steps,
                save_steps=self.save_steps,
                eval_steps=self.eval_steps,
                evaluation_strategy="steps" if val_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=self.fp16 and self.device.type == "cuda",
                dataloader_drop_last=True,
                report_to=None,  # Disable wandb/tensorboard
                seed=42
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                max_length=512
            )
            
            # Create trainer
            self.trainer = RewardTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Train
            self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            self.logger.info(f"Training completed. Model saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the reward model."""
        if not self.trainer:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            self.logger.info("Evaluating reward model")
            
            # Get predictions
            eval_results = self.trainer.evaluate(eval_dataset)
            
            self.logger.info(f"Evaluation results: {eval_results}")
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {}
    
    def predict_reward(self, chosen_text: str, rejected_text: str) -> Tuple[float, float]:
        """Predict reward scores for chosen and rejected texts."""
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Tokenize texts
                chosen_tokens = self.tokenizer(
                    chosen_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                rejected_tokens = self.tokenizer(
                    rejected_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Get predictions
                chosen_logits = self.model(**chosen_tokens).logits
                rejected_logits = self.model(**rejected_tokens).logits
                
                chosen_score = chosen_logits.item()
                rejected_score = rejected_logits.item()
                
                return chosen_score, rejected_score
                
        except Exception as e:
            self.logger.error(f"Error predicting rewards: {e}")
            return 0.0, 0.0
    
    def score_summaries(self, summaries: List[str]) -> List[float]:
        """Score a list of summaries using the reward model."""
        scores = []
        
        for summary in summaries:
            try:
                self.model.eval()
                
                with torch.no_grad():
                    tokens = self.tokenizer(
                        summary,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    ).to(self.device)
                    
                    logits = self.model(**tokens).logits
                    score = logits.item()
                    scores.append(score)
                    
            except Exception as e:
                self.logger.error(f"Error scoring summary: {e}")
                scores.append(0.0)
        
        return scores
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model."""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            if self.trainer:
                self.trainer.save_model(output_dir)
            else:
                self.model.save_pretrained(output_dir)
            
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training metadata
            metadata = {
                "model_name": self.reward_model_name,
                "num_labels": self.reward_model_num_labels,
                "training_config": {
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "warmup_steps": self.warmup_steps
                },
                "timestamp": format_timestamp()
            }
            
            with open(Path(output_dir) / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_dir: str) -> None:
        """Load a trained model."""
        try:
            self.logger.info(f"Loading trained model from {model_dir}")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                trust_remote_code=True
            ).to(self.device)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            
            self.logger.info("Trained model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading trained model: {e}")
            raise

class RewardModelEvaluator:
    """Evaluates reward model performance and compares with other metrics."""
    
    def __init__(self, reward_model_trainer: RewardModelTrainer):
        self.reward_trainer = reward_model_trainer
        self.logger = setup_logging("INFO")
    
    def evaluate_preference_accuracy(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate how well the reward model predicts human preferences."""
        correct_predictions = 0
        total_predictions = 0
        
        for example in test_data:
            chosen_text = example["chosen"]
            rejected_text = example["rejected"]
            
            # Get reward scores
            chosen_score, rejected_score = self.reward_trainer.predict_reward(
                chosen_text, rejected_text
            )
            
            # Check if model agrees with human preference
            if chosen_score > rejected_score:
                correct_predictions += 1
            
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "preference_accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions
        }
    
    def compare_with_automatic_metrics(self, summaries: List[str], 
                                     references: List[str]) -> Dict[str, Any]:
        """Compare reward model scores with ROUGE and BERTScore."""
        try:
            from evaluate import load
            
            # Load metrics
            rouge = load("rouge")
            bertscore = load("bertscore")
            
            # Compute automatic metrics
            rouge_scores = rouge.compute(
                predictions=summaries, 
                references=references
            )
            
            bertscore_scores = bertscore.compute(
                predictions=summaries,
                references=references,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli"
            )
            
            # Compute reward model scores
            reward_scores = self.reward_trainer.score_summaries(summaries)
            
            # Normalize scores for comparison
            normalized_reward_scores = self._normalize_scores(reward_scores)
            
            return {
                "rouge_scores": rouge_scores,
                "bertscore_scores": {
                    "precision": bertscore_scores["precision"],
                    "recall": bertscore_scores["recall"],
                    "f1": bertscore_scores["f1"]
                },
                "reward_scores": reward_scores,
                "normalized_reward_scores": normalized_reward_scores,
                "correlations": self._compute_correlations(
                    rouge_scores, bertscore_scores, normalized_reward_scores
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing metrics: {e}")
            return {}
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _compute_correlations(self, rouge_scores: Dict, bertscore_scores: Dict, 
                            reward_scores: List[float]) -> Dict[str, float]:
        """Compute correlations between different metrics."""
        try:
            import numpy as np
            from scipy.stats import pearsonr, spearmanr
            
            correlations = {}
            
            # ROUGE correlations
            rouge_metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
            for metric in rouge_metrics:
                if metric in rouge_scores:
                    rouge_values = rouge_scores[metric]
                    if isinstance(rouge_values, list):
                        pearson_corr, _ = pearsonr(rouge_values, reward_scores)
                        spearman_corr, _ = spearmanr(rouge_values, reward_scores)
                        correlations[f"rouge_{metric}_pearson"] = pearson_corr
                        correlations[f"rouge_{metric}_spearman"] = spearman_corr
            
            # BERTScore correlations
            bertscore_f1 = bertscore_scores["f1"]
            if isinstance(bertscore_f1, list):
                pearson_corr, _ = pearsonr(bertscore_f1, reward_scores)
                spearman_corr, _ = spearmanr(bertscore_f1, reward_scores)
                correlations["bertscore_f1_pearson"] = pearson_corr
                correlations["bertscore_f1_spearman"] = spearman_corr
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error computing correlations: {e}")
            return {}
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("REWARD MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated at: {format_timestamp()}")
        report.append("")
        
        # Preference accuracy
        if "preference_accuracy" in evaluation_results:
            acc = evaluation_results["preference_accuracy"]
            report.append(f"Preference Accuracy: {acc:.3f}")
            report.append(f"Correct Predictions: {evaluation_results.get('correct_predictions', 0)}")
            report.append(f"Total Predictions: {evaluation_results.get('total_predictions', 0)}")
            report.append("")
        
        # Metric comparisons
        if "correlations" in evaluation_results:
            report.append("METRIC CORRELATIONS:")
            report.append("-" * 30)
            for metric, corr in evaluation_results["correlations"].items():
                report.append(f"{metric}: {corr:.3f}")
            report.append("")
        
        # ROUGE scores
        if "rouge_scores" in evaluation_results:
            report.append("ROUGE SCORES:")
            report.append("-" * 20)
            rouge = evaluation_results["rouge_scores"]
            for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                if metric in rouge:
                    report.append(f"{metric}: {rouge[metric]:.3f}")
            report.append("")
        
        # BERTScore
        if "bertscore_scores" in evaluation_results:
            report.append("BERTSCORE:")
            report.append("-" * 15)
            bert = evaluation_results["bertscore_scores"]
            report.append(f"Precision: {np.mean(bert['precision']):.3f}")
            report.append(f"Recall: {np.mean(bert['recall']):.3f}")
            report.append(f"F1: {np.mean(bert['f1']):.3f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
