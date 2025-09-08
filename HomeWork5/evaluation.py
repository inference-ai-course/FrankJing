import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from hybrid_search import HybridSearchSystem
import matplotlib.pyplot as plt
import seaborn as sns

class SearchEvaluator:
    def __init__(self, search_system: HybridSearchSystem):
        self.search_system = search_system
        self.test_queries = []
        self.relevant_docs = {}
        
    def load_test_data(self):
        """Load test queries and their relevant documents."""
        # Create comprehensive test dataset
        test_documents = [
            {
                'title': 'Deep Learning with Neural Networks',
                'author': 'Alice Johnson',
                'year': 2023,
                'keywords': 'deep learning, neural networks, backpropagation, gradient descent',
                'chunk_text': 'Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. These networks can learn complex patterns through backpropagation and gradient descent optimization.'
            },
            {
                'title': 'Introduction to Machine Learning Algorithms',
                'author': 'Bob Smith',
                'year': 2022,
                'keywords': 'machine learning, algorithms, supervised learning, unsupervised learning',
                'chunk_text': 'Machine learning encompasses various algorithms for pattern recognition and prediction. Supervised learning uses labeled data, while unsupervised learning finds hidden patterns in unlabeled datasets.'
            },
            {
                'title': 'Natural Language Processing Fundamentals',
                'author': 'Carol Davis',
                'year': 2024,
                'keywords': 'NLP, natural language processing, text analysis, language models',
                'chunk_text': 'Natural language processing enables computers to understand and generate human language. Modern NLP relies heavily on transformer models and attention mechanisms for text understanding.'
            },
            {
                'title': 'Computer Vision and Image Recognition',
                'author': 'David Wilson',
                'year': 2023,
                'keywords': 'computer vision, image recognition, CNN, convolutional neural networks',
                'chunk_text': 'Computer vision uses convolutional neural networks to analyze and understand visual information. CNNs excel at feature extraction and pattern recognition in images.'
            },
            {
                'title': 'Reinforcement Learning Principles',
                'author': 'Emma Brown',
                'year': 2022,
                'keywords': 'reinforcement learning, Q-learning, policy gradient, reward systems',
                'chunk_text': 'Reinforcement learning trains agents to make decisions through trial and error. Q-learning and policy gradient methods are popular approaches for optimizing sequential decision making.'
            },
            {
                'title': 'Data Science and Statistical Analysis',
                'author': 'Frank Lee',
                'year': 2023,
                'keywords': 'data science, statistics, data analysis, statistical modeling',
                'chunk_text': 'Data science combines statistical analysis with computational methods to extract insights from data. Statistical modeling helps understand relationships and make predictions from complex datasets.'
            },
            {
                'title': 'Artificial Intelligence Ethics and Bias',
                'author': 'Grace Chen',
                'year': 2024,
                'keywords': 'AI ethics, bias, fairness, responsible AI, algorithmic transparency',
                'chunk_text': 'AI ethics addresses bias, fairness, and transparency in artificial intelligence systems. Responsible AI development requires careful consideration of algorithmic bias and societal impact.'
            },
            {
                'title': 'Time Series Analysis and Forecasting',
                'author': 'Henry Taylor',
                'year': 2023,
                'keywords': 'time series, forecasting, temporal data, LSTM, recurrent networks',
                'chunk_text': 'Time series analysis involves studying temporal data patterns for forecasting. LSTM and other recurrent neural networks are effective for modeling sequential dependencies in time-based data.'
            },
            {
                'title': 'Clustering and Dimensionality Reduction',
                'author': 'Iris Wang',
                'year': 2022,
                'keywords': 'clustering, dimensionality reduction, PCA, k-means, unsupervised learning',
                'chunk_text': 'Clustering algorithms like k-means group similar data points together. Dimensionality reduction techniques such as PCA help visualize and analyze high-dimensional data.'
            },
            {
                'title': 'Transfer Learning and Pre-trained Models',
                'author': 'Jack Miller',
                'year': 2024,
                'keywords': 'transfer learning, pre-trained models, fine-tuning, feature extraction',
                'chunk_text': 'Transfer learning leverages pre-trained models to solve new tasks with limited data. Fine-tuning and feature extraction are common approaches to adapt existing models for specific domains.'
            }
        ]
        
        # Add documents to search system
        self.search_system.add_documents(test_documents)
        
        # Define test queries with relevant document IDs
        self.test_queries = [
            {
                'query': 'neural networks deep learning',
                'relevant_docs': [1, 4],  # Deep Learning, Computer Vision
                'description': 'General neural networks query'
            },
            {
                'query': 'machine learning algorithms supervised',
                'relevant_docs': [2],  # Introduction to ML Algorithms
                'description': 'Specific ML algorithms query'
            },
            {
                'query': 'natural language processing text',
                'relevant_docs': [3],  # NLP Fundamentals
                'description': 'NLP-specific query'
            },
            {
                'query': 'CNN convolutional networks images',
                'relevant_docs': [4],  # Computer Vision
                'description': 'Computer vision query'
            },
            {
                'query': 'reinforcement learning Q-learning',
                'relevant_docs': [5],  # RL Principles
                'description': 'RL-specific query'
            },
            {
                'query': 'statistical analysis data science',
                'relevant_docs': [6],  # Data Science
                'description': 'Statistics and data science query'
            },
            {
                'query': 'AI ethics bias fairness',
                'relevant_docs': [7],  # AI Ethics
                'description': 'AI ethics query'
            },
            {
                'query': 'time series forecasting LSTM',
                'relevant_docs': [8],  # Time Series
                'description': 'Time series analysis query'
            },
            {
                'query': 'clustering k-means PCA',
                'relevant_docs': [9],  # Clustering
                'description': 'Clustering and dimensionality reduction'
            },
            {
                'query': 'transfer learning fine-tuning models',
                'relevant_docs': [10],  # Transfer Learning
                'description': 'Transfer learning query'
            },
            {
                'query': 'backpropagation gradient descent',
                'relevant_docs': [1],  # Deep Learning
                'description': 'Specific training algorithm query'
            },
            {
                'query': 'pattern recognition unsupervised',
                'relevant_docs': [2, 9],  # ML Algorithms, Clustering
                'description': 'Broader pattern recognition query'
            }
        ]
    
    def calculate_recall_at_k(self, retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """Calculate recall@k metric."""
        if not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_retrieved = len(set(top_k) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_precision_at_k(self, retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """Calculate precision@k metric."""
        if k == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        if not top_k:
            return 0.0
        
        relevant_retrieved = len(set(top_k) & set(relevant_docs))
        return relevant_retrieved / len(top_k)
    
    def calculate_ndcg_at_k(self, retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """Calculate NDCG@k metric."""
        def dcg_at_k(relevance_scores, k):
            if k > len(relevance_scores):
                k = len(relevance_scores)
            dcg = 0
            for i in range(k):
                dcg += relevance_scores[i] / np.log2(i + 2)
            return dcg
        
        # Create relevance scores for retrieved docs
        relevance_scores = []
        for doc_id in retrieved_docs[:k]:
            if doc_id in relevant_docs:
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.0)
        
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (perfect ranking)
        ideal_relevance = [1.0] * min(len(relevant_docs), k) + [0.0] * max(0, k - len(relevant_docs))
        idcg = dcg_at_k(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_search_method(self, method_name: str, search_function, k_values: List[int] = [1, 3, 5]) -> Dict:
        """Evaluate a specific search method."""
        results = {
            'method': method_name,
            'recall': {f'recall@{k}': [] for k in k_values},
            'precision': {f'precision@{k}': [] for k in k_values},
            'ndcg': {f'ndcg@{k}': [] for k in k_values},
            'query_results': []
        }
        
        for test_case in self.test_queries:
            query = test_case['query']
            relevant_docs = test_case['relevant_docs']
            
            # Get search results
            search_results = search_function(query, max(k_values) * 2)
            retrieved_docs = [doc_id for doc_id, _ in search_results]
            
            query_result = {
                'query': query,
                'relevant_docs': relevant_docs,
                'retrieved_docs': retrieved_docs[:max(k_values)],
                'description': test_case['description']
            }
            
            # Calculate metrics for each k
            for k in k_values:
                recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k)
                precision = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k)
                ndcg = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, k)
                
                results['recall'][f'recall@{k}'].append(recall)
                results['precision'][f'precision@{k}'].append(precision)
                results['ndcg'][f'ndcg@{k}'].append(ndcg)
                
                query_result[f'recall@{k}'] = recall
                query_result[f'precision@{k}'] = precision
                query_result[f'ndcg@{k}'] = ndcg
            
            results['query_results'].append(query_result)
        
        # Calculate average metrics
        for metric_type in ['recall', 'precision', 'ndcg']:
            for metric_name in results[metric_type]:
                values = results[metric_type][metric_name]
                results[metric_type][metric_name] = {
                    'values': values,
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return results
    
    def run_full_evaluation(self) -> Dict:
        """Run evaluation on all search methods."""
        print("Loading test data...")
        self.load_test_data()
        
        print("Running evaluations...")
        evaluation_results = {}
        
        # Vector search
        print("Evaluating vector search...")
        evaluation_results['vector'] = self.evaluate_search_method(
            'Vector Search',
            lambda q, k: self.search_system.vector_search(q, k)
        )
        
        # Keyword search
        print("Evaluating keyword search...")
        evaluation_results['keyword'] = self.evaluate_search_method(
            'Keyword Search',
            lambda q, k: self.search_system.keyword_search(q, k)
        )
        
        # Weighted hybrid search (different alpha values)
        for alpha in [0.3, 0.5, 0.7]:
            print(f"Evaluating weighted hybrid search (alpha={alpha})...")
            evaluation_results[f'weighted_hybrid_{alpha}'] = self.evaluate_search_method(
                f'Weighted Hybrid (α={alpha})',
                lambda q, k, a=alpha: self.search_system.weighted_hybrid_search(q, k, a)
            )
        
        # RRF hybrid search
        print("Evaluating RRF hybrid search...")
        evaluation_results['rrf_hybrid'] = self.evaluate_search_method(
            'RRF Hybrid',
            lambda q, k: self.search_system.rrf_hybrid_search(q, k)
        )
        
        return evaluation_results
    
    def create_comparison_report(self, results: Dict) -> pd.DataFrame:
        """Create a comparison report of all methods."""
        comparison_data = []
        
        for method_key, method_results in results.items():
            method_name = method_results['method']
            
            row = {'Method': method_name}
            
            # Add recall metrics
            for recall_metric in method_results['recall']:
                row[recall_metric] = f"{method_results['recall'][recall_metric]['mean']:.3f} ± {method_results['recall'][recall_metric]['std']:.3f}"
            
            # Add precision metrics
            for precision_metric in method_results['precision']:
                row[precision_metric] = f"{method_results['precision'][precision_metric]['mean']:.3f} ± {method_results['precision'][precision_metric]['std']:.3f}"
            
            # Add NDCG metrics
            for ndcg_metric in method_results['ndcg']:
                row[ndcg_metric] = f"{method_results['ndcg'][ndcg_metric]['mean']:.3f} ± {method_results['ndcg'][ndcg_metric]['std']:.3f}"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_performance_comparison(self, results: Dict, save_path: str = None):
        """Create performance comparison plots."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        methods = []
        recall_3_scores = []
        precision_3_scores = []
        ndcg_3_scores = []
        
        for method_key, method_results in results.items():
            methods.append(method_results['method'])
            recall_3_scores.append(method_results['recall']['recall@3']['mean'])
            precision_3_scores.append(method_results['precision']['precision@3']['mean'])
            ndcg_3_scores.append(method_results['ndcg']['ndcg@3']['mean'])
        
        # Recall@3 comparison
        axes[0].bar(methods, recall_3_scores, color='skyblue', alpha=0.7)
        axes[0].set_title('Recall@3 Comparison')
        axes[0].set_ylabel('Recall@3')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0, 1)
        
        # Precision@3 comparison
        axes[1].bar(methods, precision_3_scores, color='lightcoral', alpha=0.7)
        axes[1].set_title('Precision@3 Comparison')
        axes[1].set_ylabel('Precision@3')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1)
        
        # NDCG@3 comparison
        axes[2].bar(methods, ndcg_3_scores, color='lightgreen', alpha=0.7)
        axes[2].set_title('NDCG@3 Comparison')
        axes[2].set_ylabel('NDCG@3')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_detailed_results(self, results: Dict, filename: str = "evaluation_results.json"):
        """Save detailed evaluation results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {filename}")


def main():
    """Main evaluation function."""
    # Initialize search system and evaluator
    search_system = HybridSearchSystem()
    evaluator = SearchEvaluator(search_system)
    
    # Run full evaluation
    results = evaluator.run_full_evaluation()
    
    # Create and display comparison report
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    comparison_df = evaluator.create_comparison_report(results)
    print(comparison_df.to_string(index=False))
    
    # Save results
    evaluator.save_detailed_results(results)
    
    # Create performance plots
    print("\nGenerating performance comparison plots...")
    evaluator.plot_performance_comparison(results, "performance_comparison.png")
    
    # Print best performing methods
    print("\n" + "="*80)
    print("BEST PERFORMING METHODS")
    print("="*80)
    
    best_recall = max(results.keys(), key=lambda k: results[k]['recall']['recall@3']['mean'])
    best_precision = max(results.keys(), key=lambda k: results[k]['precision']['precision@3']['mean'])
    best_ndcg = max(results.keys(), key=lambda k: results[k]['ndcg']['ndcg@3']['mean'])
    
    print(f"Best Recall@3: {results[best_recall]['method']} ({results[best_recall]['recall']['recall@3']['mean']:.3f})")
    print(f"Best Precision@3: {results[best_precision]['method']} ({results[best_precision]['precision']['precision@3']['mean']:.3f})")
    print(f"Best NDCG@3: {results[best_ndcg]['method']} ({results[best_ndcg]['ndcg']['ndcg@3']['mean']:.3f})")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()