"""
Example script demonstrating how to use the multimodal summarization pipeline.
This script shows different ways to run the pipeline components.
"""

import json
import logging
from pathlib import Path

from main import MultimodalSummarizationPipeline

def example_full_pipeline():
    """Example: Run the complete pipeline."""
    print("=" * 60)
    print("EXAMPLE: Full Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = MultimodalSummarizationPipeline()
    
    # Run full pipeline with small dataset for demo
    results = pipeline.run_full_pipeline(num_papers=5, num_eval_papers=3)
    
    print(f"Pipeline completed: {results['success']}")
    if results['success']:
        print(f"Papers collected: {results['papers_collected']}")
        print(f"Summaries generated: {results['summaries_generated']}")
        print(f"Annotations collected: {results['annotations_collected']}")
        print(f"Training examples: {results['training_examples']}")
    
    return results

def example_step_by_step():
    """Example: Run pipeline step by step."""
    print("=" * 60)
    print("EXAMPLE: Step-by-Step Pipeline")
    print("=" * 60)
    
    pipeline = MultimodalSummarizationPipeline()
    
    # Step 1: Collect papers
    print("Step 1: Collecting papers...")
    papers = pipeline.run_data_collection_only(num_papers=3)
    print(f"Collected {len(papers)} papers")
    
    # Save papers for next step
    with open("temp_papers.json", "w") as f:
        json.dump(papers, f, indent=2, default=str)
    
    # Step 2: Generate summaries
    print("Step 2: Generating summaries...")
    summaries = pipeline.run_summarization_only(papers)
    print(f"Generated {len(summaries)} summaries")
    
    # Step 3: Create summary pairs
    print("Step 3: Creating summary pairs...")
    summary_pairs = pipeline.summary_generator.create_summary_pairs(summaries)
    print(f"Created {len(summary_pairs)} summary pairs")
    
    # Save summary pairs for annotation
    with open("temp_summary_pairs.json", "w") as f:
        json.dump(summary_pairs, f, indent=2, default=str)
    
    print("Summary pairs saved to temp_summary_pairs.json")
    print("You can now run annotation manually or continue with the pipeline")
    
    return papers, summaries, summary_pairs

def example_custom_configuration():
    """Example: Run with custom configuration."""
    print("=" * 60)
    print("EXAMPLE: Custom Configuration")
    print("=" * 60)
    
    # Note: Configuration is now hardcoded in individual modules
    # To customize, you would modify the values directly in the module files
    
    # Initialize pipeline
    pipeline = MultimodalSummarizationPipeline()
    
    # Run with custom settings
    papers = pipeline.run_data_collection_only(num_papers=3)
    print(f"Collected {len(papers)} papers")
    print("Note: To customize configuration, modify values in individual module files")
    
    return papers

def example_evaluation_only():
    """Example: Run evaluation on existing data."""
    print("=" * 60)
    print("EXAMPLE: Evaluation Only")
    print("=" * 60)
    
    # Create sample data for evaluation
    predictions = [
        "This paper presents a novel approach to natural language processing using transformer models.",
        "The research introduces a new method for text summarization with improved accuracy.",
        "This work demonstrates significant improvements in machine learning model performance."
    ]
    
    references = [
        "This paper introduces a new transformer-based approach for natural language processing tasks.",
        "We present a novel text summarization method that achieves better performance than existing approaches.",
        "Our research shows substantial improvements in machine learning model accuracy and efficiency."
    ]
    
    # Initialize pipeline
    pipeline = MultimodalSummarizationPipeline()
    
    # Run evaluation
    results = pipeline.run_evaluation_only(predictions, references)
    
    print("Evaluation Results:")
    if "rouge_scores" in results:
        print("ROUGE Scores:")
        for metric, score in results["rouge_scores"].items():
            if isinstance(score, (int, float)):
                print(f"  {metric}: {score:.4f}")
    
    if "bertscore_scores" in results:
        print("BERTScore:")
        bert = results["bertscore_scores"]
        print(f"  F1: {bert.get('f1', 0):.4f}")
    
    return results

def example_annotation_interface():
    """Example: Demonstrate annotation interface."""
    print("=" * 60)
    print("EXAMPLE: Annotation Interface")
    print("=" * 60)
    
    # Create sample summary pairs
    summary_pairs = [
        {
            "pair_id": "demo_pair_1",
            "paper_id": "demo_paper_1",
            "paper_title": "Demo Paper 1",
            "summary_1": {
                "text": "This is a concise summary of the paper.",
                "strategy": "concise",
                "length": 50
            },
            "summary_2": {
                "text": "This is a more detailed and comprehensive summary that covers all aspects of the paper including methodology, results, and implications.",
                "strategy": "comprehensive",
                "length": 120
            }
        }
    ]
    
    # Initialize annotation interface
    from annotation_interface import CommandLineAnnotationInterface
    annotation_interface = CommandLineAnnotationInterface()
    
    print("Annotation interface ready. In a real scenario, you would:")
    print("1. Load your summary pairs")
    print("2. Run the annotation interface")
    print("3. Collect human preferences")
    print("4. Save annotations for training")
    
    return summary_pairs

def cleanup_temp_files():
    """Clean up temporary files created during examples."""
    temp_files = ["temp_papers.json", "temp_summary_pairs.json"]
    for file in temp_files:
        if Path(file).exists():
            Path(file).unlink()
            print(f"Cleaned up {file}")

def main():
    """Run all examples."""
    print("Multimodal Summarization Pipeline - Examples")
    print("=" * 60)
    
    try:
        # Example 1: Step-by-step pipeline
        print("\n1. Step-by-Step Pipeline Example")
        papers, summaries, summary_pairs = example_step_by_step()
        
        # Example 2: Custom configuration
        print("\n2. Custom Configuration Example")
        example_custom_configuration()
        
        # Example 3: Evaluation only
        print("\n3. Evaluation Only Example")
        example_evaluation_only()
        
        # Example 4: Annotation interface
        print("\n4. Annotation Interface Example")
        example_annotation_interface()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
        # Note about full pipeline
        print("\nNOTE: To run the full pipeline, use:")
        print("python main.py --mode full --num-papers 10 --num-eval-papers 10")
        print("\nFor more options, see:")
        print("python main.py --help")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        logging.error(f"Example execution failed: {e}")
    
    finally:
        # Clean up
        cleanup_temp_files()

if __name__ == "__main__":
    main()
