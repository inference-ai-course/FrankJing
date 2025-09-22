# Multimodal Summarization and Reward Modeling

A comprehensive system for generating and evaluating academic paper summaries using LLaMA 3, with human preference-based reward modeling for improved evaluation metrics.

## Overview

This project implements a complete pipeline for:
- Collecting academic papers from arXiv
- Generating diverse summaries using LLaMA 3 with different prompting strategies
- Collecting human preferences through annotation interfaces
- Training a reward model (DeBERTa-v3) on preference data
- Evaluating summaries using ROUGE, BERTScore, and the trained reward model
- Comparing different evaluation metrics and their correlations

## Architecture

The system is built with a modular architecture:

```
├── config.py                 # Configuration management
├── utils.py                  # Utility functions
├── data_collector.py         # Paper collection and preprocessing
├── summary_generator.py      # LLaMA 3 summarization
├── annotation_interface.py   # Human annotation interfaces
├── reward_trainer.py         # Reward model training
├── evaluator.py              # Evaluation metrics and analysis
├── main.py                   # Main pipeline orchestration
└── requirements.txt          # Dependencies
```

## Features

### Data Collection
- Automated arXiv paper collection with search queries
- PDF text extraction and section parsing
- Image extraction from PDFs for multimodal processing
- Support for various academic paper formats

### Summary Generation
- Multiple prompting strategies for diverse summaries
- LLaMA 3 (7B) integration with quantization support
- Configurable generation parameters
- Multimodal content incorporation

### Human Annotation
- Command-line annotation interface
- Web-based annotation interface
- Batch processing for multiple annotators
- Preference data collection and formatting

### Reward Model Training
- DeBERTa-v3 based reward model
- TRL (Transformers Reinforcement Learning) integration
- Preference-based training on human annotations
- Comprehensive evaluation metrics

### Evaluation
- ROUGE score computation
- BERTScore evaluation
- Reward model scoring
- Comparative analysis and correlation studies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multimodal-summarization-reward-modeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
export CUDA_VISIBLE_DEVICES=0  # For GPU usage
export WANDB_API_KEY=your_key  # For experiment tracking
```

## Usage

### Full Pipeline

Run the complete pipeline with default settings:

```bash
python main.py --mode full --num-papers 10 --num-eval-papers 10
```

### Individual Components

#### Data Collection Only
```bash
python main.py --mode data --num-papers 20
```

#### Summarization Only
```bash
python main.py --mode summarize --input-file papers.json
```

#### Annotation Only
```bash
python main.py --mode annotate --input-file summary_pairs.json
```

#### Reward Model Training Only
```bash
python main.py --mode train --input-file training_data.json
```

#### Evaluation Only
```bash
python main.py --mode evaluate --input-file eval_data.json
```

### Configuration

Modify `config.py` to adjust:
- Model parameters (LLaMA 3, DeBERTa-v3)
- Training hyperparameters
- Data processing settings
- Evaluation metrics

### Customization

#### Adding New Prompting Strategies

```python
# In summary_generator.py
new_strategy = {
    "name": "custom_strategy",
    "template": "Your custom prompt template: {content}",
    "generation_config": {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_new_tokens": 300
    }
}
self.prompt_strategies.append(new_strategy)
```

#### Custom Evaluation Metrics

```python
# In evaluator.py
def custom_metric(self, predictions, references):
    # Your custom metric implementation
    return scores
```

## Output Structure

The pipeline generates:

```
outputs/
├── papers/                   # Collected papers
├── summaries/               # Generated summaries
├── annotations/             # Human annotations
├── models/                  # Trained reward model
└── reports/                 # Evaluation reports
    ├── pipeline_results_*.json
    ├── evaluation_report_*.txt
    └── pipeline_summary_*.txt
```

## Example Workflow

1. **Collect Papers**: Download 10 academic papers from arXiv
2. **Generate Summaries**: Create 2 diverse summaries per paper using different strategies
3. **Human Annotation**: Collect preferences between summary pairs
4. **Train Reward Model**: Fine-tune DeBERTa-v3 on preference data
5. **Evaluate**: Test on new papers and compare metrics

## Configuration Options

### Model Configuration
- `llama_model_name`: LLaMA 3 model identifier
- `reward_model_name`: DeBERTa-v3 model identifier
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for training

### Data Configuration
- `num_papers`: Number of papers to collect
- `num_eval_papers`: Number of evaluation papers
- `max_paper_length`: Maximum paper text length
- `max_summary_length`: Maximum summary length

### Evaluation Configuration
- `rouge_metrics`: ROUGE metrics to compute
- `bertscore_lang`: Language for BERTScore
- `bertscore_model_type`: BERTScore model type

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check model names and availability
3. **Annotation Interface Issues**: Ensure proper file paths and permissions

### Performance Optimization

1. **GPU Usage**: Set `CUDA_VISIBLE_DEVICES` environment variable
2. **Memory Management**: Use quantization for large models
3. **Batch Processing**: Adjust batch sizes based on available memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_summarization_reward_modeling,
  title={Multimodal Summarization and Reward Modeling},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/multimodal-summarization-reward-modeling}
}
```

## Acknowledgments

- Hugging Face for transformers and datasets libraries
- Meta for LLaMA 3 model
- Microsoft for DeBERTa-v3 model
- arXiv for providing academic papers
- The open-source community for various tools and libraries
