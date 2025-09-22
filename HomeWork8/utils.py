"""
Utility functions for multimodal summarization and reward modeling project.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("multimodal_summarization")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    # Remove multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    return text.strip()

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to maximum length while preserving word boundaries."""
    if len(text) <= max_length:
        return text
    
    # Find the last complete sentence within the limit
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    last_space = truncated.rfind(' ')
    
    # Use the last complete sentence if possible
    if last_period > max_length * 0.8:  # If we have a sentence ending close to the limit
        return truncated[:last_period + 1]
    elif last_space > max_length * 0.8:  # Otherwise use word boundary
        return truncated[:last_space]
    else:
        return truncated

def extract_paper_sections(text: str) -> Dict[str, str]:
    """Extract different sections from a research paper."""
    sections = {
        'abstract': '',
        'introduction': '',
        'methodology': '',
        'results': '',
        'conclusion': '',
        'full_text': text
    }
    
    # Simple section extraction based on common patterns
    section_patterns = {
        'abstract': r'(?i)(?:abstract|summary)\s*:?\s*(.*?)(?=\n\s*(?:introduction|1\.|keywords|index terms))',
        'introduction': r'(?i)(?:introduction|1\.\s*introduction)\s*:?\s*(.*?)(?=\n\s*(?:2\.|methodology|related work|background))',
        'methodology': r'(?i)(?:methodology|methods?|approach|2\.)\s*:?\s*(.*?)(?=\n\s*(?:3\.|results|experiments?|evaluation))',
        'results': r'(?i)(?:results?|experiments?|evaluation|3\.)\s*:?\s*(.*?)(?=\n\s*(?:4\.|conclusion|discussion|future work))',
        'conclusion': r'(?i)(?:conclusion|discussion|4\.)\s*:?\s*(.*?)(?=\n\s*(?:references?|bibliography|acknowledgments?))'
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[section_name] = clean_text(match.group(1))
    
    return sections

def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """Calculate basic text statistics."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0
    }

def format_timestamp() -> str:
    """Get formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def log_model_info(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Log model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model device: {next(model.parameters()).device}")

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': format_timestamp()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def create_summary_pair(paper_id: str, summary1: str, summary2: str, 
                       preferred: int, annotator: str = "human") -> Dict[str, Any]:
    """Create a summary pair entry for reward model training."""
    return {
        "paper_id": paper_id,
        "chosen": summary1 if preferred == 1 else summary2,
        "rejected": summary2 if preferred == 1 else summary1,
        "preferred_index": preferred,
        "annotator": annotator,
        "timestamp": format_timestamp()
    }

def validate_summary_pair(pair: Dict[str, Any]) -> bool:
    """Validate a summary pair entry."""
    required_fields = ["paper_id", "chosen", "rejected", "preferred_index"]
    return all(field in pair for field in required_fields) and pair["chosen"] != pair["rejected"]

