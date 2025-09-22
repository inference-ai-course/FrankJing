"""
Summary generation module using LLaMA 3 for multimodal summarization.
"""

import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
import random
import json
from pathlib import Path

from utils import setup_logging, truncate_text, format_timestamp, get_device, log_model_info

class SummaryGenerator:
    """Generates diverse summaries using LLaMA 3 with different prompting strategies."""
    
    def __init__(self):
        self.logger = setup_logging("INFO")
        self.device = get_device()
        
        # Model configuration
        self.llama_model_name = "meta-llama/Llama-3.1-7B-Instruct"
        self.llama_max_length = 4096
        self.llama_temperature = 0.7
        self.llama_top_p = 0.9
        self.llama_num_return_sequences = 2
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # Summary generation strategies
        self.prompt_strategies = self._create_prompt_strategies()
        
    def _load_model(self):
        """Load LLaMA 3 model and tokenizer."""
        try:
            self.logger.info(f"Loading LLaMA 3 model: {self.llama_model_name}")
            
            # Configure tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llama_model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure model with quantization for memory efficiency
            if self.device.type == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llama_model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device.type == "cuda" else None,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            # Move to device if not using device_map
            if self.device.type != "cuda" or quantization_config is None:
                self.model = self.model.to(self.device)
            
            log_model_info(self.model, self.logger)
            self.logger.info("LLaMA 3 model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading LLaMA 3 model: {e}")
            raise
    
    def _create_prompt_strategies(self) -> List[Dict[str, Any]]:
        """Create different prompting strategies for diverse summaries."""
        strategies = [
            {
                "name": "comprehensive",
                "template": "Summarize the following research paper comprehensively. Include the main contributions, methodology, key findings, and implications:\n\n{content}",
                "generation_config": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_new_tokens": 300,
                    "do_sample": True,
                    "repetition_penalty": 1.1
                }
            },
            {
                "name": "concise",
                "template": "Provide a concise summary of this research paper focusing on the core contributions and results:\n\n{content}",
                "generation_config": {
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "max_new_tokens": 200,
                    "do_sample": True,
                    "repetition_penalty": 1.2
                }
            },
            {
                "name": "technical",
                "template": "Write a technical summary of this paper emphasizing the methodology and technical innovations:\n\n{content}",
                "generation_config": {
                    "temperature": 0.6,
                    "top_p": 0.85,
                    "max_new_tokens": 250,
                    "do_sample": True,
                    "repetition_penalty": 1.15
                }
            },
            {
                "name": "impact_focused",
                "template": "Summarize this research paper with emphasis on its impact, applications, and future implications:\n\n{content}",
                "generation_config": {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_new_tokens": 280,
                    "do_sample": True,
                    "repetition_penalty": 1.05
                }
            }
        ]
        return strategies
    
    def generate_summary(self, content: str, strategy: str = "comprehensive") -> str:
        """Generate a single summary using specified strategy."""
        try:
            # Find strategy
            strategy_config = next(
                (s for s in self.prompt_strategies if s["name"] == strategy), 
                self.prompt_strategies[0]
            )
            
            # Prepare prompt
            prompt = strategy_config["template"].format(content=content)
            
            # Truncate content if too long
            max_input_length = self.llama_max_length - 200  # Reserve space for generation
            if len(prompt) > max_input_length:
                content_truncated = truncate_text(content, max_input_length - len(prompt) + len(content))
                prompt = strategy_config["template"].format(content=content_truncated)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.llama_max_length
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **strategy_config["generation_config"],
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up the generated text
            summary = self._clean_summary(generated_text)
            
            self.logger.debug(f"Generated summary using strategy '{strategy}': {len(summary)} characters")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary with strategy '{strategy}': {e}")
            return ""
    
    def generate_diverse_summaries(self, content: str, num_summaries: int = 2) -> List[Dict[str, Any]]:
        """Generate multiple diverse summaries using different strategies."""
        summaries = []
        
        # Select random strategies
        selected_strategies = random.sample(
            self.prompt_strategies, 
            min(num_summaries, len(self.prompt_strategies))
        )
        
        for i, strategy in enumerate(selected_strategies):
            self.logger.info(f"Generating summary {i+1}/{num_summaries} using strategy: {strategy['name']}")
            
            summary_text = self.generate_summary(content, strategy['name'])
            
            if summary_text:
                summaries.append({
                    'summary_id': f"summary_{i+1}",
                    'strategy': strategy['name'],
                    'text': summary_text,
                    'length': len(summary_text),
                    'timestamp': format_timestamp()
                })
        
        self.logger.info(f"Generated {len(summaries)} diverse summaries")
        return summaries
    
    def _clean_summary(self, text: str) -> str:
        """Clean and format generated summary."""
        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Summary:", "Here's a summary:", "The paper", "This paper",
            "In this paper", "The research", "This research"
        ]
        
        text = text.strip()
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure it ends with proper punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def generate_summaries_for_papers(self, papers: List[Dict[str, Any]], 
                                    num_summaries_per_paper: int = 2) -> List[Dict[str, Any]]:
        """Generate summaries for a list of papers."""
        all_summaries = []
        
        for i, paper in enumerate(papers):
            self.logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.get('paper_id', 'unknown')}")
            
            # Get content for summarization
            content = self._get_paper_content(paper)
            if not content:
                self.logger.warning(f"No content found for paper {paper.get('paper_id', 'unknown')}")
                continue
            
            # Generate summaries
            summaries = self.generate_diverse_summaries(content, num_summaries_per_paper)
            
            # Add paper metadata to summaries
            for summary in summaries:
                summary.update({
                    'paper_id': paper['paper_id'],
                    'paper_title': paper.get('title', ''),
                    'paper_authors': paper.get('authors', []),
                    'paper_source': paper.get('source', '')
                })
            
            all_summaries.extend(summaries)
        
        self.logger.info(f"Generated {len(all_summaries)} total summaries for {len(papers)} papers")
        return all_summaries
    
    def _get_paper_content(self, paper: Dict[str, Any]) -> str:
        """Extract content from paper for summarization."""
        # Try different content sources in order of preference
        sections = paper.get('sections', {})
        
        # Use abstract if available and substantial
        if sections.get('abstract') and len(sections['abstract']) > 100:
            return sections['abstract']
        
        # Use introduction if available
        if sections.get('introduction'):
            return sections['introduction']
        
        # Use methodology + results if available
        methodology = sections.get('methodology', '')
        results = sections.get('results', '')
        if methodology and results:
            return f"{methodology}\n\n{results}"
        
        # Use full text as last resort
        full_text = paper.get('full_text', '')
        if full_text:
            # Truncate to reasonable length
            return truncate_text(full_text, 4000)
        
        return ""
    
    def save_summaries(self, summaries: List[Dict[str, Any]], filepath: str) -> None:
        """Save summaries to file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for summary in summaries:
                    f.write(json.dumps(summary, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Saved {len(summaries)} summaries to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving summaries: {e}")
    
    def load_summaries(self, filepath: str) -> List[Dict[str, Any]]:
        """Load summaries from file."""
        try:
            summaries = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        summaries.append(json.loads(line.strip()))
            
            self.logger.info(f"Loaded {len(summaries)} summaries from {filepath}")
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error loading summaries: {e}")
            return []
    
    def create_summary_pairs(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create summary pairs for annotation from individual summaries."""
        # Group summaries by paper
        papers_summaries = {}
        for summary in summaries:
            paper_id = summary['paper_id']
            if paper_id not in papers_summaries:
                papers_summaries[paper_id] = []
            papers_summaries[paper_id].append(summary)
        
        # Create pairs for each paper
        summary_pairs = []
        for paper_id, paper_summaries in papers_summaries.items():
            if len(paper_summaries) >= 2:
                # Create all possible pairs
                for i in range(len(paper_summaries)):
                    for j in range(i + 1, len(paper_summaries)):
                        pair = {
                            'paper_id': paper_id,
                            'paper_title': paper_summaries[0].get('paper_title', ''),
                            'summary_1': paper_summaries[i],
                            'summary_2': paper_summaries[j],
                            'pair_id': f"{paper_id}_pair_{i}_{j}"
                        }
                        summary_pairs.append(pair)
        
        self.logger.info(f"Created {len(summary_pairs)} summary pairs from {len(summaries)} summaries")
        return summary_pairs

class MultimodalSummaryGenerator(SummaryGenerator):
    """Extended summary generator that incorporates multimodal content."""
    
    def __init__(self):
        super().__init__()
        self.logger.info("Initialized multimodal summary generator")
    
    def generate_multimodal_summary(self, paper: Dict[str, Any], strategy: str = "comprehensive") -> str:
        """Generate summary incorporating both text and visual content."""
        try:
            # Get text content
            text_content = self._get_paper_content(paper)
            
            # Get image descriptions if available
            image_descriptions = self._describe_images(paper.get('images', []))
            
            # Create multimodal prompt
            if image_descriptions:
                multimodal_content = f"{text_content}\n\nVisual Content:\n{image_descriptions}"
            else:
                multimodal_content = text_content
            
            # Generate summary
            summary = self.generate_summary(multimodal_content, strategy)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating multimodal summary: {e}")
            return ""
    
    def _describe_images(self, images: List[Dict[str, Any]]) -> str:
        """Generate descriptions for images in the paper."""
        if not images:
            return ""
        
        descriptions = []
        for i, img in enumerate(images[:3]):  # Limit to first 3 images
            descriptions.append(f"Figure {i+1}: [Image from page {img.get('page', 'unknown')}]")
        
        return "\n".join(descriptions)
