"""
Data collection module for academic papers.
Handles paper extraction, preprocessing, and multimodal content processing.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import glob
import arxiv

from utils import setup_logging, clean_text, extract_paper_sections, ensure_directory

class PaperCollector:
    """Collects and processes academic papers from various sources."""
    
    def __init__(self):
        self.logger = setup_logging("INFO")
        self.papers_dir = Path("data")
        ensure_directory(str(self.papers_dir))
        
    def collect_arxiv_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Collect papers from arXiv based on search query."""
        self.logger.info(f"Collecting {max_results} papers from arXiv with query: {query}")
        
        # arXiv API endpoint
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.find_all('entry')
            
            papers = []
            for entry in entries:
                paper_data = self._parse_arxiv_entry(entry)
                if paper_data:
                    papers.append(paper_data)
                    
            self.logger.info(f"Successfully collected {len(papers)} papers from arXiv")
            return papers
            
        except Exception as e:
            self.logger.error(f"Error collecting papers from arXiv: {e}")
            return []
    
    def _parse_arxiv_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse individual arXiv entry."""
        try:
            paper_id = entry.find('id').text.split('/')[-1]
            title = entry.find('title').text.strip()
            abstract = entry.find('summary').text.strip()
            authors = [author.find('name').text for author in entry.find_all('author')]
            published = entry.find('published').text
            pdf_url = entry.find('link', {'title': 'pdf'})['href']
            
            return {
                'paper_id': paper_id,
                'title': clean_text(title),
                'abstract': clean_text(abstract),
                'authors': authors,
                'published_date': published,
                'pdf_url': pdf_url,
                'source': 'arxiv'
            }
        except Exception as e:
            self.logger.warning(f"Error parsing arXiv entry: {e}")
            return None
    
    def download_paper_pdf(self, paper_data: Dict[str, Any]) -> Optional[str]:
        """Download PDF of a paper."""
        try:
            pdf_url = paper_data['pdf_url']
            paper_id = paper_data['paper_id']
            
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()
            
            pdf_path = self.papers_dir / f"{paper_id}.pdf"
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            self.logger.info(f"Downloaded PDF for paper {paper_id}")
            return str(pdf_path)
            
        except Exception as e:
            self.logger.error(f"Error downloading PDF for paper {paper_data.get('paper_id', 'unknown')}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            text = ""
            
            # Try PyMuPDF first (better for academic papers)
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text += page.get_text()
                doc.close()
            except Exception as e:
                self.logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            
            return clean_text(text)
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF file."""
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))
                        
                        images.append({
                            'page': page_num + 1,
                            'index': img_index,
                            'image': img_pil,
                            'size': img_pil.size,
                            'format': 'PNG'
                        })
                    pix = None
                    
            doc.close()
            self.logger.info(f"Extracted {len(images)} images from PDF")
            
        except Exception as e:
            self.logger.error(f"Error extracting images from PDF {pdf_path}: {e}")
            
        return images
    
    def process_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete paper including text and images."""
        # paper_id = paper_data['paper_id']
        # self.logger.info(f"Processing paper: {paper_id}")

        query="deep learning"
        # Create a search object
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Execute the search
        papers = list(search.results())
        
        # Format the results
        result = f"Found {len(papers)} papers for '{query}':\n\n"
        
        for i, paper in enumerate(papers, 1):
            # Get authors as a string
            authors_str = ", ".join([author.name for author in paper.authors[:3]])  # Limit to first 3 authors
            if len(paper.authors) > 3:
                authors_str += " et al."
            
            result += f"{i}. {paper.title}\n"
            result += f"   Authors: {authors_str}\n"
            result += f"   Abstract: {paper.summary[:80]}...\n"
            result += f"   arXiv ID: {paper.get_short_id()}\n"
            result += f"   PDF URL: {paper.pdf_url}\n\n"



        # Download PDF
        # pdf_path = self.download_paper_pdf(paper_data)
        # if not pdf_path:
        #     return None
        
        # Extract text
        full_text = self.extract_text_from_pdf(paper_data)
        if not full_text:
            self.logger.warning(f"No text extracted from paper")
            return None
        
        # Extract sections
        sections = extract_paper_sections(full_text)
        
        # Extract images
        images = self.extract_images_from_pdf(paper_data)
        
        # Create processed paper data
        processed_paper = {
            'paper_id': paper_id,
            'title': paper_data['title'],
            'abstract': paper_data['abstract'],
            'authors': paper_data['authors'],
            'published_date': paper_data['published_date'],
            'source': paper_data['source'],
            'pdf_path': pdf_path,
            'full_text': full_text,
            'sections': sections,
            'images': images,
            'text_length': len(full_text),
            'num_images': len(images)
        }
        
        # Save processed paper
        self._save_processed_paper(processed_paper)
        
        return processed_paper
    
    def _save_processed_paper(self, paper_data: Dict[str, Any]) -> None:
        """Save processed paper data."""
        paper_id = paper_data['paper_id']
        
        # Save text data
        text_file = self.papers_dir / f"{paper_id}_processed.json"
        paper_to_save = paper_data.copy()
        
        # Remove PIL Image objects for JSON serialization
        if 'images' in paper_to_save:
            paper_to_save['images'] = [
                {
                    'page': img['page'],
                    'index': img['index'],
                    'size': img['size'],
                    'format': img['format']
                }
                for img in paper_to_save['images']
            ]
        
        import json
        with open(text_file, 'w', encoding='utf-8') as f:
            json.dump(paper_to_save, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved processed paper data: {text_file}")
    
    def load_processed_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Load processed paper data."""
        text_file = self.papers_dir / f"{paper_id}_processed.json"
        
        if not text_file.exists():
            return None
        
        try:
            import json
            with open(text_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading processed paper {paper_id}: {e}")
            return None
    
    def collect_sample_papers(self, num_papers: int = 5) -> List[Dict[str, Any]]:
        """Collect a sample of papers for the project."""
        self.logger.info(f"Collecting {num_papers} sample papers")
        
        # Process papers
        processed_papers = []

        # Get all PDF files from the "data" folder
        pdf_files = glob.glob(os.path.join("data", "*.pdf"))

        for pdf_file in pdf_files:
            processed = self.process_paper(pdf_file)
            if processed:
                processed_papers.append(processed)

        self.logger.info(f"Successfully processed {len(processed_papers)} papers from {len(pdf_files)} pdf files")
        return processed_papers
    
    def get_paper_summary(self, paper_data: Dict[str, Any]) -> str:
        """Get a summary of paper content for prompt generation."""
        sections = paper_data.get('sections', {})
        
        # Prioritize abstract, then introduction, then full text
        if sections.get('abstract'):
            return sections['abstract']
        elif sections.get('introduction'):
            return sections['introduction']
        else:
            # Use first part of full text
            full_text = paper_data.get('full_text', '')
            return full_text[:2000]  # First 2000 characters

# DataFormatter class moved to data_formatter.py for better separation of concerns
