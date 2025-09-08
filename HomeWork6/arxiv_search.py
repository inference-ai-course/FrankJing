import arxiv
from typing import List, Dict, Any


def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for papers related to the given query and return formatted results.
    
    Args:
        query (str): Search query for arXiv papers
        max_results (int): Maximum number of results to return
        
    Returns:
        str: Formatted string containing paper information
    """
    try:
        # Create a search object
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Execute the search
        papers = list(search.results())
        
        if not papers:
            return f"No papers found for query: '{query}'"
        
        # Format the results
        result = f"Found {len(papers)} papers for '{query}':\n\n"
        
        for i, paper in enumerate(papers, 1):
            # Get authors as a string
            authors_str = ", ".join([author.name for author in paper.authors[:3]])  # Limit to first 3 authors
            if len(paper.authors) > 3:
                authors_str += " et al."
            
            result += f"{i}. **{paper.title}**\n"
            result += f"   Authors: {authors_str}\n"
            result += f"   Abstract: {paper.summary[:150]}...\n"
            result += f"   arXiv ID: {paper.get_short_id()}\n"
            result += f"   PDF URL: {paper.pdf_url}\n\n"
        
        return result
        
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"


def get_paper_details(arxiv_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific arXiv paper by ID.
    
    Args:
        arxiv_id (str): The arXiv ID of the paper
        
    Returns:
        Dict[str, Any]: Dictionary containing paper details
    """
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        return {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "arxiv_id": paper.get_short_id(),
            "pdf_url": paper.pdf_url,
            "published": paper.published.strftime("%Y-%m-%d"),
            "categories": paper.categories
        }
        
    except Exception as e:
        return {"error": f"Error fetching paper details: {str(e)}"}