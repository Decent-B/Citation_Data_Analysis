"""Search algorithm implementations using PaperSearcher."""

import pandas as pd
from typing import List, Tuple

from ranking.search import PaperSearcher
from ui.data_access import load_pagerank_scores


def run_search(query: str, algorithm: str, k: int, papers_df: pd.DataFrame) -> List[str]:
    """
    Run search using the specified algorithm.
    
    Args:
        query: Search query string
        algorithm: One of "Keyword Matches", "Keyword Matches + PageRank"
        k: Number of results to return
        papers_df: DataFrame containing paper metadata
        
    Returns:
        List of paper IDs
    """
    if not query or query.strip() == "":
        return []
    
    topic_df = papers_df.copy()
    if 'cited_by_count' not in topic_df.columns:
        topic_df['cited_by_count'] = 0
    
    if algorithm == "Keyword Matches":
        searcher = PaperSearcher(topic_df)
        results_df = searcher.search_by_keywords(query, top_k=k)
        return results_df['paper_id'].tolist() if len(results_df) > 0 else []
    
    elif algorithm == "Keyword Matches + PageRank":
        pagerank_scores = load_pagerank_scores()
        
        if pagerank_scores is None:
            searcher = PaperSearcher(topic_df)
            results_df = searcher.search_by_keywords(query, top_k=k)
        else:
            searcher = PaperSearcher(topic_df, pagerank_scores=pagerank_scores)
            results_df = searcher.search_with_pagerank(query, keyword_threshold=1, top_k=k)
        
        return results_df['paper_id'].tolist() if len(results_df) > 0 else []
    
    else:
        searcher = PaperSearcher(topic_df)
        results_df = searcher.search_by_keywords(query, top_k=k)
        return results_df['paper_id'].tolist() if len(results_df) > 0 else []


def run_both_searches(query: str, k: int, papers_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Run BOTH search algorithms simultaneously and return results.
    
    This is more efficient than calling run_search twice because:
    - The PaperSearcher is only created once
    - PageRank scores are loaded only once
    
    Args:
        query: Search query string
        k: Number of results to return
        papers_df: DataFrame containing paper metadata
        
    Returns:
        Tuple of (keyword_results, pagerank_results) where each is a list of paper IDs
    """
    if not query or query.strip() == "":
        return [], []
    
    # Prepare DataFrame
    topic_df = papers_df.copy()
    if 'cited_by_count' not in topic_df.columns:
        topic_df['cited_by_count'] = 0
    
    # Load PageRank scores once
    print(f"Running both algorithms for query: {query}")
    pagerank_scores = load_pagerank_scores()
    
    # Create searcher with pagerank scores
    if pagerank_scores is not None:
        searcher = PaperSearcher(topic_df, pagerank_scores=pagerank_scores)
    else:
        searcher = PaperSearcher(topic_df)
    
    # Run Keyword Matches
    keyword_df = searcher.search_by_keywords(query, top_k=k)
    keyword_results = keyword_df['paper_id'].tolist() if len(keyword_df) > 0 else []
    
    # Run PageRank search (uses same searcher)
    if pagerank_scores is not None:
        pagerank_df = searcher.search_with_pagerank(query, keyword_threshold=1, top_k=k)
        pagerank_results = pagerank_df['paper_id'].tolist() if len(pagerank_df) > 0 else []
    else:
        # Fallback to keyword search if no PageRank available
        pagerank_results = keyword_results
    
    print(f"  Keyword Matches: {len(keyword_results)} results")
    print(f"  PageRank: {len(pagerank_results)} results")
    
    return keyword_results, pagerank_results
