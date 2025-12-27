"""Search algorithm implementations."""

import pandas as pd
from typing import List

def run_search(query: str, algorithm: str, k: int, papers_df: pd.DataFrame) -> List[str]:
    """
    Run search using the specified algorithm.
    
    Args:
        query: Search query string
        algorithm: One of "BM25", "PageRank + BM25", "HITS"
        k: Number of results to return
        papers_df: DataFrame containing paper metadata
        
    Returns:
        List of paper IDs
    """
    # TODO: Replace with actual BM25/PageRank/HITS implementations
    # For now, use different search strategies based on algorithm
    
    if not query or query.strip() == "":
        return []
    
    query_lower = query.lower()
    
    # Different search strategies based on algorithm
    if algorithm == "BM25":
        print(f"Running BM25 search for query: {query}")
        # BM25-like: Search primarily in titles with term frequency consideration
        # matches = papers_df[
        #     papers_df['title'].str.lower().str.contains(query_lower, na=False, regex=False)
        # ].copy()
        matches = papers_df.sample(n=k, random_state=42)        
        # Sort by title length (shorter titles with match = higher relevance)
        if len(matches) > 0:
            matches['title_len'] = matches['title'].str.len()
            matches = matches.sort_values('title_len')  # type: ignore
        
    elif algorithm == "PageRank + BM25":
        print(f"Running PageRank + BM25 search for query: {query}")
        # PageRank + BM25: Search in title and favor highly cited papers
        matches = papers_df.sample(n=k, random_state=42)        
        
        # Simulate PageRank by sorting by publication date (older papers often more cited)
        if len(matches) > 0:
            matches['pub_date'] = pd.to_datetime(matches['publication_date'], errors='coerce')
            matches = matches.sort_values('pub_date', na_position='last')  # type: ignore
        
    elif algorithm == "HITS":
        print(f"Running HITS search for query: {query}")
        # HITS: Search across all fields (broader search)
        matches = papers_df.sample(n=k, random_state=42)        
    else:
        # Default fallback
        matches = papers_df.sample(n=k, random_state=42)        
    
    # Return top-k results
    result_ids = matches['id'].head(k).tolist()
    
    # If we don't have enough matches, pad with random papers
    if len(result_ids) < k:
        remaining = papers_df[~papers_df['id'].isin(result_ids)]
        if len(remaining) > 0:
            # Use different random seeds for different algorithms to show variation
            seed = {'BM25': 42, 'PageRank + BM25': 123, 'HITS': 456}.get(algorithm, 42)
            additional = remaining['id'].sample(
                min(k - len(result_ids), len(remaining)),
                random_state=seed
            ).tolist()
            result_ids.extend(additional)
    
    return result_ids[:k]
