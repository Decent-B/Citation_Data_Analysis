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
    # TODO: Replace with actual search implementations
    # For now, return stub that searches titles, IDs, and DOIs
    
    if not query or query.strip() == "":
        return []
    
    # Simple case-insensitive search in title, id, and doi
    query_lower = query.lower()
    matches = papers_df[
        (papers_df['title'].str.lower().str.contains(query_lower, na=False, regex=False)) |
        (papers_df['id'].str.lower().str.contains(query_lower, na=False, regex=False)) |
        (papers_df['doi'].astype(str).str.lower().str.contains(query_lower, na=False, regex=False))
    ]
    
    # Return top-k results
    result_ids = matches['id'].head(k).tolist()
    
    # If we don't have enough matches, pad with random papers
    if len(result_ids) < k:
        remaining = papers_df[~papers_df['id'].isin(result_ids)]
        if len(remaining) > 0:
            additional = remaining['id'].sample(
                min(k - len(result_ids), len(remaining)),
                random_state=42
            ).tolist()
            result_ids.extend(additional)
    
    return result_ids[:k]
