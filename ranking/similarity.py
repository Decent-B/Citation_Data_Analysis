"""
Paper similarity and finding methods using co-citation analysis.
"""

import pandas as pd
from typing import List, Set, Optional


def compute_cocitation_scores(
    topic_df: pd.DataFrame,
    seed_paper_ids: List[str],
    refs_col: str = 'referenced_works_parsed',
    top_k: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute co-citation scores for papers relative to seed papers.
    
    Score = number of papers commonly cited by both the candidate paper and seed papers.
    
    Args:
        topic_df: DataFrame with paper data
        seed_paper_ids: List of seed paper IDs
        refs_col: Column name containing parsed referenced works list
        top_k: Optional limit on results (None for all)
    
    Returns:
        DataFrame with paper_id, cocitation_score, title, date, citations
    """
    # Get all papers cited by seed papers (union of references)
    seed_citations: Set[str] = set()
    seed_set = set(seed_paper_ids)
    
    for seed_id in seed_paper_ids:
        seed_row = topic_df[topic_df['id'] == seed_id]
        if not seed_row.empty:
            refs = seed_row.iloc[0][refs_col]
            seed_citations.update(refs)
    
    # Calculate co-citation score for each paper
    results = []
    for _, row in topic_df.iterrows():
        paper_id = row['id']
        
        # Skip seed papers
        if paper_id in seed_set:
            continue
        
        # Get papers cited by this paper
        cited_by_paper = set(row[refs_col])
        
        # Calculate overlap (co-citation count)
        common_citations = seed_citations.intersection(cited_by_paper)
        score = len(common_citations)
        
        if score > 0:
            results.append({
                'paper_id': paper_id,
                'cocitation_score': score,
                'title': row['title'],
                'date': row['publication_date'],
                'citations': row['cited_by_count']
            })
    
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        return results_df
    
    results_df = results_df.sort_values('cocitation_score', ascending=False)
    
    if top_k:
        results_df = results_df.head(top_k)
    
    return results_df.reset_index(drop=True)


def find_related_papers(
    topic_df: pd.DataFrame,
    seed_paper_ids: List[str],
    method: str = 'cocitation',
    top_k: int = 10,
    **kwargs
) -> pd.DataFrame:
    """
    Find papers related to seed papers using specified method.
    
    Args:
        topic_df: DataFrame with paper data
        seed_paper_ids: List of seed paper IDs
        method: Method to use ('cocitation')
        top_k: Number of results to return
        **kwargs: Additional arguments for specific methods
    
    Returns:
        DataFrame with related papers
    """
    if method == 'cocitation':
        return compute_cocitation_scores(
            topic_df, 
            seed_paper_ids, 
            top_k=top_k,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'cocitation'")
