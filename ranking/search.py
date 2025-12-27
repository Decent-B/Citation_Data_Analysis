"""
Paper search functionality with keyword matching and PageRank ranking.
"""

import re
import pandas as pd
from typing import List, Optional, Set

from ranking.utils import STOP_WORDS


def tokenize_query(query: str) -> List[str]:
    """
    Tokenize and normalize a search query into keywords.
    
    Args:
        query: Search query string
    
    Returns:
        List of normalized keywords (lowercase, stop words removed)
    """
    # Convert to lowercase and split on non-alphanumeric characters
    words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
    # Remove common stop words
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def count_keyword_matches(title: Optional[str], keywords: List[str]) -> int:
    """
    Count how many keywords appear in a title.
    
    Args:
        title: Paper title (can be None)
        keywords: List of keywords to search for
    
    Returns:
        Number of matching keywords
    """
    if title is None or pd.isna(title):
        return 0
    title_lower = title.lower()
    title_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', title_lower))
    return sum(1 for kw in keywords if kw in title_words)


class PaperSearcher:
    """
    Search papers using keyword matching and optional PageRank-based ranking.
    """
    
    def __init__(self, topic_df: pd.DataFrame, pagerank_scores: Optional[pd.DataFrame] = None):
        """
        Initialize the searcher.
        
        Args:
            topic_df: DataFrame containing paper data (must have 'id', 'title', etc.)
            pagerank_scores: Optional DataFrame with 'paper_id' and 'pagerank' columns
        """
        self.topic_df = topic_df
        self.pagerank_scores = pagerank_scores
    
    def search_by_keywords(
        self,
        query: str,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Search papers by keyword matching in titles.
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            DataFrame with matching papers sorted by match count
        """
        keywords = tokenize_query(query)
        
        if not keywords:
            return pd.DataFrame()
        
        results = []
        for _, row in self.topic_df.iterrows():
            score = count_keyword_matches(row['title'], keywords)
            if score > 0:
                results.append({
                    'paper_id': row['id'],
                    'keyword_score': score,
                    'title': row['title'],
                    'date': row['publication_date'],
                    'citations': row['cited_by_count']
                })
        
        results_df = pd.DataFrame(results)
        if len(results_df) == 0:
            return results_df
        
        results_df = results_df.sort_values('keyword_score', ascending=False)
        return results_df.head(top_k).reset_index(drop=True)
    
    def search_with_pagerank(
        self,
        query: str,
        keyword_threshold: int = 1,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Search papers using keywords then rank by PageRank.
        
        First filters papers meeting keyword threshold, then sorts by PageRank.
        
        Args:
            query: Search query string
            keyword_threshold: Minimum number of keywords that must match
            top_k: Number of results to return
        
        Returns:
            DataFrame with matching papers sorted by PageRank
        """
        if self.pagerank_scores is None:
            raise ValueError("PageRank scores not set. Provide pagerank_scores in constructor.")
        
        keywords = tokenize_query(query)
        
        if not keywords:
            return pd.DataFrame()
        
        # Step 1: Filter by keyword threshold
        results = []
        for _, row in self.topic_df.iterrows():
            score = count_keyword_matches(row['title'], keywords)
            if score >= keyword_threshold:
                results.append({
                    'paper_id': row['id'],
                    'keyword_score': score,
                    'title': row['title'],
                    'date': row['publication_date'],
                    'citations': row['cited_by_count']
                })
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Step 2: Merge with PageRank and sort
        results_df = results_df.merge(
            self.pagerank_scores,
            on='paper_id',
            how='left'
        )
        results_df['pagerank'] = results_df['pagerank'].fillna(0)
        results_df = results_df.sort_values('pagerank', ascending=False)
        
        return results_df.head(top_k).reset_index(drop=True)
