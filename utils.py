import pandas as pd
import sqlite3
import numpy as np
from collections import Counter
import ast
from typing import Optional, List


def parse_list_cell(x):
    """
    Parse referenced_works into Python lists.
    
    Args:
        x: Input cell value (can be string, list, tuple, or None)
    
    Returns:
        List of referenced work IDs
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return ast.literal_eval(x)
    except Exception:
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if not s:
            return []
        return [e.strip().strip('"').strip("'") for e in s.split(",") if e.strip()]


def extract_edges_from_db(
    db_path: str = "data/openalex_works.db",
    topics: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract citation edges from OpenAlex database.
    
    This function connects to the OpenAlex SQLite database, loads papers from
    specified topics, and builds a citation network edge list where both the
    citing and cited papers are in the topic set.
    
    Args:
        db_path: Path to the SQLite database file
        topics: List of topic IDs to filter (e.g., ['T10078', 'T10001'])
                If None, uses default topics
        verbose: If True, print progress information
    
    Returns:
        DataFrame with columns ['source_id', 'target_id'] representing
        directed citation edges (source cites target)
    
    Example:
        >>> edges_df = extract_edges_from_db()
        >>> print(edges_df.head())
           source_id     target_id
        0  W123456789    W987654321
        1  W123456789    W111111111
        ...
    """
    # Default topics if not specified
    if topics is None:
        topics = ['T10078', 'T10001', 'T10018', 'T10030', 'T10017']
    
    # Connect to database
    if verbose:
        print(f"Connecting to database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # Build query with topic filter
    topic_list_str = ", ".join([f"'{t}'" for t in topics])
    query = f"""
        SELECT id, referenced_works
        FROM works
        WHERE topic IN ({topic_list_str})
    """
    
    if verbose:
        print(f"Loading papers from topics: {topics}")
    
    # Load papers with their references
    topic_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if verbose:
        print(f"Loaded {len(topic_df):,} papers from database")
    
    # Parse referenced_works column into lists
    topic_df['referenced_works_parsed'] = topic_df['referenced_works'].apply(parse_list_cell)
    
    # Create edge list (only include edges where both nodes are in the topic set)
    valid_ids = set(topic_df['id'].values)
    
    if verbose:
        print(f"Building edge list (filtering to {len(valid_ids):,} valid paper IDs)...")
    
    edges = [
        (src, dst)
        for src, refs in zip(topic_df['id'], topic_df['referenced_works_parsed'])
        for dst in refs
        if dst and dst in valid_ids  # Both source and target must be in topic set
    ]
    
    # Create DataFrame with standard column names for metrics module
    edges_df = pd.DataFrame(edges, columns=['source_id', 'target_id'])
    
    if verbose:
        print(f"✓ Built {len(edges_df):,} edges from {len(topic_df):,} papers")
        if len(topic_df) > 0:
            print(f"  Average references per paper: {len(edges_df) / len(topic_df):.2f}")
    
    return edges_df


if __name__ == "__main__":
    # Example usage
    edges_df = extract_edges_from_db(verbose=True)
    print("\nFirst few edges:")
    print(edges_df.head())
    print(f"\nEdge DataFrame shape: {edges_df.shape}")