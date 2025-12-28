"""Data access layer for loading papers metadata and search results."""

import pandas as pd
from pathlib import Path
from typing import Optional

from database.connection import get_engine, test_connection

DATA_DIR = Path("data")


def load_papers_metadata() -> pd.DataFrame:
    """
    Load paper metadata from PostgreSQL database.
    
    Returns:
        DataFrame with columns: id, title, doi, publication_date
    """
    # Test connection first
    if not test_connection():
        # Return placeholder if database is not available
        print("⚠️ Database not available, using placeholder data")
        return _get_placeholder_data()
    
    try:
        print("Accessing PostgreSQL database...")
        engine = get_engine()
        query = """
            SELECT id, title, doi, publication_date
            FROM papers
        """
        df = pd.read_sql_query(query, engine)
        
        # Ensure id is string type
        df['id'] = df['id'].astype(str)
        
        return df
    except Exception as e:
        # Database error - return placeholder
        print(f"⚠️ Database error: {e}. Using placeholder data")
        return _get_placeholder_data()


def _get_placeholder_data() -> pd.DataFrame:
    """Return placeholder data when database is unavailable."""
    return pd.DataFrame({
        'id': ['W1775749144', 'W2100837269', 'W2128635872'],
        'title': [
            'Protein measurement with the Folin phenol reagent',
            'Cleavage of structural proteins during the assembly of the head of bacteriophage T4',
            'A rapid and sensitive method for the quantitation of microgram quantities'
        ],
        'doi': ['10.1016/s0021-9258(19)52451-6', '10.1038/227680a0', '10.1006/abio.1976.9999'],
        'publication_date': ['1951-11-01', '1970-08-01', '1976-05-07']
    })


def get_papers_by_ids(papers_df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """
    Get paper metadata for a list of IDs, preserving order.
    
    Args:
        papers_df: DataFrame containing paper metadata
        ids: List of paper IDs to retrieve
        
    Returns:
        DataFrame with rows in the same order as ids
    """
    # Create a mapping for fast lookup
    papers_dict = papers_df.set_index('id').to_dict('index')
    
    # Build results in order
    results = []
    for paper_id in ids:
        if paper_id in papers_dict:
            row = papers_dict[paper_id]
            results.append({
                'id': paper_id,
                'title': row.get('title', 'Unknown'),
                'doi': row.get('doi', ''),
                'publication_date': row.get('publication_date', '')
            })
        else:
            # Paper not found in metadata
            results.append({
                'id': paper_id,
                'title': 'Unknown',
                'doi': '',
                'publication_date': ''
            })
    
    return pd.DataFrame(results)


def load_communities(algorithm: str) -> Optional[pd.DataFrame]:
    """
    Load community detection results for a given algorithm.
    
    Args:
        algorithm: One of "Girvan-Newman", "Kernighan-Lin", "Louvain"
        
    Returns:
        DataFrame with columns: id, community
        Returns None if file doesn't exist
    """
    # Map algorithm names to file paths
    algorithm_files = {
        "Girvan-Newman": DATA_DIR / "communities_gn.csv",
        "Kernighan-Lin": DATA_DIR / "communities_kl.csv",
        "Louvain": DATA_DIR / "communities_louvain.csv"
    }
    
    # Try algorithm-specific file first
    filepath = algorithm_files.get(algorithm)
    if filepath and filepath.exists():
        df = pd.read_csv(filepath, dtype={'id': str})
        # Handle legacy 'paper_id' column name
        if 'paper_id' in df.columns and 'id' not in df.columns:
            df = df.rename(columns={'paper_id': 'id'})
        return df
    
    # Fallback to generic communities.csv
    fallback = DATA_DIR / "communities.csv"
    if fallback.exists():
        df = pd.read_csv(fallback, dtype={'id': str})
        if 'paper_id' in df.columns and 'id' not in df.columns:
            df = df.rename(columns={'paper_id': 'id'})
        return df
    
    return None


def load_edges() -> Optional[pd.DataFrame]:
    """
    Load edge list for graph visualization.
    
    Returns:
        DataFrame with columns: source_id, target_id
        Returns None if file doesn't exist
    """
    edges_file = DATA_DIR / "edges.csv"
    if edges_file.exists():
        return pd.read_csv(edges_file, dtype={'source_id': str, 'target_id': str})
    return None


if __name__ == "__main__":
    # Test loading functions
    papers_df = load_papers_metadata()
    print("Papers Metadata:")
    print(papers_df.head())
