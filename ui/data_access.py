"""Data access layer for loading papers metadata and search results.

Uses the ranking package to load papers filtered by specific topics.
Supports both GPU (CUDA/cuGraph) and CPU (NetworkX) for PageRank computation.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from database.connection import test_connection
from ranking import load_topic_data, build_graph, build_cugraph, run_pagerank
from ranking.utils import check_gpu_available
from ui.config import TARGET_TOPICS

DATA_DIR = Path("data")

# Module-level caches
_papers_cache: Optional[pd.DataFrame] = None
_edges_cache: Optional[pd.DataFrame] = None
_pagerank_cache: Optional[pd.DataFrame] = None
_cugraph_cache: Optional[Tuple] = None  # (G_gpu, vid_df)


def load_papers_metadata() -> pd.DataFrame:
    """
    Load paper metadata from PostgreSQL database for target topics.
    
    Uses the ranking package's load_topic_data() method to load papers
    filtered by the target topics
    
    Returns:
        DataFrame with columns: id, title, doi, publication_date, cited_by_count, topic
    """
    global _papers_cache, _edges_cache, _cugraph_cache
    
    # Return cached data if available
    if _papers_cache is not None:
        return _papers_cache
    
    # Test connection first
    if not test_connection():
        print("⚠️ Database not available, using placeholder data")
        return _get_placeholder_data()
    
    try:
        print("Loading ALL papers from database (not filtered by topics)")
        
        # Check if GPU is available
        gpu_available = check_gpu_available()
        
        if gpu_available:
            print("GPU available - using cuGraph for graph operations")
            # Use cuGraph for GPU-accelerated operations
            # Pass None to load ALL papers instead of filtering by topics
            topic_df, edges_df, G_gpu, vid_df = build_cugraph(None)
            # Cache the cuGraph objects for PageRank
            if G_gpu is not None:
                _cugraph_cache = (G_gpu, vid_df)
        else:
            print("GPU not available - using CPU for graph operations")
            # Fall back to CPU
            # Pass None to load ALL papers instead of filtering by topics
            topic_df, edges_df = build_graph(None)
        
        # Cache the edges for fallback PageRank computation
        _edges_cache = edges_df
        
        # Ensure required columns exist and are properly typed
        topic_df['id'] = topic_df['id'].astype(str)
        topic_df['cited_by_count'] = pd.to_numeric(
            topic_df['cited_by_count'], errors='coerce'
        ).fillna(0).astype(int)
        
        # Ensure publication_date is string for display
        if 'publication_date' in topic_df.columns:
            topic_df['publication_date'] = topic_df['publication_date'].astype(str)
        
        # Cache the result
        _papers_cache = topic_df
        
        print(f"Loaded {len(topic_df):,} papers from database")
        return topic_df
        
    except Exception as e:
        print(f"⚠️ Error loading papers: {e}. Using placeholder data")
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
        'publication_date': ['1951-11-01', '1970-08-01', '1976-05-07'],
        'cited_by_count': [100, 200, 150],
        'topic': ['T10078', 'T10078', 'T10078']
    })


def load_pagerank_scores() -> Optional[pd.DataFrame]:
    """
    Compute PageRank scores using CUDA (cuGraph) if available, otherwise CPU (NetworkX).
    
    Uses the ranking package's run_pagerank() method with cuGraph for GPU acceleration.
    Falls back to NetworkX if GPU is not available.
    
    Returns:
        DataFrame with columns: paper_id, pagerank
        Returns None if computation fails
    """
    global _pagerank_cache, _edges_cache, _cugraph_cache
    
    # Return cached scores if available
    if _pagerank_cache is not None:
        return _pagerank_cache
    
    # Ensure papers are loaded (this also loads the graph)
    if _papers_cache is None:
        load_papers_metadata()
    
    try:
        # Try GPU first if cuGraph is cached
        if _cugraph_cache is not None:
            G_gpu, vid_df = _cugraph_cache
            print("Computing PageRank using CUDA (cuGraph)...")
            
            # Use the ranking package's run_pagerank for GPU
            pr_result = run_pagerank(G_gpu, vid_df, alpha=0.85, max_iter=100, tol=1e-6)
            _pagerank_cache = pr_result.scores
            
            print(f"✓ CUDA PageRank computed for {len(_pagerank_cache)} papers")
            return _pagerank_cache
        
        # Fallback to CPU if GPU not available
        if _edges_cache is not None and len(_edges_cache) > 0:
            print("⚠️ GPU not available, falling back to CPU PageRank...")
            import networkx as nx
            
            G = nx.DiGraph()
            G.add_edges_from(zip(_edges_cache['source'], _edges_cache['target']))
            
            print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            pr_scores = nx.pagerank(G, alpha=0.85)
            
            _pagerank_cache = pd.DataFrame([
                {'paper_id': k, 'pagerank': v} for k, v in pr_scores.items()
            ])
            
            print(f"✓ CPU PageRank computed for {len(_pagerank_cache)} papers")
            return _pagerank_cache
        
        print("⚠️ No edges available for PageRank computation")
        return None
        
    except Exception as e:
        print(f"⚠️ PageRank computation error: {e}")
        # Try CPU fallback on any GPU error
        if _edges_cache is not None and len(_edges_cache) > 0:
            try:
                print("Attempting CPU fallback after GPU error...")
                import networkx as nx
                
                G = nx.DiGraph()
                G.add_edges_from(zip(_edges_cache['source'], _edges_cache['target']))
                pr_scores = nx.pagerank(G, alpha=0.85)
                
                _pagerank_cache = pd.DataFrame([
                    {'paper_id': k, 'pagerank': v} for k, v in pr_scores.items()
                ])
                
                print(f"✓ CPU PageRank fallback computed for {len(_pagerank_cache)} papers")
                return _pagerank_cache
                
            except Exception as cpu_error:
                print(f"⚠️ CPU fallback also failed: {cpu_error}")
        
        return None


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
                'publication_date': row.get('publication_date', ''),
                'cited_by_count': row.get('cited_by_count', 0)
            })
        else:
            # Paper not found in metadata
            results.append({
                'id': paper_id,
                'title': 'Unknown',
                'doi': '',
                'publication_date': '',
                'cited_by_count': 0
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
    
    Returns the cached edges from the ranking package's build_graph() method.
    
    Returns:
        DataFrame with columns: source, target
        Returns None if no edges available
    """
    global _edges_cache
    
    # Ensure papers are loaded (this also loads edges)
    if _edges_cache is None:
        load_papers_metadata()
    
    return _edges_cache


def clear_cache():
    """Clear all cached data. Useful for testing or refreshing data."""
    global _papers_cache, _edges_cache, _pagerank_cache, _cugraph_cache
    _papers_cache = None
    _edges_cache = None
    _pagerank_cache = None
    _cugraph_cache = None


def get_gpu_status() -> str:
    """Return a string describing GPU availability status."""
    if check_gpu_available():
        return "CUDA (GPU)"
    return "CPU"


if __name__ == "__main__":
    # Test loading functions
    print("Testing data access functions...")
    print("=" * 60)
    
    # Check GPU status
    print(f"GPU Status: {get_gpu_status()}")
    print()
    
    papers_df = load_papers_metadata()
    print(f"\nPapers Metadata ({len(papers_df):,} rows):")
    print(papers_df.head())
    if 'topic' in papers_df.columns:
        print(f"\nUnique topics: {papers_df['topic'].nunique()}")
    
    pr_scores = load_pagerank_scores()
    if pr_scores is not None:
        print(f"\nPageRank Scores ({len(pr_scores):,} rows):")
        print(pr_scores.nlargest(5, 'pagerank'))
    

