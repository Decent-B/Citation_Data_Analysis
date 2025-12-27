"""
Graph building utilities for citation networks.
"""

import ast
import pandas as pd
from typing import List, Tuple, Set, Optional, Any


def parse_list_cell(x: Any) -> List[str]:
    """
    Parse a list-like cell value from database (handles JSON strings, lists, etc.).
    
    Args:
        x: Value to parse (can be None, list, tuple, or JSON-like string)
    
    Returns:
        List of string values
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return ast.literal_eval(x)
    except Exception:
        # Fallback: try to strip and split (defensive parsing)
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if not s:
            return []
        return [e.strip().strip('"').strip("'") for e in s.split(",") if e.strip()]


def build_edges_df(
    topic_df: pd.DataFrame,
    source_col: str = 'id',
    refs_col: str = 'referenced_works_parsed'
) -> pd.DataFrame:
    """
    Build an edge list DataFrame from paper data.
    
    Creates edges from papers to their references, filtering to only include
    edges where both source and destination are in the dataset.
    
    Args:
        topic_df: DataFrame with paper data
        source_col: Column name for paper IDs
        refs_col: Column name for parsed referenced works
    
    Returns:
        DataFrame with 'source' and 'target' columns
    """
    valid_ids: Set[str] = set(topic_df[source_col].values)
    
    edges = [
        (src, dst) 
        for src, refs in zip(topic_df[source_col], topic_df[refs_col])
        for dst in refs
        if dst and dst in valid_ids
    ]
    
    edges_df = pd.DataFrame(edges, columns=['source', 'target'])
    return edges_df


def build_networkx_graph(edges_df: pd.DataFrame):
    """
    Build a NetworkX directed graph from edge list.
    
    Args:
        edges_df: DataFrame with 'source' and 'target' columns
    
    Returns:
        NetworkX DiGraph
    """
    import networkx as nx
    
    G = nx.DiGraph()
    G.add_edges_from(edges_df.itertuples(index=False, name=None))
    return G


def build_cugraph_with_mapping(edges_df: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
    """
    Build a cuGraph graph with vertex ID mapping.
    
    cuGraph requires numeric vertex IDs, so this function creates
    a mapping between original string IDs and integer IDs.
    
    Args:
        edges_df: DataFrame with 'source' and 'target' columns (string IDs)
    
    Returns:
        Tuple of (cugraph.Graph, vid_df) where vid_df maps vertex_id to int_id
    """
    import cudf
    import cugraph as cg
    
    # Convert to cuDF
    cudf_edges = cudf.from_pandas(edges_df.astype(str))
    
    # Create vertex ID mapping
    unique_vertices = cudf.concat([cudf_edges['source'], cudf_edges['target']]).unique()
    vid_df = unique_vertices.reset_index(drop=True).to_frame(name='vertex_id')
    vid_df['int_id'] = cudf.RangeIndex(len(vid_df))
    
    # Map edges to integer IDs
    cudf_edges = cudf_edges.merge(
        vid_df, left_on='source', right_on='vertex_id', how='left'
    ).rename(columns={'int_id': 'src_id'}).drop(columns=['vertex_id'])
    
    cudf_edges = cudf_edges.merge(
        vid_df, left_on='target', right_on='vertex_id', how='left'
    ).rename(columns={'int_id': 'dst_id'}).drop(columns=['vertex_id'])
    
    # Build cuGraph Graph
    G_gpu = cg.Graph(directed=True)
    try:
        G_gpu.from_cudf_edgelist(
            cudf_edges, 
            source='src_id', 
            destination='dst_id', 
            edge_attr=None
        )
    except Exception:
        # Fallback for older cuGraph versions
        G_gpu.add_edge_list(cudf_edges['src_id'], cudf_edges['dst_id'], None)
    
    return G_gpu, vid_df
