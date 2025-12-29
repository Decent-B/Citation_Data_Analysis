"""
Community Detection Evaluation Metrics

This module provides functions to load community detection results and calculate
evaluation metrics comparing predicted clusters to ground truth.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Union, Tuple, Dict, Optional
from sklearn.metrics import (
    adjusted_mutual_info_score, 
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score
)
from sklearn.metrics.cluster import contingency_matrix
import random
def sample_graph_nodes(G: nx.Graph, max_nodes: int, seed: int) -> nx.Graph:
    if G.number_of_nodes() <= max_nodes:
        return G
    random.seed(seed)
    sampled_nodes = random.sample(list(G.nodes()), max_nodes)
    print(f"Sampled {len(sampled_nodes)} nodes from {G.number_of_nodes()} total nodes")
    return G.subgraph(sampled_nodes).copy()

def load_community_labels(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load community detection results or ground truth from CSV file.
    
    Args:
        filepath: Path to CSV file containing community assignments
                 Expected columns: 'paper_id', 'cluster_id'
    
    Returns:
        DataFrame with columns: 'paper_id', 'cluster_id'
        Sorted by paper_id for consistent ordering
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Community CSV file not found: {filepath}")
    
    # Load CSV
    df = pd.read_csv(filepath, dtype={'paper_id': str, 'id': str})
    
    # Validate required columns
    if 'paper_id' not in df.columns:
        raise ValueError(f"CSV must contain 'paper_id' column. Found: {list(df.columns)}")
    if 'cluster_id' not in df.columns:
        raise ValueError(f"CSV must contain 'cluster_id' column. Found: {list(df.columns)}")
    
    # Keep only required columns and sort by paper_id for consistency
    df = df[['paper_id', 'cluster_id']].sort_values('paper_id').reset_index(drop=True)
    
    # Ensure cluster_id is string for consistency
    df['cluster_id'] = df['cluster_id'].astype(str)
    
    return df


def align_labels(preds_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align prediction and ground truth labels by paper_id.
    
    Only papers that appear in both dataframes are included in the output.
    
    Args:
        preds_df: DataFrame with predicted clusters (columns: paper_id, cluster_id)
        ground_truth_df: DataFrame with ground truth clusters (columns: paper_id, cluster_id)
    
    Returns:
        Tuple of (preds, ground_truths) as numpy arrays of cluster labels
        Both arrays have the same length and correspond to the same papers
    """
    # Merge on paper_id to find common papers
    merged = preds_df.merge(
        ground_truth_df,
        on='paper_id',
        suffixes=('_pred', '_true')
    )
    
    if len(merged) == 0:
        raise ValueError("No common papers found between predictions and ground truth")
    
    # Extract aligned labels
    preds = merged['cluster_id_pred'].values
    ground_truths = merged['cluster_id_true'].values
    
    return preds, ground_truths


def calculate_ami(preds: np.ndarray, ground_truths: np.ndarray) -> float:
    """
    Calculate Adjusted Mutual Information (AMI).
    
    AMI measures the agreement between two clusterings, adjusted for chance.
    - Range: [-1, 1] (typically [0, 1])
    - 1.0 = perfect agreement
    - 0.0 = random clustering
    - Negative values indicate worse than random
    
    Args:
        preds: Predicted cluster labels (1D array)
        ground_truths: Ground truth cluster labels (1D array)
    
    Returns:
        AMI score (float)
    
    Raises:
        ValueError: If input arrays have different lengths
    """
    if len(preds) != len(ground_truths):
        raise ValueError(
            f"Input arrays must have same length. "
            f"Got preds: {len(preds)}, ground_truths: {len(ground_truths)}"
        )
    
    return adjusted_mutual_info_score(ground_truths, preds)


def calculate_ari(preds: np.ndarray, ground_truths: np.ndarray) -> float:
    """
    Calculate Adjusted Rand Index (ARI).
    
    ARI measures the similarity between two clusterings, adjusted for chance.
    - Range: [-1, 1]
    - 1.0 = perfect agreement (identical clustering)
    - 0.0 = random clustering
    - Negative values indicate worse than random
    
    Args:
        preds: Predicted cluster labels (1D array)
        ground_truths: Ground truth cluster labels (1D array)
    
    Returns:
        ARI score (float)
    
    Raises:
        ValueError: If input arrays have different lengths
    """
    if len(preds) != len(ground_truths):
        raise ValueError(
            f"Input arrays must have same length. "
            f"Got preds: {len(preds)}, ground_truths: {len(ground_truths)}"
        )
    
    return adjusted_rand_score(ground_truths, preds)


def calculate_vi(preds: np.ndarray, ground_truths: np.ndarray) -> float:
    """
    Calculate Variation of Information (VI).
    
    VI measures the distance between two clusterings based on entropy.
    - Range: [0, log(N)] where N is the number of samples
    - 0.0 = perfect agreement (identical clustering)
    - Higher values = more different clusterings
    - Unlike AMI/ARI, lower is better
    
    Args:
        preds: Predicted cluster labels (1D array)
        ground_truths: Ground truth cluster labels (1D array)
    
    Returns:
        VI score (float)
    
    Raises:
        ValueError: If input arrays have different lengths
    """
    if len(preds) != len(ground_truths):
        raise ValueError(
            f"Input arrays must have same length. "
            f"Got preds: {len(preds)}, ground_truths: {len(ground_truths)}"
        )
    
    # Get contingency matrix
    contingency = contingency_matrix(ground_truths, preds)
    
    # Calculate probabilities
    n = np.sum(contingency)
    contingency = contingency.astype(float)
    
    # Joint probability
    pij = contingency / n
    
    # Marginal probabilities
    pi = np.sum(pij, axis=1)  # P(true cluster)
    pj = np.sum(pij, axis=0)  # P(pred cluster)
    
    # Calculate entropies (with safe log to handle zeros)
    def safe_entropy(p):
        """Calculate entropy handling zero probabilities."""
        p_nonzero = p[p > 0]
        return -np.sum(p_nonzero * np.log(p_nonzero))
    
    h_true = safe_entropy(pi)  # H(ground_truth)
    h_pred = safe_entropy(pj)  # H(predictions)
    
    # Mutual information
    pij_nonzero = pij[pij > 0]
    pi_outer_pj = np.outer(pi, pj)
    mutual_info = np.sum(pij_nonzero * np.log(pij_nonzero / pi_outer_pj[pij > 0]))
    
    # Variation of Information = H(true) + H(pred) - 2*MI
    vi = h_true + h_pred - 2 * mutual_info
    
    return vi


# ============================================================================
# ADDITIONAL EXTERNAL INDICES
# ============================================================================

def calculate_nmi(preds: np.ndarray, ground_truths: np.ndarray) -> float:
    """
    Calculate Normalized Mutual Information (NMI).
    
    NMI measures mutual information normalized to [0,1], but NOT adjusted for chance.
    AMI is generally preferred over NMI when comparing partitions with different numbers
    of clusters.
    
    - Range: [0, 1]
    - 1.0 = perfect agreement
    - 0.0 = no mutual information
    
    Args:
        preds: Predicted cluster labels (1D array)
        ground_truths: Ground truth cluster labels (1D array)
    
    Returns:
        NMI score (float)
    """
    if len(preds) != len(ground_truths):
        raise ValueError(
            f"Input arrays must have same length. "
            f"Got preds: {len(preds)}, ground_truths: {len(ground_truths)}"
        )
    
    return normalized_mutual_info_score(ground_truths, preds)


def calculate_homogeneity_completeness_v(preds: np.ndarray, ground_truths: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate Homogeneity, Completeness, and V-measure.
    
    - Homogeneity: Each cluster contains only members of a single class
    - Completeness: All members of a given class are assigned to the same cluster
    - V-measure: Harmonic mean of homogeneity and completeness
    
    All metrics range [0, 1], where 1.0 is perfect.
    
    Args:
        preds: Predicted cluster labels (1D array)
        ground_truths: Ground truth cluster labels (1D array)
    
    Returns:
        Tuple of (homogeneity, completeness, v_measure)
    """
    if len(preds) != len(ground_truths):
        raise ValueError(
            f"Input arrays must have same length. "
            f"Got preds: {len(preds)}, ground_truths: {len(ground_truths)}"
        )
    
    return homogeneity_completeness_v_measure(ground_truths, preds)


def calculate_fowlkes_mallows(preds: np.ndarray, ground_truths: np.ndarray) -> float:
    """
    Calculate Fowlkes-Mallows Index (FMI).
    
    FMI is the geometric mean of pairwise precision and recall.
    Based on counts of node pairs that are co-clustered in both vs mismatched.
    
    - Range: [0, 1]
    - 1.0 = perfect agreement
    - 0.0 = no agreement
    
    Args:
        preds: Predicted cluster labels (1D array)
        ground_truths: Ground truth cluster labels (1D array)
    
    Returns:
        FMI score (float)
    """
    if len(preds) != len(ground_truths):
        raise ValueError(
            f"Input arrays must have same length. "
            f"Got preds: {len(preds)}, ground_truths: {len(ground_truths)}"
        )
    
    return fowlkes_mallows_score(ground_truths, preds)


def calculate_purity(preds: np.ndarray, ground_truths: np.ndarray) -> float:
    """
    Calculate Purity.
    
    Purity measures the extent to which each cluster contains data points from
    a single class. For each cluster, we count the majority class and sum these
    counts across all clusters.
    
    - Range: [0, 1]
    - 1.0 = perfect (each cluster contains only one class)
    - Higher values indicate purer clusters
    
    Note: Purity is biased toward producing many small clusters.
    
    Args:
        preds: Predicted cluster labels (1D array)
        ground_truths: Ground truth cluster labels (1D array)
    
    Returns:
        Purity score (float)
    """
    if len(preds) != len(ground_truths):
        raise ValueError(
            f"Input arrays must have same length. "
            f"Got preds: {len(preds)}, ground_truths: {len(ground_truths)}"
        )
    
    # Build contingency matrix
    contingency = contingency_matrix(ground_truths, preds)
    
    # For each cluster (column), find the majority class (max in that column)
    # Sum these maxima and divide by total number of samples
    purity_score = np.sum(np.max(contingency, axis=0)) / np.sum(contingency)
    
    return float(purity_score)


# ============================================================================
# INTERNAL INDICES (no ground truth required)
# ============================================================================

def load_graph_from_edges(edges_file: Union[str, Path], max_edges: Optional[int] = None) -> nx.Graph:
    """
    Load a NetworkX graph from an edge list CSV file.
    
    WARNING: For very large graphs (>100M edges), consider using a graph database
    or sampling a subset of edges for metric calculation.
    
    Args:
        edges_file: Path to CSV file with columns 'source_id', 'target_id'
        max_edges: Optional limit on number of edges to load (for testing/sampling)
    
    Returns:
        NetworkX Graph or DiGraph
    """
    edges_df = pd.read_csv(edges_file, nrows=max_edges)
    
    # Create directed graph (for citation networks)
    G = nx.DiGraph()
    
    print(f"Loading {len(edges_df):,} edges into graph...")
    
    for _, row in edges_df.iterrows():
        G.add_edge(str(row['source_id']), str(row['target_id']))
    
    print(f"✓ Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    return G


def load_graph_from_dataframe(edges_df: pd.DataFrame, max_edges: Optional[int] = None) -> nx.Graph:
    """
    Load a NetworkX graph from an edge list DataFrame.
    
    This function is useful when you already have edges in memory (e.g., from
    extract_edges_from_db in utils.py) and want to avoid writing to disk.
    
    Args:
        edges_df: DataFrame with columns 'source_id', 'target_id'
        max_edges: Optional limit on number of edges to use (for sampling)
    
    Returns:
        NetworkX DiGraph (directed graph for citation networks)
    """
    # Sample edges if max_edges specified
    if max_edges is not None and len(edges_df) > max_edges:
        edges_df = edges_df.sample(n=max_edges, random_state=42)
        print(f"Sampled {max_edges:,} edges from {len(edges_df):,} total edges")
    
    # Create directed graph (for citation networks)
    print(f"Loading {len(edges_df):,} edges into graph...")
    
    # More efficient bulk loading using from_pandas_edgelist
    G = nx.from_pandas_edgelist(
        edges_df,
        source='source_id',
        target='target_id',
        create_using=nx.DiGraph  # type: ignore
    )
    
    print(f"✓ Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    return G


def calculate_modularity(G: nx.Graph, communities_dict: Dict[str, int]) -> float:
    """
    Calculate (Directed) Modularity Q.
    
    Modularity measures how much more internally linked a partition is than expected
    by chance. For directed graphs, uses in/out degree structure.
    
    - Range: typically [-0.5, 1.0]
    - Higher is better
    - Positive values indicate stronger community structure than random
    
    Args:
        G: NetworkX Graph or DiGraph
        communities_dict: Dictionary mapping node_id -> community_id
    
    Returns:
        Modularity score (float)
    """
    # Convert dict to list of sets for networkx
    communities_by_id = {}
    for node, comm in communities_dict.items():
        if comm not in communities_by_id:
            communities_by_id[comm] = set()
        communities_by_id[comm].add(node)
    
    communities_list = list(communities_by_id.values())
    
    # Use NetworkX's modularity function (handles directed graphs)
    return nx.algorithms.community.modularity(G, communities_list)


def calculate_coverage(G: nx.Graph, communities_dict: Dict[str, int]) -> float:
    """
    Calculate Coverage.
    
    Coverage is the fraction of intra-community edges to total edges.
    
    - Range: [0, 1]
    - Higher is better
    - 1.0 means all edges are within communities
    
    Args:
        G: NetworkX Graph or DiGraph
        communities_dict: Dictionary mapping node_id -> community_id
    
    Returns:
        Coverage score (float)
    """
    # Convert dict to list of sets
    communities_by_id = {}
    for node, comm in communities_dict.items():
        if comm not in communities_by_id:
            communities_by_id[comm] = set()
        communities_by_id[comm].add(node)
    
    communities_list = list(communities_by_id.values())
    
    # Calculate coverage manually: intra-community edges / total edges
    intra_edges = 0
    for community in communities_list:
        for u in community:
            for v in G.neighbors(u):
                if v in community:
                    intra_edges += 1
    
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0.0
    
    # For directed graphs, we've counted each intra-edge once
    # For undirected graphs, each edge is counted twice, but total_edges counts each once
    return intra_edges / total_edges if G.is_directed() else intra_edges / (2 * total_edges)


def calculate_performance(G: nx.Graph, communities_dict: Dict[str, int]) -> float:
    """
    Calculate Performance (simplified for large graphs).
    
    WARNING: True performance metric requires O(n²) comparisons and is not practical
    for large graphs (millions of nodes). This simplified version uses coverage as proxy.
    
    For large graphs, use Coverage and Modularity instead.
    
    - Range: [0, 1]
    - Higher is better
    
    Args:
        G: NetworkX Graph or DiGraph
        communities_dict: Dictionary mapping node_id -> community_id
    
    Returns:
        Performance score (float) - returns coverage for large graphs
    """
    n = len(communities_dict)
    
    # For large graphs (>100k nodes), performance metric is impractical
    # Return coverage instead as a proxy
    if n > 100_000:
        print(f"WARNING: Performance metric is O(n²) and impractical for {n:,} nodes.")
        print(f"         Returning coverage as a proxy metric instead.")
        return calculate_coverage(G, communities_dict)
    
    # Convert dict to list of sets
    communities_by_id = {}
    for node, comm in communities_dict.items():
        if comm not in communities_by_id:
            communities_by_id[comm] = set()
        communities_by_id[comm].add(node)
    
    communities_list = list(communities_by_id.values())
    
    if n <= 1:
        return 1.0
    
    # Count intra-community edges
    intra_edges = 0
    for community in communities_list:
        for u in community:
            for v in G.neighbors(u):
                if v in community:
                    intra_edges += 1
    
    # For undirected graphs, we counted each edge twice
    if not G.is_directed():
        intra_edges //= 2
    
    # Count all existing edges
    actual_edges = G.number_of_edges()
    
    # Inter-community edges
    inter_edges = actual_edges - intra_edges
    
    # Total possible edges
    total_possible = n * (n - 1) if G.is_directed() else n * (n - 1) // 2
    
    # Inter-community non-edges (edges that could exist but don't between communities)
    inter_non_edges = total_possible - actual_edges - (sum(len(c) * (len(c) - 1) for c in communities_list) // (1 if G.is_directed() else 2))
    
    # Performance = (correctly classified pairs) / total pairs
    # Correctly classified = intra-edges (same cluster, connected) + inter-non-edges (diff cluster, not connected)
    correct_pairs = intra_edges + inter_non_edges
    
    return correct_pairs / total_possible if total_possible > 0 else 0.0


def calculate_conductance(G: nx.Graph, community_nodes: set) -> float:
    """
    Calculate Conductance for a single community.
    
    Conductance measures how many edges leave the community relative to
    the volume (degree sum) of the smaller side.
    
    - Range: [0, 1]
    - Lower is better (fewer edges leaving the community)
    - 0.0 means perfect isolation
    
    Args:
        G: NetworkX Graph or DiGraph
        community_nodes: Set of node IDs in the community
    
    Returns:
        Conductance score (float)
    """
    # Calculate cut size (edges leaving the community)
    cut_size = 0
    for node in community_nodes:
        for neighbor in G.neighbors(node):
            if neighbor not in community_nodes:
                cut_size += 1
    
    # Calculate volume (sum of degrees within community)
    # For directed graphs, use out-degree
    volume = 0
    if G.is_directed():
        for node in community_nodes:
            if node in G:
                volume += G.out_degree[node]  # type: ignore
    else:
        for node in community_nodes:
            if node in G:
                volume += G.degree[node]  # type: ignore
    
    # Conductance = cut_size / volume
    if volume == 0:
        return 0.0
    
    return cut_size / volume


def calculate_average_conductance(G: nx.Graph, communities_dict: Dict[str, int]) -> float:
    """
    Calculate average conductance across all communities.
    
    Args:
        G: NetworkX Graph or DiGraph
        communities_dict: Dictionary mapping node_id -> community_id
    
    Returns:
        Average conductance score (float)
    """
    # Group nodes by community
    communities_by_id = {}
    for node, comm in communities_dict.items():
        if comm not in communities_by_id:
            communities_by_id[comm] = set()
        communities_by_id[comm].add(node)
    
    conductances = []
    for comm_nodes in communities_by_id.values():
        if len(comm_nodes) > 0:
            cond = calculate_conductance(G, comm_nodes)
            conductances.append(cond)
    
    return np.mean(conductances) if conductances else 0.0


def calculate_internal_indices(
    communities_file: Union[str, Path],
    edges_file: Optional[Union[str, Path]] = None,
    max_edges: Optional[int] = None,
    n_edges: Optional[int] = None,
    verbose: bool = True
) -> dict:
    """
    Calculate internal quality indices for community detection (no ground truth required).
    
    Internal indices evaluate the quality of a partition based on the graph structure alone.
    
    WARNING: For very large graphs (>100M edges), this function may consume excessive
    memory and time. Consider using max_edges to sample a subset for evaluation, or
    provide n_edges parameter to skip graph loading entirely (limited metrics available).
    
    Args:
        communities_file: Path to CSV with columns 'paper_id', 'cluster_id'
        edges_file: Path to CSV with columns 'source_id', 'target_id' (optional if n_edges provided)
        max_edges: Optional limit on edges to load (for sampling large graphs)
        n_edges: If provided, skip loading graph and only calculate metrics that don't require it
        verbose: If True, print summary statistics
    
    Returns:
        Dictionary with metrics:
        {
            'modularity': Modularity Q score (None if n_edges provided without graph),
            'coverage': Coverage score (None if n_edges provided without graph),
            'performance': Performance score (None if n_edges provided without graph),
            'avg_conductance': Average conductance (None if n_edges provided without graph),
            'n_nodes': Number of nodes,
            'n_edges': Number of edges,
            'n_communities': Number of communities
        }
    """
    # Load data
    communities_df = load_community_labels(communities_file)
    
    # Create communities dictionary
    communities_dict = dict(zip(communities_df['paper_id'], communities_df['cluster_id']))
    
    # If n_edges is provided, skip graph loading
    if n_edges is not None:
        n_nodes = len(communities_dict)
        n_communities = len(set(communities_dict.values()))
        
        if verbose:
            print("=" * 60)
            print("Internal Community Detection Indices")
            print("=" * 60)
            print(f"Nodes in communities:       {n_nodes:,}")
            print(f"Edges in graph:             {n_edges:,}")
            print(f"Number of communities:      {n_communities:,}")
            print("-" * 60)
            print("⚠ Graph not loaded (n_edges provided)")
            print("  Only basic statistics available")
            print("  For full metrics, omit n_edges parameter")
            print("=" * 60)
        
        results = {
            'modularity': None,
            'coverage': None,
            'performance': None,
            'avg_conductance': None,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_communities': n_communities
        }
        
        return results
    
    # Load graph if n_edges not provided
    if edges_file is None:
        raise ValueError("Either edges_file or n_edges must be provided")
    
    G = load_graph_from_edges(edges_file, max_edges=max_edges)
    
    # Filter to nodes that exist in the graph
    communities_dict = {node: comm for node, comm in communities_dict.items() if node in G}
    
    if len(communities_dict) == 0:
        raise ValueError("No nodes from communities file found in the graph. Check that node IDs match.")
    
    # Summary statistics (calculate early for warnings)
    n_nodes = len(communities_dict)
    n_edges_actual = G.number_of_edges()
    n_communities = len(set(communities_dict.values()))
    
    if verbose:
        print("=" * 60)
        print("Internal Community Detection Indices")
        print("=" * 60)
        print(f"Nodes in graph:             {G.number_of_nodes():,}")
        print(f"Nodes with communities:     {n_nodes:,}")
        print(f"Edges in graph:             {n_edges_actual:,}")
        print(f"Number of communities:      {n_communities:,}")
        
        if max_edges:
            print(f"\n⚠ Using sampled graph ({max_edges:,} edges)")
            print("  Results are approximate and may not reflect full graph")
        
        print("-" * 60)
        
        if n_nodes > 1_000_000:
            print("⚠ WARNING: Large graph detected!")
            print("  This may take several minutes to compute...")
            print("-" * 60)
    
    # Calculate metrics
    modularity = calculate_modularity(G, communities_dict)
    coverage = calculate_coverage(G, communities_dict)
    performance = calculate_performance(G, communities_dict)
    avg_conductance = calculate_average_conductance(G, communities_dict)
    
    results = {
        'modularity': modularity,
        'coverage': coverage,
        'performance': performance,
        'avg_conductance': avg_conductance,
        'n_nodes': n_nodes,
        'n_edges': n_edges_actual,
        'n_communities': n_communities
    }
    
    if verbose:
        print(f"Modularity (Q):             {modularity:.4f}")
        print(f"Coverage:                   {coverage:.4f}")
        print(f"Performance:                {performance:.4f}")
        print(f"Average Conductance:        {avg_conductance:.4f}")
        print("=" * 60)
        print("\nInterpretation:")
        print("  Modularity: Higher is better (positive = structure)")
        print("  Coverage:   Higher is better (1.0 = all edges internal)")
        print("  Performance: Higher is better")
        print("  Conductance: Lower is better (0.0 = perfect isolation)")
        print("=" * 60)
    
    return results


def calculate_internal_indices_from_dataframe(
    communities_file: Union[str, Path],
    edges_df: pd.DataFrame,
    max_edges: Optional[int] = None,
    verbose: bool = True
) -> dict:
    """
    Calculate internal quality indices using an in-memory edges DataFrame.
    
    This function is designed to work directly with the output of 
    extract_edges_from_db() from utils.py, avoiding the need to write
    edges to disk first.
    
    Args:
        communities_file: Path to CSV with columns 'paper_id', 'cluster_id'
        edges_df: DataFrame with columns 'source_id', 'target_id'
        max_edges: Optional limit on edges to use (for sampling large graphs)
        verbose: If True, print summary statistics
    
    Returns:
        Dictionary with metrics:
        {
            'modularity': Modularity Q score,
            'coverage': Coverage score,
            'performance': Performance score,
            'avg_conductance': Average conductance,
            'n_nodes': Number of nodes,
            'n_edges': Number of edges,
            'n_communities': Number of communities
        }
    
    Example:
        >>> from utils import extract_edges_from_db
        >>> edges_df = extract_edges_from_db()
        >>> results = calculate_internal_indices_from_dataframe(
        ...     'results/leiden_communities.csv',
        ...     edges_df
        ... )
    """
    # Load communities
    communities_df = load_community_labels(communities_file)
    communities_dict = dict(zip(communities_df['paper_id'], communities_df['cluster_id']))
    
    # Load graph from dataframe
    G = load_graph_from_dataframe(edges_df, max_edges=max_edges)
    G = sample_graph_nodes(G, 3000, 42)

    # Filter to nodes that exist in the graph
    communities_dict = {node: comm for node, comm in communities_dict.items() if node in G}
    
    if len(communities_dict) == 0:
        raise ValueError("No nodes from communities file found in the graph. Check that node IDs match.")
    
    # Summary statistics
    n_nodes = len(communities_dict)
    n_edges_actual = G.number_of_edges()
    n_communities = len(set(communities_dict.values()))
    
    if verbose:
        print("=" * 60)
        print("Internal Community Detection Indices")
        print("=" * 60)
        print(f"Nodes in graph:             {G.number_of_nodes():,}")
        print(f"Nodes with communities:     {n_nodes:,}")
        print(f"Edges in graph:             {n_edges_actual:,}")
        print(f"Number of communities:      {n_communities:,}")
        
        if max_edges and len(edges_df) > max_edges:
            print(f"\n⚠ Using sampled graph ({max_edges:,} edges)")
            print("  Results are approximate and may not reflect full graph")
        
        print("-" * 60)
        
        if n_nodes > 1_000_000:
            print("⚠ WARNING: Large graph detected!")
            print("  This may take several minutes to compute...")
            print("-" * 60)
    
    # Calculate metrics
    modularity = calculate_modularity(G, communities_dict)
    coverage = calculate_coverage(G, communities_dict)
    performance = calculate_performance(G, communities_dict)
    avg_conductance = calculate_average_conductance(G, communities_dict)
    
    results = {
        'modularity': modularity,
        'coverage': coverage,
        'performance': performance,
        'avg_conductance': avg_conductance,
        'n_nodes': n_nodes,
        'n_edges': n_edges_actual,
        'n_communities': n_communities
    }
    
    if verbose:
        print(f"Modularity (Q):             {modularity:.4f}")
        print(f"Coverage:                   {coverage:.4f}")
        print(f"Performance:                {performance:.4f}")
        print(f"Average Conductance:        {avg_conductance:.4f}")
        print("=" * 60)
        print("\nInterpretation:")
        print("  Modularity: Higher is better (positive = structure)")
        print("  Coverage:   Higher is better (1.0 = all edges internal)")
        print("  Performance: Higher is better")
        print("  Conductance: Lower is better (0.0 = perfect isolation)")
        print("=" * 60)
    
    return results


# ============================================================================
# EXTERNAL INDICES (with ground truth)
# ============================================================================

def calculate_external_indices(
    preds_file: Union[str, Path],
    ground_truth_file: Union[str, Path],
    verbose: bool = True
) -> dict:
    """
    Evaluate clustering predictions against ground truth using all external indices.
    
    Calculates AMI, ARI, VI, NMI, Homogeneity, Completeness, V-measure, 
    Fowlkes-Mallows, and Purity.
    
    Args:
        preds_file: Path to predictions CSV (columns: paper_id, cluster_id)
        ground_truth_file: Path to ground truth CSV (columns: paper_id, cluster_id)
        verbose: If True, print summary statistics
    
    Returns:
        Dictionary with all external metrics
    """
    # Load data
    preds_df = load_community_labels(preds_file)
    ground_truth_df = load_community_labels(ground_truth_file)
    
    # Align labels
    preds, ground_truths = align_labels(preds_df, ground_truth_df)
    
    # Calculate primary metrics (AMI, ARI, VI)
    ami = calculate_ami(preds, ground_truths)
    ari = calculate_ari(preds, ground_truths)
    vi = calculate_vi(preds, ground_truths)
    
    # Calculate additional external metrics
    nmi = calculate_nmi(preds, ground_truths)
    homogeneity, completeness, v_measure = calculate_homogeneity_completeness_v(preds, ground_truths)
    fmi = calculate_fowlkes_mallows(preds, ground_truths)
    purity = calculate_purity(preds, ground_truths)
    
    # Summary statistics
    n_papers = len(preds)
    n_pred_clusters = len(np.unique(preds))
    n_true_clusters = len(np.unique(ground_truths))
    
    results = {
        # Primary metrics
        'ami': ami,
        'ari': ari,
        'vi': vi,
        # Additional metrics
        'nmi': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'fowlkes_mallows': fmi,
        'purity': purity,
        # Statistics
        'n_papers': n_papers,
        'n_pred_clusters': n_pred_clusters,
        'n_true_clusters': n_true_clusters
    }
    
    if verbose:
        print("=" * 70)
        print("External Community Detection Evaluation Results")
        print("=" * 70)
        print(f"Papers evaluated:           {n_papers:,}")
        print(f"Predicted clusters:         {n_pred_clusters:,}")
        print(f"Ground truth clusters:      {n_true_clusters:,}")
        print("-" * 70)
        print("PRIMARY METRICS (Recommended):")
        print(f"  Adjusted Mutual Info (AMI):  {ami:.4f}")
        print(f"  Adjusted Rand Index (ARI):   {ari:.4f}")
        print(f"  Variation of Info (VI):      {vi:.4f}")
        print("-" * 70)
        print("ADDITIONAL METRICS:")
        print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
        print(f"  Homogeneity:                  {homogeneity:.4f}")
        print(f"  Completeness:                 {completeness:.4f}")
        print(f"  V-measure:                    {v_measure:.4f}")
        print(f"  Fowlkes-Mallows Index (FMI):  {fmi:.4f}")
        print(f"  Purity:                       {purity:.4f}")
        print("=" * 70)
        print("\nInterpretation:")
        print("  AMI, ARI: Higher is better (1.0 = perfect, 0.0 = random)")
        print("  VI:       Lower is better (0.0 = perfect)")
        print("  NMI, Homogeneity, Completeness, V-measure, FMI, Purity:")
        print("            Higher is better (1.0 = perfect)")
        print("\nNote: Purity is biased toward many small clusters")
        print("=" * 70)
    
    return results

if __name__ == "__main__":
    # Example usage
    TARGET_TOPICS_DEFAULT = [
        "T10036", "T10320", "T10775", "T11273", "T11714", "T11307", "T11652", "T11512",
        "T10203", "T10538", "T12016", "T10028", "T12026", "T11396", "T11636", "T14414",
        "T10906", "T10100", "T10963", "T10848", "T10050", "T11975", "T12101", "T11612",
        "T10181", "T11710", "T11550", "T10664", "T12031", "T10201", "T10860", "T13083",
        "T13629", "T12380", "T12262", "T10531", "T11105", "T12923", "T14339", "T10627",
        "T10824", "T10601", "T10057", "T11448", "T11094", "T10331", "T10052", "T11659",
        "T10688", "T11019", "T11165", "T12549", "T10719", "T10812", "T10260", "T10430",
        "T12490", "T10743", "T11450", "T11675", "T12127", "T12423", "T10317", "T10101",
        "T14067", "T11181", "T11106", "T11719", "T11986", "T13650", "T13398", "T10799",
        "T11891", "T14280", "T11937", "T14201", "T11478", "T10715", "T10772", "T10054",
        "T10273", "T12222", "T10575", "T10125", "T11458", "T11409", "T10080", "T10711",
        "T10651", "T10714", "T10742", "T10796", "T11896", "T11932", "T12791", "T12024",
        "T10138", "T12216", "T12326", "T10237", "T10951", "T10400", "T10734", "T11045",
        "T10764", "T11504", "T11800", "T11644", "T11241", "T12122", "T11598", "T13999",
        "T10927", "T10270", "T13913", "T10462", "T10586", "T10653", "T10879", "T10191",
        "T10326", "T10709", "T10868", "T10888", "T10648", "T10789", "T11024", "T11938",
        "T13985", "T11572", "T11499", "T12885", "T10953", "T10350", "T10912", "T11446",
        "T14380", "T10712", "T13673", "T13166", "T12289", "T12863", "T10102", "T13976",
    ]
    preds_path = Path("results/checkpoint_level_10.csv")
    ground_truth_path = Path("results/ground_truth_300_paper_per_topic.csv")
    
    results = calculate_external_indices(preds_path, ground_truth_path, verbose=True)
