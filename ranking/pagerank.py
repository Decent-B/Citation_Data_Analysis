"""
PageRank computation for ranking.

Supports both GPU (cuGraph) and CPU (NetworkX) implementations.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union

from ranking.utils import clear_gpu_memory


@dataclass
class PageRankResult:
    """Result of PageRank computation."""
    scores: pd.DataFrame  # DataFrame with paper_id and pagerank columns
    iterations: Optional[int] = None
    converged: bool = True


def run_pagerank_cpu(
    edges_df: pd.DataFrame,
    alpha: float = 0.85
) -> PageRankResult:
    """
    Compute PageRank using NetworkX (CPU).
    
    Args:
        edges_df: DataFrame with 'source' and 'target' columns
        alpha: Damping factor (default 0.85)
    
    Returns:
        PageRankResult with scores
    """
    import networkx as nx
    from paper_ranking.graph import build_networkx_graph
    
    G = build_networkx_graph(edges_df)
    pr_scores = nx.pagerank(G, alpha=alpha)
    
    scores_df = pd.DataFrame([
        {'paper_id': k, 'pagerank': v} for k, v in pr_scores.items()
    ])
    
    return PageRankResult(scores=scores_df)


def run_pagerank(
    G_gpu,
    vid_df: pd.DataFrame,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> PageRankResult:
    """
    Compute PageRank using cuGraph (GPU).
    
    Args:
        G_gpu: cuGraph Graph object
        vid_df: DataFrame mapping vertex_id to int_id
        alpha: Damping factor (default 0.85)
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        PageRankResult with scores
    """
    import cugraph as cg
    
    pr_result = cg.pagerank(G_gpu, alpha=alpha, max_iter=max_iter, tol=tol)
    
    # Handle tuple return (df, converged) if fail_on_nonconvergence is used
    if isinstance(pr_result, tuple):
        pr_df = pr_result[0]
        converged = pr_result[1] if len(pr_result) > 1 else True
    else:
        pr_df = pr_result
        converged = True
    
    # Map back to original vertex IDs
    pr_df = pr_df.merge(vid_df, left_on='vertex', right_on='int_id', how='left')
    
    scores_df = pr_df[['vertex_id', 'pagerank']].to_pandas()
    scores_df = scores_df.rename(columns={'vertex_id': 'paper_id'})
    
    return PageRankResult(scores=scores_df, converged=converged)


def run_personalized_pagerank(
    G_gpu,
    vid_df: pd.DataFrame,
    seed_paper_ids: List[str],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> PageRankResult:
    """
    Compute Personalized PageRank with teleportation biased towards seed papers.
    
    Args:
        G_gpu: cuGraph Graph object
        vid_df: DataFrame mapping vertex_id to int_id
        seed_paper_ids: List of paper IDs to use as personalization
        alpha: Damping factor (default 0.85)
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        PageRankResult with personalized scores
    """
    import cudf
    import cugraph as cg
    
    # Map seed paper IDs to integer vertex IDs
    seed_int_ids = []
    for seed_id in seed_paper_ids:
        match = vid_df[vid_df['vertex_id'] == seed_id]
        if len(match) > 0:
            seed_int_ids.append(int(match['int_id'].iloc[0]))
    
    if len(seed_int_ids) == 0:
        raise ValueError("No seed papers found in the graph")
    
    # Create personalization vector (equal weight for all seeds)
    personalization_weight = 1.0 / len(seed_int_ids)
    personalization_df = cudf.DataFrame({
        'vertex': seed_int_ids,
        'values': [personalization_weight] * len(seed_int_ids)
    })
    
    # Run Personalized PageRank
    ppr_result = cg.pagerank(
        G_gpu,
        alpha=alpha,
        personalization=personalization_df,
        max_iter=max_iter,
        tol=tol
    )
    
    # Handle tuple return
    if isinstance(ppr_result, tuple):
        ppr_df = ppr_result[0]
        converged = ppr_result[1] if len(ppr_result) > 1 else True
    else:
        ppr_df = ppr_result
        converged = True
    
    # Map back to original vertex IDs
    ppr_df = ppr_df.merge(vid_df, left_on='vertex', right_on='int_id', how='left')
    
    scores_df = ppr_df[['vertex_id', 'pagerank']].to_pandas()
    scores_df = scores_df.rename(columns={'vertex_id': 'paper_id'})
    
    return PageRankResult(scores=scores_df, converged=converged)


def track_pagerank_convergence(
    G_gpu,
    vid_df: pd.DataFrame,
    alpha: float = 0.85,
    max_iter: int = 100
) -> Tuple[PageRankResult, List[float]]:
    """
    Track PageRank convergence by computing at each iteration.
    
    Args:
        G_gpu: cuGraph Graph object
        vid_df: DataFrame mapping vertex_id to int_id
        alpha: Damping factor
        max_iter: Maximum iterations
    
    Returns:
        Tuple of (final PageRankResult, convergence_history)
        where convergence_history is a list of L1 differences per iteration
    """
    import cugraph as cg
    
    convergence_history = []
    pr_prev = None
    pr_final = None
    
    for iteration in range(1, max_iter + 1):
        try:
            pr_result_raw = cg.pagerank(
                G_gpu,
                alpha=alpha,
                max_iter=iteration,
                tol=0,  # Disable tolerance-based stopping
                fail_on_nonconvergence=False
            )
            
            # Extract DataFrame from tuple if necessary
            if isinstance(pr_result_raw, tuple):
                pr_current = pr_result_raw[0]
            else:
                pr_current = pr_result_raw
            
            # Calculate difference from previous iteration
            if pr_prev is not None:
                merged = pr_prev.merge(pr_current, on='vertex', suffixes=('_prev', '_curr'))
                diff = (merged['pagerank_curr'] - merged['pagerank_prev']).abs().sum()
                convergence_history.append(float(diff))
                
                # Check for convergence
                if diff < 1e-7 and iteration > 10:
                    pr_final = pr_current
                    break
            
            pr_prev = pr_current.copy()
            
        except Exception as e:
            if pr_prev is not None:
                pr_final = pr_prev
                break
            raise
    else:
        pr_final = pr_current
    
    # Map to original IDs
    pr_final = pr_final.merge(vid_df, left_on='vertex', right_on='int_id', how='left')
    scores_df = pr_final[['vertex_id', 'pagerank']].to_pandas()
    scores_df = scores_df.rename(columns={'vertex_id': 'paper_id'})
    
    result = PageRankResult(scores=scores_df, iterations=len(convergence_history))
    return result, convergence_history
