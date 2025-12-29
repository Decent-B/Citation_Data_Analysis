"""
ranking - A Python package for academic paper ranking using citation networks.

This package provides tools for:
- Loading paper data from SQLite databases
- Building citation graphs (GPU via cuGraph or CPU via NetworkX)
- Computing PageRank for paper importance
- Finding related papers using co-citation analysis
- Searching papers with PageRank-enhanced ranking
"""

from ranking.data_loader import load_topic_data, build_graph, build_cugraph
from ranking.utils import clear_gpu_memory, check_gpu_available
from ranking.pagerank import (
    PageRankResult,
    run_pagerank,
    run_pagerank_cpu,
    run_personalized_pagerank,
    track_pagerank_convergence
)
from ranking.search import PaperSearcher, tokenize_query, count_keyword_matches
from ranking.similarity import compute_cocitation_scores, find_related_papers
from ranking.config import TARGET_TOPICS

__version__ = "0.1.0"

__all__ = [
    # Data loading
    "load_topic_data",
    "build_graph",
    "build_cugraph",
    # PageRank
    "PageRankResult",
    "run_pagerank",
    "run_pagerank_cpu",
    "run_personalized_pagerank",
    "track_pagerank_convergence",
    # Search
    "PaperSearcher",
    "tokenize_query",
    "count_keyword_matches",
    # Similarity
    "compute_cocitation_scores",
    "find_related_papers",
    # Utilities
    "clear_gpu_memory",
    "check_gpu_available",
    # Config
    "TARGET_TOPICS",
]

