import os
import matplotlib.pyplot as plt

from ranking.data_loader import load_topic_data, build_cugraph
from ranking.search import PaperSearcher
from ranking.pagerank import PageRankResult, run_pagerank, track_pagerank_convergence
from ranking.config import TARGET_TOPICS


def plot_convergence(convergence_history: list, save_path: str) -> None:
    """
    Plot PageRank convergence and save to file.
    
    Args:
        convergence_history: List of L1 differences per iteration
        save_path: Path to save the plot image
    """
    plt.figure(figsize=(10, 6))
    
    iterations = list(range(2, len(convergence_history) + 2))
    
    plt.plot(iterations, convergence_history, 'b-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Difference from Previous Iteration', fontsize=12)
    plt.title('PageRank Convergence', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add convergence threshold line
    plt.axhline(y=1e-7, color='r', linestyle='--', label='Convergence threshold (1e-7)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence plot saved to: {save_path}")


if __name__ == '__main__':
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data from PostgreSQL (topics are loaded via environment-configured connection)
    topic_df, edges_df, G_gpu, vid_df = build_cugraph(
        target_topics=TARGET_TOPICS
    )
    
    # Track PageRank convergence
    print("Running PageRank with convergence tracking...")
    pagerank_result, convergence_history = track_pagerank_convergence(
        G_gpu, vid_df, alpha=0.85, max_iter=100
    )
    
    print(f"Converged after {pagerank_result.iterations} iterations")
    
    # Plot and save convergence
    plot_path = os.path.join(results_dir, 'pagerank_convergence.png')
    plot_convergence(convergence_history, plot_path)
    
    # Get top papers
    searcher = PaperSearcher(topic_df, pagerank_result.scores)
    results = searcher.search_with_pagerank('photocatalysis', keyword_threshold=1, top_k=10)

    print(results)