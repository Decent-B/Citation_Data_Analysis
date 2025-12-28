"""
Community detection algorithms for citation networks.

This module provides utilities for building citation graphs from the SQLite database
and running community detection algorithms (primarily Girvan-Newman).
"""

import gc
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import networkx as nx
import pandas as pd

# Try to import psutil for memory monitoring
try:
    import psutil as _psutil
    HAS_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore
    HAS_PSUTIL = False


def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    if HAS_PSUTIL and _psutil is not None:
        return _psutil.Process().memory_info().rss / 1024 / 1024
    return 0.0


def print_memory(prefix: str = "") -> None:
    """Print current memory usage."""
    if HAS_PSUTIL:
        print(f"{prefix}💾 Memory: {get_memory_mb():.0f} MB")

DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "openalex_works.db"


@dataclass
class GirvanNewmanResult:
    """Result of Girvan-Newman community detection."""
    
    partition: tuple[frozenset[str], ...]
    modularity: float
    num_communities: int
    num_nodes: int
    num_edges: int
    levels_evaluated: int
    elapsed_seconds: float
    parameters: dict


def parse_referenced_works(value: str | None) -> list[str]:
    """
    Parse the referenced_works field from the database.
    
    The field is stored as a JSON string array of paper IDs (without URL prefix).
    Handles: None, empty string, JSON arrays, legacy string formats.
    
    Args:
        value: The raw referenced_works value from the database.
        
    Returns:
        List of paper ID strings.
    """
    if value is None or value == "" or value == "null":
        return []
    
    # Try JSON parsing first (expected format)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        return []
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Fallback: try ast.literal_eval for Python literal format
    try:
        import ast
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if item]
    except (ValueError, SyntaxError):
        pass
    
    # Last resort: try to parse as comma-separated string
    value_str = str(value).strip()
    if value_str.startswith("[") and value_str.endswith("]"):
        value_str = value_str[1:-1].strip()
    if not value_str:
        return []
    
    return [s.strip().strip('"').strip("'") for s in value_str.split(",") if s.strip()]


def load_papers_with_citations(
    db_path: Path | str = DB_FILE,
    topic_filter: str | None = None,
    limit: int | None = None
) -> pd.DataFrame:
    """
    Load paper IDs and their referenced works from the SQLite database.
    
    Args:
        db_path: Path to the SQLite database file.
        topic_filter: Optional topic ID to filter papers (e.g., "T10181").
        limit: Optional limit on number of papers to load.
        
    Returns:
        DataFrame with columns: id, referenced_works (parsed as list).
        
    Raises:
        FileNotFoundError: If database file doesn't exist.
        sqlite3.Error: If database query fails.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    try:
        query = "SELECT id, referenced_works FROM works"
        conditions = []
        
        if topic_filter:
            conditions.append(f"topic = '{topic_filter}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn)
        df['id'] = df['id'].astype(str)
        df['referenced_works_parsed'] = df['referenced_works'].apply(parse_referenced_works)
        
        return df
    finally:
        conn.close()


def load_papers_by_topics(
    db_path: Path | str,
    topics: list[str],
    papers_per_topic: int = 2000
) -> pd.DataFrame:
    """
    Load papers for multiple topics with a limit per topic.
    
    Uses SQL window function to efficiently limit papers per topic,
    ordered by cited_by_count (most cited first).
    
    Args:
        db_path: Path to the SQLite database file.
        topics: List of topic IDs to load.
        papers_per_topic: Maximum papers to load per topic.
        
    Returns:
        DataFrame with columns: id, topic, referenced_works_parsed.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    try:
        # Create placeholders for the IN clause
        placeholders = ', '.join(['?' for _ in topics])
        
        # Use window function to limit papers per topic
        query = f"""
            SELECT id, topic, referenced_works
            FROM (
                SELECT id, topic, referenced_works,
                       ROW_NUMBER() OVER (PARTITION BY topic ORDER BY cited_by_count DESC) as row_num
                FROM works
                WHERE topic IN ({placeholders})
            )
            WHERE row_num <= ?
        """
        
        df = pd.read_sql_query(query, conn, params=(*topics, papers_per_topic))
        df['id'] = df['id'].astype(str)
        df['referenced_works_parsed'] = df['referenced_works'].apply(parse_referenced_works)
        df = df.drop(columns=['referenced_works'])  # Free memory
        
        return df
    finally:
        conn.close()


def build_citation_graph(
    papers_df: pd.DataFrame,
    undirected: bool = True
) -> nx.Graph | nx.DiGraph:
    """
    Build a citation graph from papers DataFrame.
    
    Args:
        papers_df: DataFrame with 'id' and 'referenced_works_parsed' columns.
        undirected: If True, return undirected graph (default for community detection).
        
    Returns:
        NetworkX graph with paper IDs as nodes.
    """
    # Collect all paper IDs that exist in our dataset
    paper_ids = set(papers_df['id'].astype(str))
    
    # Build edge list: (source_paper, referenced_paper)
    edges: list[tuple[str, str]] = []
    for _, row in papers_df.iterrows():
        src = str(row['id'])
        refs = row.get('referenced_works_parsed', [])
        if refs:
            for dst in refs:
                dst = str(dst)
                # Only add edge if target exists in our dataset
                # (or include external refs - depends on use case)
                edges.append((src, dst))
    
    # Create graph
    if undirected:
        G: nx.Graph | nx.DiGraph = nx.Graph()
    else:
        G = nx.DiGraph()
    
    # Add all paper nodes (ensures isolated nodes get community labels)
    G.add_nodes_from(paper_ids)
    
    # Add edges
    G.add_edges_from(edges)
    
    return G


def build_edges_dataframe(papers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an edge list DataFrame for visualization.
    
    Args:
        papers_df: DataFrame with 'id' and 'referenced_works_parsed' columns.
        
    Returns:
        DataFrame with columns: source_id, target_id
    """
    edges: list[dict[str, str]] = []
    paper_ids = set(papers_df['id'].astype(str))
    
    for _, row in papers_df.iterrows():
        src = str(row['id'])
        refs = row.get('referenced_works_parsed', [])
        if refs:
            for dst in refs:
                dst = str(dst)
                # Only include edges where both endpoints are in our dataset
                if dst in paper_ids:
                    edges.append({'source_id': src, 'target_id': dst})
    
    return pd.DataFrame(edges)


def detect_girvan_newman_best_modularity(
    G: nx.Graph,
    max_levels: int = 50,
    progress_callback: Callable[[int, int, float], None] | None = None
) -> GirvanNewmanResult:
    """
    Run Girvan-Newman algorithm and select partition with best modularity.
    
    Args:
        G: Undirected NetworkX graph.
        max_levels: Maximum number of partition levels to evaluate.
        progress_callback: Optional callback(level, num_communities, modularity).
        
    Returns:
        GirvanNewmanResult with best partition and metadata.
    """
    start_time = time.time()
    
    # Handle empty or trivial graphs
    if G.number_of_nodes() == 0:
        return GirvanNewmanResult(
            partition=tuple(),
            modularity=0.0,
            num_communities=0,
            num_nodes=0,
            num_edges=0,
            levels_evaluated=0,
            elapsed_seconds=0.0,
            parameters={'max_levels': max_levels}
        )
    
    if G.number_of_edges() == 0:
        # All nodes are isolated - each is its own community
        partition = tuple(frozenset([node]) for node in G.nodes())
        return GirvanNewmanResult(
            partition=partition,
            modularity=0.0,
            num_communities=len(partition),
            num_nodes=G.number_of_nodes(),
            num_edges=0,
            levels_evaluated=1,
            elapsed_seconds=time.time() - start_time,
            parameters={'max_levels': max_levels}
        )
    
    # Run Girvan-Newman
    communities_generator: Iterator[tuple[frozenset, ...]] = nx.community.girvan_newman(G)
    
    best_partition: tuple[frozenset, ...] | None = None
    best_modularity = -1.0
    levels_evaluated = 0
    
    for level, partition in enumerate(communities_generator):
        if level >= max_levels:
            break
        
        levels_evaluated = level + 1
        
        # Compute modularity for this partition
        try:
            mod = nx.community.modularity(G, partition)
        except (ZeroDivisionError, ValueError):
            mod = 0.0
        
        if progress_callback:
            progress_callback(level + 1, len(partition), mod)
        
        if mod > best_modularity:
            best_modularity = mod
            best_partition = partition
        
        # Early stopping: if we hit single-node communities, stop
        if len(partition) >= G.number_of_nodes():
            break
    
    elapsed = time.time() - start_time
    
    if best_partition is None:
        # Fallback: all nodes in one community
        best_partition = (frozenset(G.nodes()),)
        best_modularity = 0.0
    
    return GirvanNewmanResult(
        partition=best_partition,
        modularity=best_modularity,
        num_communities=len(best_partition),
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        levels_evaluated=levels_evaluated,
        elapsed_seconds=elapsed,
        parameters={'max_levels': max_levels}
    )


def partition_to_communities_df(
    partition: tuple[frozenset[str], ...],
    G: nx.Graph | None = None
) -> pd.DataFrame:
    """
    Convert a partition (tuple of frozensets) to a DataFrame.
    
    Args:
        partition: Tuple of frozensets, each containing node IDs for one community.
        G: Optional graph to include isolated nodes not in partition.
        
    Returns:
        DataFrame with columns: id (str), community (int).
    """
    rows: list[dict] = []
    
    for community_idx, community in enumerate(partition):
        for node_id in community:
            rows.append({
                'id': str(node_id),
                'community': community_idx
            })
    
    # Include any nodes from G that weren't in the partition
    if G is not None:
        included_nodes = {row['id'] for row in rows}
        next_community = len(partition)
        for node in G.nodes():
            if str(node) not in included_nodes:
                rows.append({
                    'id': str(node),
                    'community': next_community
                })
                next_community += 1
    
    return pd.DataFrame(rows)


def save_communities_csv(
    communities_df: pd.DataFrame,
    output_path: Path | str = DATA_DIR / "communities_gn.csv"
) -> Path:
    """
    Save communities DataFrame to CSV.
    
    Args:
        communities_df: DataFrame with 'id' and 'community' columns.
        output_path: Path for output CSV file.
        
    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    communities_df.to_csv(output_path, index=False)
    return output_path


def save_edges_csv(
    edges_df: pd.DataFrame,
    output_path: Path | str = DATA_DIR / "edges.csv"
) -> Path:
    """
    Save edges DataFrame to CSV.
    
    Args:
        edges_df: DataFrame with 'source_id' and 'target_id' columns.
        output_path: Path for output CSV file.
        
    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(output_path, index=False)
    return output_path


def save_metadata_json(
    result: GirvanNewmanResult,
    output_path: Path | str = DATA_DIR / "communities_gn_meta.json"
) -> Path:
    """
    Save Girvan-Newman result metadata to JSON.
    
    Args:
        result: GirvanNewmanResult object.
        output_path: Path for output JSON file.
        
    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'algorithm': 'girvan-newman',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'num_nodes': result.num_nodes,
        'num_edges': result.num_edges,
        'num_communities': result.num_communities,
        'best_modularity': round(result.modularity, 6),
        'levels_evaluated': result.levels_evaluated,
        'elapsed_seconds': round(result.elapsed_seconds, 2),
        'parameters': result.parameters
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_path


def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
    """
    Extract the largest connected component from a graph.
    
    Args:
        G: NetworkX graph (undirected).
        
    Returns:
        Subgraph containing only the largest connected component.
    """
    if G.number_of_nodes() == 0:
        return G
    
    # For undirected graphs
    if isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
        components = list(nx.connected_components(G))
    else:
        # For directed graphs, use weakly connected components
        components = list(nx.weakly_connected_components(G))
    
    if not components:
        return G
    
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


def run_girvan_newman_pipeline(
    max_levels: int = 50,
    max_nodes: int = 3000,
    largest_component_only: bool = True,
    topic_filter: str | None = None,
    limit: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    level_callback: Callable[[int, int, float], None] | None = None,
    save_results: bool = True
) -> tuple[GirvanNewmanResult, pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline: load data, build graph, run GN, save results.
    
    Args:
        max_levels: Maximum GN levels to evaluate.
        max_nodes: Maximum nodes to include (samples if exceeded).
        largest_component_only: If True, restrict to largest connected component.
        topic_filter: Optional topic ID to filter papers.
        limit: Optional limit on papers to load.
        progress_callback: Callback for status messages.
        level_callback: Callback for per-level progress (level, num_communities, modularity).
        save_results: Whether to save CSV/JSON outputs.
        
    Returns:
        Tuple of (GirvanNewmanResult, communities_df, edges_df).
        
    Raises:
        FileNotFoundError: If database doesn't exist.
        ValueError: If no papers found.
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
    
    # Step 1: Load papers
    log("Loading papers from database...")
    papers_df = load_papers_with_citations(topic_filter=topic_filter, limit=limit)
    
    if len(papers_df) == 0:
        raise ValueError("No papers found in database")
    
    log(f"Loaded {len(papers_df)} papers")
    
    # Step 2: Build graph
    log("Building citation graph...")
    G = build_citation_graph(papers_df, undirected=True)
    log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 3: Restrict to largest component if requested
    if largest_component_only and G.number_of_nodes() > 0:
        log("Extracting largest connected component...")
        G = get_largest_connected_component(G)
        log(f"Largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 4: Sample if too large
    if G.number_of_nodes() > max_nodes:
        log(f"Graph exceeds {max_nodes} nodes, sampling...")
        import random
        random.seed(42)
        sampled_nodes = random.sample(list(G.nodes()), max_nodes)
        G = G.subgraph(sampled_nodes).copy()
        log(f"Sampled graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 5: Build edges DataFrame (for visualization)
    # Filter papers_df to only include nodes in the graph
    graph_nodes = set(G.nodes())
    papers_in_graph = papers_df[papers_df['id'].isin(graph_nodes)]
    edges_df = build_edges_dataframe(papers_in_graph)
    
    # Step 6: Run Girvan-Newman
    log("Running Girvan-Newman algorithm...")
    result = detect_girvan_newman_best_modularity(
        G,
        max_levels=max_levels,
        progress_callback=level_callback
    )
    log(f"Completed in {result.elapsed_seconds:.1f}s")
    
    # Step 7: Convert partition to DataFrame
    communities_df = partition_to_communities_df(result.partition, G)
    
    # Step 8: Save results if requested
    if save_results:
        log("Saving results...")
        save_communities_csv(communities_df)
        save_edges_csv(edges_df)
        save_metadata_json(result)
        log(f"Saved to {DATA_DIR}/")
    
    return result, communities_df, edges_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Girvan-Newman community detection on citation network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with defaults
  python community_detection.py
  
  # Quick test on 200 papers
  python community_detection.py --limit 200 --test
  
  # Single topic
  python community_detection.py --topic T10181
  
  # Multiple topics (comma-separated)
  python community_detection.py --topics T10181,T10036,T10320
  
  # Process topics in batches (memory-efficient)
  python community_detection.py --topics T10181,T10036,T10320 --batch-size 3 --papers-per-topic 1000
        """
    )
    
    parser.add_argument(
        "--max-levels", "-l",
        type=int,
        default=50,
        help="Maximum GN levels to evaluate (default: 50)"
    )
    
    parser.add_argument(
        "--max-nodes", "-n",
        type=int,
        default=3000,
        help="Maximum nodes to include (default: 3000)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers loaded from DB (for testing)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DB_FILE),
        help=f"Path to SQLite database (default: {DB_FILE})"
    )
    
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help="Filter papers by single topic ID (e.g., T10181)"
    )
    
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated list of topic IDs (e.g., T10181,T10036,T10320)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of topics to process per batch (default: 5)"
    )
    
    parser.add_argument(
        "--papers-per-topic",
        type=int,
        default=2000,
        help="Max papers to load per topic (default: 2000)"
    )
    
    parser.add_argument(
        "--largest-component",
        action="store_true",
        default=True,
        help="Restrict to largest connected component (default: True)"
    )
    
    parser.add_argument(
        "--no-largest-component",
        action="store_true",
        help="Include all components, not just the largest"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV path (default: data/communities_gn.csv)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run validation checks on output"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle largest component flag
    largest_component = args.largest_component and not args.no_largest_component
    
    # Resolve database path
    db_path = Path(args.db_path)
    
    # Check database exists
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Run the OpenAlex scraper first: python data_scraping/openalex_scraper.py")
        import sys
        sys.exit(1)
    
    print("=" * 60)
    print("Girvan-Newman Community Detection")
    print("=" * 60)
    print_memory("Initial ")
    
    # Parse topics list
    topic_list = []
    if args.topics:
        topic_list = [t.strip() for t in args.topics.split(',') if t.strip()]
    elif args.topic:
        topic_list = [args.topic]
    
    # If we have multiple topics, process in batches
    if len(topic_list) > 1:
        print(f"\n📋 Processing {len(topic_list)} topics in batches of {args.batch_size}")
        print(f"   Papers per topic: {args.papers_per_topic}")
        
        all_results = []
        
        # Split topics into batches
        for batch_idx in range(0, len(topic_list), args.batch_size):
            batch_topics = topic_list[batch_idx:batch_idx + args.batch_size]
            batch_num = batch_idx // args.batch_size + 1
            total_batches = (len(topic_list) + args.batch_size - 1) // args.batch_size
            
            print(f"\n{'='*60}")
            print(f"BATCH {batch_num}/{total_batches}: {', '.join(batch_topics)}")
            print(f"{'='*60}")
            print_memory("Start ")
            
            # Load papers for this batch
            print(f"\n📚 Loading papers for batch...")
            papers_df = load_papers_by_topics(
                db_path=db_path,
                topics=batch_topics,
                papers_per_topic=args.papers_per_topic
            )
            
            # Print per-topic counts
            for topic in batch_topics:
                count = len(papers_df[papers_df['topic'] == topic])
                print(f"   {topic}: {count} papers")
            print(f"   Total: {len(papers_df)} papers")
            
            if len(papers_df) == 0:
                print("⚠️ No papers found for this batch, skipping...")
                continue
            
            # Build graph
            print("\n🔗 Building citation graph...")
            G = build_citation_graph(papers_df, undirected=True)
            print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Largest component
            if largest_component:
                print("   Extracting largest connected component...")
                G = get_largest_connected_component(G)
                print(f"   Component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Sample if needed
            if G.number_of_nodes() > args.max_nodes:
                print(f"   ⚠️ Sampling to {args.max_nodes} nodes...")
                import random
                random.seed(42)
                sampled_nodes = random.sample(list(G.nodes()), args.max_nodes)
                G = G.subgraph(sampled_nodes).copy()
                print(f"   Sampled: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            graph_nodes = set(G.nodes())
            
            # Run Girvan-Newman
            print(f"\n🔬 Running Girvan-Newman (max {args.max_levels} levels)...")
            print_memory("Before GN ")
            
            def progress_callback(level, num_communities, modularity):
                if args.verbose or level % 10 == 0:
                    print(f"   Level {level}: {num_communities} communities, modularity={modularity:.4f}")
            
            result = detect_girvan_newman_best_modularity(
                G,
                max_levels=args.max_levels,
                progress_callback=progress_callback
            )
            
            print(f"\n✅ Batch completed in {result.elapsed_seconds:.1f}s")
            print(f"   Best modularity: {result.modularity:.4f}")
            print(f"   Communities found: {result.num_communities}")
            
            # Convert to DataFrame
            communities_df = partition_to_communities_df(result.partition, G)
            
            # Build edges DataFrame
            papers_in_graph = papers_df[papers_df['id'].isin(graph_nodes)]
            edges_df = build_edges_dataframe(papers_in_graph)
            
            # Save batch results
            batch_name = f"batch_{batch_num}"
            output_dir = DATA_DIR / "batches"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"communities_{batch_name}.csv"
            save_communities_csv(communities_df, output_path)
            
            edges_path = output_dir / f"edges_{batch_name}.csv"
            save_edges_csv(edges_df, edges_path)
            
            meta_path = output_dir / f"metadata_{batch_name}.json"
            result.parameters['topics'] = batch_topics
            save_metadata_json(result, meta_path)
            
            print(f"💾 Saved to {output_dir}/")
            
            all_results.append({
                'batch': batch_num,
                'topics': batch_topics,
                'modularity': result.modularity,
                'communities': result.num_communities,
                'nodes': result.num_nodes,
                'edges': result.num_edges
            })
            
            # CRITICAL: Clean up memory
            del papers_df, G, communities_df, edges_df, papers_in_graph, result
            gc.collect()
            print_memory("After cleanup ")
        
        # Print summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        for r in all_results:
            print(f"Batch {r['batch']} ({', '.join(r['topics'][:3])}...): "
                  f"{r['communities']} communities, modularity={r['modularity']:.4f}")
        print(f"\nAll results saved to: {DATA_DIR / 'batches'}/")
    
    else:
        # Single topic or no topic filter - original behavior
        print(f"\n📚 Loading papers from {db_path}...")
        if args.topic:
            print(f"   Filtering by topic: {args.topic}")
        if args.limit:
            print(f"   Limiting to {args.limit} papers")
        
        papers_df = load_papers_with_citations(
            db_path=db_path,
            topic_filter=args.topic,
            limit=args.limit
        )
        print(f"   Loaded {len(papers_df)} papers")
        
        if len(papers_df) == 0:
            print("❌ No papers found!")
            import sys
            sys.exit(1)
        
        # Build graph
        print("\n🔗 Building citation graph...")
        G = build_citation_graph(papers_df, undirected=True)
        print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Largest component
        if largest_component:
            print("\n📊 Extracting largest connected component...")
            G = get_largest_connected_component(G)
            print(f"   Component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Sample if needed
        if G.number_of_nodes() > args.max_nodes:
            print(f"\n⚠️  Graph exceeds {args.max_nodes} nodes, sampling...")
            import random
            random.seed(42)
            sampled_nodes = random.sample(list(G.nodes()), args.max_nodes)
            G = G.subgraph(sampled_nodes).copy()
            print(f"   Sampled: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        graph_nodes = set(G.nodes())
        
        # Run Girvan-Newman
        print(f"\n🔬 Running Girvan-Newman (max {args.max_levels} levels)...")
        
        def progress_callback(level, num_communities, modularity):
            if args.verbose or level % 10 == 0:
                print(f"   Level {level}: {num_communities} communities, modularity={modularity:.4f}")
        
        result = detect_girvan_newman_best_modularity(
            G,
            max_levels=args.max_levels,
            progress_callback=progress_callback
        )
        
        print(f"\n✅ Completed in {result.elapsed_seconds:.1f} seconds")
        print(f"   Levels evaluated: {result.levels_evaluated}")
        print(f"   Best modularity: {result.modularity:.4f}")
        print(f"   Communities found: {result.num_communities}")
        
        # Convert to DataFrame
        communities_df = partition_to_communities_df(result.partition, G)
        
        # Build edges DataFrame
        papers_in_graph = papers_df[papers_df['id'].isin(graph_nodes)]
        edges_df = build_edges_dataframe(papers_in_graph)
        
        # Validate if requested
        if args.test:
            print("\n📋 Validating output...")
            errors = []
            
            if communities_df['id'].duplicated().any():
                dup_count = communities_df['id'].duplicated().sum()
                errors.append(f"Found {dup_count} duplicate IDs")
            
            output_ids = set(communities_df['id'])
            missing = graph_nodes - output_ids
            if missing:
                errors.append(f"Missing {len(missing)} node IDs from output")
            
            communities = sorted(communities_df['community'].unique())
            expected = list(range(len(communities)))
            if communities != expected:
                errors.append(f"Community labels not consecutive: got {communities[:10]}...")
            
            if errors:
                print("❌ Validation failed:")
                for err in errors:
                    print(f"   - {err}")
                import sys
                sys.exit(1)
            
            print("✅ Validation passed!")
            print(f"   - {len(communities_df)} unique paper IDs")
            print(f"   - {len(communities)} communities (0 to {len(communities)-1})")
        
        # Save outputs
        output_path = Path(args.output) if args.output else DATA_DIR / "communities_gn.csv"
        
        print(f"\n💾 Saving outputs...")
        save_communities_csv(communities_df, output_path)
        print(f"   Communities: {output_path}")
        
        edges_path = output_path.parent / "edges.csv"
        save_edges_csv(edges_df, edges_path)
        print(f"   Edges: {edges_path}")
        
        meta_path = output_path.with_suffix(".json").with_name(
            output_path.stem + "_meta.json"
        )
        save_metadata_json(result, meta_path)
        print(f"   Metadata: {meta_path}")
    
    print("\n" + "=" * 60)
    print("Done! Load results in UI: streamlit run app.py")
    print("=" * 60)
