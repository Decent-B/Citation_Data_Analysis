"""
Data loading utilities for paper ranking.

This module provides functions to load paper data from SQLite databases
and build citation graphs for analysis.
"""

import sqlite3
import pandas as pd
from typing import Union, List, Optional

from ranking.graph import parse_list_cell, build_edges_df, build_cugraph_with_mapping


def load_topic_data(db_path: str, target_topics: Union[str, List[str], None] = None) -> pd.DataFrame:
    """
    Load papers from the database, optionally filtering by topic(s).
    
    Args:
        db_path: Path to SQLite database
        target_topics: Single topic string, list of topics, or None for all papers
    
    Returns:
        DataFrame containing paper data with columns:
        - id: Paper ID
        - doi: Digital Object Identifier
        - title: Paper title
        - topic: Topic classification
        - referenced_works_count: Number of references
        - referenced_works: JSON list of referenced paper IDs
        - authors: JSON list of author IDs
        - cited_by_count: Number of citations
        - publication_date: Publication date
        - related_works: JSON list of related paper IDs
    """
    conn = sqlite3.connect(db_path)
    
    if target_topics:
        # Convert single string to list for uniform handling
        topics = [target_topics] if isinstance(target_topics, str) else target_topics
        
        # Build SQL query with IN clause for multiple topics
        if len(topics) == 1:
            topic_filter = f"WHERE topic = '{topics[0]}'"
        else:
            topic_list = "', '".join(topics)
            topic_filter = f"WHERE topic IN ('{topic_list}')"
        
        query = f"SELECT * FROM works {topic_filter}"
        print(f"Loading papers for topic(s): {topics}")
    else:
        query = "SELECT * FROM works"
        print("Loading all papers from database")
    
    topic_df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(topic_df)} papers")
    return topic_df


def build_graph(db_path: str, target_topics: Union[str, List[str], None] = None):
    """
    Build citation graph from database.
    
    Loads paper data and creates an edge list representing citation relationships.
    Only includes edges where both source and destination papers are in the dataset.
    
    Args:
        db_path: Path to SQLite database
        target_topics: Single topic string, list of topics, or None for all papers
    
    Returns:
        Tuple of (topic_df, edges_df) where:
        - topic_df: DataFrame with paper data (includes 'referenced_works_parsed' column)
        - edges_df: DataFrame with 'source' and 'target' columns
    """
    topic_df = load_topic_data(db_path, target_topics)
    
    # Parse referenced works
    topic_df['referenced_works_parsed'] = topic_df['referenced_works'].apply(parse_list_cell)
    
    # Build edge list
    edges_df = build_edges_df(topic_df)
    print(f"Built {len(edges_df)} edges")

    return topic_df, edges_df

def build_cugraph(db_path: str, target_topics: Union[str, List[str], None] = None):
    """
    Build citation graph using cuGraph (GPU) from database.
    
    Args:
        db_path: Path to SQLite database
        target_topics: Single topic string, list of topics, or None for all papers
    
    Returns:
        Tuple of (topic_df, edges_df, G_gpu, vid_df) where:
        - topic_df: DataFrame with paper data
        - edges_df: DataFrame with edges
        - G_gpu: cuGraph Graph object (or None if no edges)
        - vid_df: DataFrame mapping vertex_id to int_id (or None if no edges)
    """
    topic_df, edges_df = build_graph(db_path, target_topics)
    
    if len(edges_df) == 0:
        print("No edges to build graph.")
        return topic_df, edges_df, None, None

    # Create cuGraph Graph with vertex ID mapping
    G_gpu, vid_df = build_cugraph_with_mapping(edges_df)
    
    print(f"cuGraph graph constructed: {G_gpu.number_of_vertices()} vertices, {G_gpu.number_of_edges()} edges")
    return topic_df, edges_df, G_gpu, vid_df


if __name__ == "__main__":
    DB_PATH = "data/openalex_works-ver2.db"
    TARGET_TOPICS = ["T10181", "T20234"]  # Example topics

    topic_df, edges_df = build_graph(DB_PATH, TARGET_TOPICS)
    print(topic_df.head())
    print(edges_df.head())