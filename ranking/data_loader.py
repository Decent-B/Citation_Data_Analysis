"""
Data loading utilities for paper ranking.

This module provides functions to load paper data from PostgreSQL databases
and build citation graphs for analysis.
"""

import pandas as pd
from typing import Union, List, Optional
from sqlalchemy import text

from database.connection import get_engine
from ranking.graph import parse_list_cell, build_edges_df, build_cugraph_with_mapping


def load_topic_data(target_topics: Union[str, List[str], None] = None) -> pd.DataFrame:
    """
    Load papers from the database, optionally filtering by topic(s).
    
    Args:
        target_topics: Single topic string, list of topics, or None for all papers
    
    Returns:
        DataFrame containing paper data with columns:
        - id: Paper ID
        - doi: Digital Object Identifier
        - title: Paper title
        - topic: Topic classification
        - referenced_works_count: Number of references
        - referenced_works: JSON list of referenced paper IDs
        - authors: JSON array of author IDs
        - cited_by_count: Number of citations
        - publication_date: Publication date
        - related_works: JSON list of related paper IDs
    """
    engine = get_engine()
    
    # Build base query with aggregated referenced_works and related_works
    base_query = """
        SELECT 
            p.id,
            p.doi,
            p.title,
            p.topic,
            p.referenced_works_count,
            COALESCE(
                (SELECT json_agg(pr.cited_paper_id) 
                 FROM paper_references pr 
                 WHERE pr.citing_paper_id = p.id),
                '[]'::json
            ) as referenced_works,
            p.authors,
            p.cited_by_count,
            p.publication_date,
            COALESCE(
                (SELECT json_agg(rw.related_paper_id) 
                 FROM related_works rw 
                 WHERE rw.paper_id = p.id),
                '[]'::json
            ) as related_works
        FROM papers p
    """
    
    if target_topics:
        # Convert single string to list for uniform handling
        topics = [target_topics] if isinstance(target_topics, str) else target_topics
        
        # Build SQL query with parameterized IN clause (SQLAlchemy style)
        placeholders = ', '.join([f':topic_{i}' for i in range(len(topics))])
        query = f"{base_query} WHERE p.topic IN ({placeholders})"
        params = {f'topic_{i}': topic for i, topic in enumerate(topics)}
        print(f"Loading papers for topic(s): {topics}")
        topic_df = pd.read_sql_query(text(query), engine, params=params)
    else:
        query = base_query
        print("Loading all papers from database")
        topic_df = pd.read_sql_query(text(query), engine)
    
    print(f"Loaded {len(topic_df)} papers")
    return topic_df


def build_graph(target_topics: Union[str, List[str], None] = None):
    """
    Build citation graph from database.
    
    Loads paper data and creates an edge list representing citation relationships.
    Only includes edges where both source and destination papers are in the dataset.
    
    Args:
        target_topics: Single topic string, list of topics, or None for all papers
    
    Returns:
        Tuple of (topic_df, edges_df) where:
        - topic_df: DataFrame with paper data (includes 'referenced_works_parsed' column)
        - edges_df: DataFrame with 'source' and 'target' columns
    """
    topic_df = load_topic_data(target_topics)
    
    # Parse referenced works - handle both JSON arrays and string representations
    def parse_refs(val):
        if val is None:
            return []
        if isinstance(val, list):
            return val
        return parse_list_cell(val)
    
    topic_df['referenced_works_parsed'] = topic_df['referenced_works'].apply(parse_refs)
    
    # Build edge list
    edges_df = build_edges_df(topic_df)
    print(f"Built {len(edges_df)} edges")

    return topic_df, edges_df


def build_cugraph(target_topics: Union[str, List[str], None] = None):
    """
    Build citation graph using cuGraph (GPU) from database.
    
    Args:
        target_topics: Single topic string, list of topics, or None for all papers
    
    Returns:
        Tuple of (topic_df, edges_df, G_gpu, vid_df) where:
        - topic_df: DataFrame with paper data
        - edges_df: DataFrame with edges
        - G_gpu: cuGraph Graph object (or None if no edges)
        - vid_df: DataFrame mapping vertex_id to int_id (or None if no edges)
    """
    topic_df, edges_df = build_graph(target_topics)
    
    if len(edges_df) == 0:
        print("No edges to build graph.")
        return topic_df, edges_df, None, None

    # Create cuGraph Graph with vertex ID mapping
    G_gpu, vid_df = build_cugraph_with_mapping(edges_df)
    
    print(f"cuGraph graph constructed: {G_gpu.number_of_vertices()} vertices, {G_gpu.number_of_edges()} edges")
    return topic_df, edges_df, G_gpu, vid_df