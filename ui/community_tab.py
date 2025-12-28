"""Community detection tab UI implementation.

This module handles loading and visualizing pre-computed community detection results.
The actual algorithms are implemented in scripts/community_detection.py.
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
from pathlib import Path
from collections import Counter
from typing import Optional
from ui.data_access import load_communities, load_edges

def create_community_graph(communities_df: pd.DataFrame, papers_df: pd.DataFrame, edges_df: Optional[pd.DataFrame] = None):
    """
    Create an interactive community graph visualization using PyVis.
    
    Args:
        communities_df: DataFrame with paper_id and community columns
        papers_df: DataFrame with paper metadata
        edges_df: Optional DataFrame with source_id and target_id columns
        
    Returns:
        HTML string for the graph visualization
    """
    # Merge communities with metadata
    merged = communities_df.merge(
        papers_df[['id', 'title', 'doi', 'publication_date']],
        on='id',
        how='left'
    )
    
    # Create NetworkX graph for layout
    G = nx.Graph()
    
    # Add nodes
    for _, row in merged.iterrows():
        G.add_node(row['id'], 
                   community=row['community'],
                   title=row.get('title', 'Unknown'),
                   doi=row.get('doi', ''),
                   date=row.get('publication_date', ''))
    
    # Add edges if available
    has_edges = False
    if edges_df is not None and len(edges_df) > 0:
        # Only add edges between nodes that exist
        node_set = set(G.nodes())
        for _, row in edges_df.iterrows():
            if row['source_id'] in node_set and row['target_id'] in node_set:
                G.add_edge(row['source_id'], row['target_id'])
                has_edges = True
    
    # Compute layout
    if has_edges:
        # Force-directed layout with edges
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    else:
        # Cluster by community without edges
        communities = merged.groupby('community')['id'].apply(list).to_dict()
        pos = {}
        import math
        num_communities = len(communities)
        
        for idx, (community, nodes) in enumerate(communities.items()):
            # Place each community around a circle
            angle = 2 * math.pi * idx / num_communities
            center_x = 500 * math.cos(angle)
            center_y = 500 * math.sin(angle)
            
            # Scatter nodes around the center
            for node_idx, node in enumerate(nodes):
                import random
                random.seed(hash(node) % 2**32)
                offset_x = random.uniform(-100, 100)
                offset_y = random.uniform(-100, 100)
                pos[node] = (center_x + offset_x, center_y + offset_y)
    
    # Create PyVis network
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100, spring_strength=0.001)
    
    # Color palette for communities
    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]
    
    # Add nodes to PyVis
    for node in G.nodes():
        data = G.nodes[node]
        community = data['community']
        color = colors[int(community) % len(colors)]
        
        # Create hover tooltip
        title = data.get('title', 'Unknown')
        doi = data.get('doi', 'N/A')
        date = data.get('date', 'N/A')
        
        tooltip = f"""
        <div style='font-family: Arial; padding: 10px;'>
            <b>Title:</b> {title}<br>
            <b>DOI:</b> {doi}<br>
            <b>Date:</b> {date}<br>
            <b>Community:</b> {community}
        </div>
        """
        
        # Get position
        x, y = pos[node]
        
        net.add_node(
            node,
            label=str(community),
            title=tooltip,
            color=color,
            x=float(x) * 100,
            y=float(y) * 100,
            size=15
        )
    
    # Add edges to PyVis
    if has_edges:
        for edge in G.edges():
            net.add_edge(edge[0], edge[1], color='#d3d3d3', width=0.5)
    
    # Generate HTML
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as html_file:
            html_content = html_file.read()
        Path(f.name).unlink()  # Clean up temp file
    
    return html_content


def render_community_tab(papers_df: pd.DataFrame):
    """
    Render the community detection tab interface.
    
    This tab loads pre-computed community detection results from CSV files.
    To generate new results, run the community_detection.py script:
        python scripts/community_detection.py
    Or use the CLI script:
        python scripts/community_detection.py --algorithm gn
    
    Args:
        papers_df: Pre-loaded papers metadata DataFrame
    """
    
    # Algorithm selection
    algorithm = st.selectbox(
        "Community Detection Algorithm",
        ["Girvan-Newman", "Kernighan-Lin", "Louvain"],
        help="Select the algorithm to load results for"
    )
    
    # Info about how to generate results
    with st.expander("ℹ️ How to generate community detection results", expanded=False):
        st.markdown("""
        Community detection results must be pre-computed using the command line:
        
        **Girvan-Newman:**
        ```bash
        python scripts/community_detection.py
        # Or with options:
        python scripts/community_detection.py --max-levels 50 --limit 500
        ```
        
        **Expected output files:**
        - `data/communities_gn.csv` - Girvan-Newman results
        - `data/communities_kl.csv` - Kernighan-Lin results  
        - `data/communities_louvain.csv` - Louvain results
        - `data/edges.csv` - Citation edges for visualization
        """)
    
    # Load button
    if st.button("🔍 Load Communities", type="primary"):
        with st.spinner(f"Loading {algorithm} community data..."):
            communities_df = load_communities(algorithm)
            
            if communities_df is None:
                st.error(
                    f"❌ Community data not found. Expected one of:\n"
                    f"- `data/communities_gn.csv` (Girvan-Newman)\n"
                    f"- `data/communities_kl.csv` (Kernighan-Lin)\n"
                    f"- `data/communities_louvain.csv` (Louvain)\n"
                    f"- `data/communities.csv` (fallback)\n\n"
                    f"Run `python scripts/community_detection.py` to generate results."
                )
                return
            
            edges_df = load_edges()
        
        # Check for required columns
        if 'id' not in communities_df.columns or 'community' not in communities_df.columns:
            st.error("❌ Community CSV must have 'id' and 'community' columns")
            return
        
        if 'id' not in papers_df.columns:
            st.error("❌ Papers metadata (data/openalex_works.db) must have 'id' column")
            return
        
        # Limit graph size for performance
        MAX_VIS_NODES = 1000
        if len(communities_df) > MAX_VIS_NODES:
            st.warning(f"⚠️ Graph has {len(communities_df)} nodes. Sampling {MAX_VIS_NODES} for visualization.")
            communities_df = communities_df.sample(n=MAX_VIS_NODES, random_state=42)
        
        # Community statistics
        community_counts = Counter(communities_df['community'])
        num_communities = len(community_counts)
        
        st.success(f"✅ Loaded {num_communities} communities from {len(communities_df)} papers")
        
        # Summary panel
        with st.expander("📊 Community Summary", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Communities", num_communities)
                st.metric("Total Papers", len(communities_df))
            
            with col2:
                top_communities = community_counts.most_common(10)
                st.markdown("**Top 10 Largest Communities:**")
                for comm, count in top_communities:
                    st.text(f"Community {comm}: {count} papers")
        
        # Create and display graph
        st.markdown("### 🌐 Community Graph Visualization")
        st.info("💡 Hover over nodes to see paper details (title, DOI, date)")
        
        try:
            html_content = create_community_graph(communities_df, papers_df, edges_df)
            st.components.v1.html(html_content, height=750, scrolling=True)
        except Exception as e:
            st.error(f"❌ Error creating visualization: {str(e)}")
            st.exception(e)
