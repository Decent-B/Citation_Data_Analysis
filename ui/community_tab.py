"""Community detection tab UI implementation."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple


def load_community_file(algorithm: str) -> Optional[pd.DataFrame]:
    """
    Load community detection results from CSV file.
    
    Args:
        algorithm: Algorithm name ('Leiden' or 'Girvan-Newman')
        
    Returns:
        DataFrame with paper_id and cluster_id columns, or None if file not found
    """
    if algorithm == "Leiden":
        file_path = Path("results/leiden_communities.csv")
    elif algorithm == "Girvan-Newman":
        file_path = Path("results/checkpoint_level_10.csv")
    else:
        return None
    
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Ensure columns are named correctly
        if 'paper_id' in df.columns and 'cluster_id' in df.columns:
            # Ensure paper_id is string and cluster_id is int
            df['paper_id'] = df['paper_id'].astype(str)
            df['cluster_id'] = df['cluster_id'].astype(int)
            return df
        else:
            st.error(f"❌ CSV file must have 'paper_id' and 'cluster_id' columns. Found: {list(df.columns)}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {str(e)}")
        return None


def load_edges_file() -> Optional[pd.DataFrame]:
    """
    Load citation edges from CSV file.
    
    Returns:
        DataFrame with source_id and target_id columns, or None if file not found
    """
    # edges.csv is not available, return None
    # Edges will be extracted from papers_df referenced_works instead
    return None

def extract_edges_from_papers(papers_df: pd.DataFrame, valid_paper_ids: set) -> pd.DataFrame:
    """
    Extract citation edges from papers_df referenced_works column.
    Only includes edges where both source and target are in valid_paper_ids.
    
    Args:
        papers_df: DataFrame with 'id' and 'referenced_works' columns
        valid_paper_ids: Set of paper IDs to include
        
    Returns:
        DataFrame with source_id and target_id columns
    """
    import ast
    import json
    
    def parse_references(ref_str):
        """Parse referenced_works field into list of IDs."""
        # Handle None/NaN
        if ref_str is None:
            return []
        
        # Use scalar check to avoid ambiguous truth value error
        if isinstance(ref_str, str):
            # Empty string check
            if ref_str == '':
                return []
            # Try parsing as JSON first, then as Python literal
            try:
                return json.loads(ref_str)
            except:
                try:
                    return ast.literal_eval(ref_str)
                except:
                    return []
        elif isinstance(ref_str, list):
            return ref_str
        else:
            # Try using pd.isna on non-string, non-list values
            try:
                if pd.isna(ref_str):
                    return []
            except:
                pass
            return []
    
    edges = []
    for _, row in papers_df.iterrows():
        source_id = str(row['id'])
        if source_id not in valid_paper_ids:
            continue
            
        if 'referenced_works' in row:
            refs = parse_references(row['referenced_works'])
            for target_id in refs:
                target_id = str(target_id)
                if target_id in valid_paper_ids:
                    edges.append({'source_id': source_id, 'target_id': target_id})
    
    return pd.DataFrame(edges) if edges else pd.DataFrame(columns=['source_id', 'target_id'])


def create_community_graph(communities_df: pd.DataFrame, papers_df: pd.DataFrame, edges_df: Optional[pd.DataFrame] = None):
    """
    Create an interactive community graph visualization using PyVis.
    
    Args:
        communities_df: DataFrame with paper_id and cluster_id columns
        papers_df: DataFrame with paper metadata (id, title, doi, publication_date, cited_by_count, referenced_works)
        edges_df: Optional DataFrame with source_id and target_id columns (will be extracted from papers_df if None)
        
    Returns:
        HTML string for the graph visualization
    """
    # Get valid paper IDs (intersection of communities and papers metadata)
    community_ids = set(communities_df['paper_id'].astype(str))
    print(f"Number of community paper IDs: {len(community_ids)}")
    paper_ids = set(papers_df['id'].astype(str))
    print(f"Number of paper IDs in database: {len(paper_ids)}")
    valid_paper_ids = community_ids & paper_ids
    
    # Debug info
    if len(valid_paper_ids) == 0:
        # Try to diagnose the issue
        sample_community = list(community_ids)[:5] if len(community_ids) > 0 else []
        sample_papers = list(paper_ids)[:5] if len(paper_ids) > 0 else []
        raise ValueError(
            f"No papers found in both communities and metadata.\n"
            f"Communities have {len(community_ids)} paper IDs (sample: {sample_community})\n"
            f"Metadata has {len(paper_ids)} paper IDs (sample: {sample_papers})\n"
            f"Check that paper ID formats match (e.g., with/without 'W' prefix)"
        )
    
    # Filter communities to only valid papers
    communities_df = communities_df[communities_df['paper_id'].isin(valid_paper_ids)].copy()
    
    # Merge communities with metadata
    merged = communities_df.merge(
        papers_df[['id', 'title', 'doi', 'publication_date', 'cited_by_count']],
        left_on='paper_id',
        right_on='id',
        how='inner'
    )
    
    if len(merged) == 0:
        raise ValueError("No papers found after merging communities with metadata")
    
    # Extract edges from papers_df if not provided
    if edges_df is None or len(edges_df) == 0:
        # Only extract edges if papers_df has referenced_works column
        if 'referenced_works' in papers_df.columns:
            edges_df = extract_edges_from_papers(papers_df, valid_paper_ids)
        else:
            # Create empty edges DataFrame if column not available
            edges_df = pd.DataFrame(columns=['source_id', 'target_id'])
    
    # Create NetworkX graph for layout
    G = nx.Graph()
    
    # Add nodes
    for _, row in merged.iterrows():
        G.add_node(row['paper_id'], 
                   cluster_id=row['cluster_id'],
                   title=row.get('title', 'Unknown'),
                   doi=row.get('doi', ''),
                   date=row.get('publication_date', ''),
                   citations=row.get('cited_by_count', 0))
    
    # Add edges if available
    has_edges = False
    if edges_df is not None and len(edges_df) > 0:
        # Only add edges between nodes that exist
        node_set = set(G.nodes())
        for _, row in edges_df.iterrows():
            source = str(row['source_id'])
            target = str(row['target_id'])
            if source in node_set and target in node_set:
                G.add_edge(source, target)
                has_edges = True
    
    # Compute layout - cluster nodes by community
    import math
    import random
    
    # Group nodes by community
    communities = merged.groupby('cluster_id')['paper_id'].apply(list).to_dict()
    num_communities = len(communities)
    pos = {}
    
    # Arrange communities in a circular pattern (more compact)
    for idx, (cluster_id, nodes) in enumerate(communities.items()):
        # Position each community center around a smaller circle for compactness
        angle = 2 * math.pi * idx / num_communities
        center_x = 300 * math.cos(angle)  # Reduced from 500 to 300
        center_y = 300 * math.sin(angle)  # Reduced from 500 to 300
        
        # Create a subgraph for this community
        community_nodes = set(nodes)
        subgraph = G.subgraph(community_nodes).copy()
        
        # Use spring layout for nodes within the community
        if len(nodes) > 1:
            try:
                # Spring layout for internal community structure
                sub_pos = nx.spring_layout(
                    subgraph, 
                    k=0.3,  # Even smaller k for tighter clustering (was 0.5)
                    iterations=30,
                    scale=50,  # Reduced scale for more compact clusters (was 80)
                    seed=42 + idx
                )
                # Offset positions to the community center
                for node, (x, y) in sub_pos.items():
                    pos[node] = (center_x + x, center_y + y)
            except:
                # Fallback: random scatter around center
                for node in nodes:
                    random.seed(hash(node) % 2**32)
                    offset_x = random.uniform(-50, 50)  # Reduced from -80, 80
                    offset_y = random.uniform(-50, 50)  # Reduced from -80, 80
                    pos[node] = (center_x + offset_x, center_y + offset_y)
        else:
            # Single node - place at center
            pos[nodes[0]] = (center_x, center_y)
    
    # Create PyVis network with white background and zoom/pan enabled
    net = Network(height="750px", width="100%", bgcolor="#ffffff")
    
    # Disable physics to keep nodes fixed in position
    net.set_options("""
    {
        "physics": {
            "enabled": false
        },
        "interaction": {
            "hover": true,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true
        }
    }
    """)
    
    # Generate distinct colors for each community
    import colorsys
    
    def generate_distinct_colors(n):
        """Generate n visually distinct colors using golden ratio for better spacing."""
        colors = []
        golden_ratio = 0.618033988749895
        
        for i in range(n):
            # Use golden ratio to maximize color spacing
            hue = (i * golden_ratio) % 1.0
            # Alternate saturation and brightness for additional distinction
            saturation = 0.65 + (i % 4) * 0.1
            value = 0.75 + (i % 3) * 0.1
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors
    
    # Generate colors based on actual number of communities
    colors = generate_distinct_colors(num_communities)
    
    # Create a mapping from cluster_id to color index
    # Map cluster IDs to colors in circular order (adjacent clusters get distant colors)
    cluster_ids = sorted(communities.keys())
    color_map = {cluster_id: colors[idx] for idx, cluster_id in enumerate(cluster_ids)}
    
    # Add nodes to PyVis - larger size for better visibility
    node_size = 25  # Increased from 15 for better visibility
    
    for node in G.nodes():
        data = G.nodes[node]
        cluster_id = data['cluster_id']
        color = color_map[cluster_id]  # Use the color map instead of modulo
        
        # Create hover tooltip with enhanced information
        title = data.get('title', 'Unknown')
        doi = data.get('doi', 'N/A')
        date = data.get('date', 'N/A')
        citations = data.get('citations', 0)
        
        # Escape HTML special characters to prevent formatting issues
        title = str(title).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
        doi = str(doi).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        tooltip = (
            f"Title: {title}\n"
            f"Date: {date}\n"
            f"Citations: {citations}\n"
            f"DOI: {doi}\n"
            f"Community: {cluster_id}"
        )
        
        # Get position
        x, y = pos[node]
        
        net.add_node(
            node,
            label=str(cluster_id),
            title=tooltip,
            color=color,
            x=float(x) * 100,
            y=float(y) * 100,
            size=node_size
        )
    
    # Add edges to PyVis with subtle visibility
    if has_edges:
        for edge in G.edges():
            net.add_edge(edge[0], edge[1], color='#cccccc', width=0.3)  # Reduced from 1.5 to 0.3, lighter color
    
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
    
    Args:
        papers_df: Pre-loaded papers metadata DataFrame with columns:
                  id, title, doi, publication_date, cited_by_count
    """
    
    st.markdown("## 🌐 Community Detection Visualization")
    st.markdown("Visualize communities detected by different algorithms on the citation network.")
    
    # Algorithm selection
    algorithm = st.selectbox(
        "Select Community Detection Algorithm",
        ["Leiden", "Girvan-Newman"],
        help="Choose the algorithm whose results you want to visualize"
    )
    
    # Display algorithm info
    if algorithm == "Leiden":
        st.info("📁 Loading from: `results/leiden_communities.csv`")
    else:
        st.info("📁 Loading from: `results/checkpoint_level_10.csv`")
    
    # Detect/Load button
    if st.button("🔍 Load and Visualize Communities", type="primary"):
        with st.spinner(f"Loading {algorithm} community detection results..."):
            # Load community data
            communities_df = load_community_file(algorithm)
            
            if communities_df is None:
                st.error(
                    f"❌ Community data not found. Expected file:\n"
                    f"- `results/leiden_communities.csv` (Leiden)\n"
                    f"- `results/checkpoint_level_10.csv` (Girvan-Newman)\n\n"
                    f"File must have columns: **paper_id** (str) and **cluster_id** (int)"
                )
                return
            
            # Load edges (optional)
            edges_df = load_edges_file()
            
            # Check papers_df has required columns
            required_columns = ['id', 'title', 'publication_date', 'cited_by_count']
            missing_columns = [col for col in required_columns if col not in papers_df.columns]
            if missing_columns:
                st.error(f"❌ Papers metadata missing required columns: {missing_columns}")
                return
            
            # Limit graph size for performance
            MAX_NODES = 3000
            if len(communities_df) > MAX_NODES:
                st.warning(f"⚠️ Graph has {len(communities_df):,} nodes. Sampling {MAX_NODES} for visualization.")
                communities_df = communities_df.sample(n=MAX_NODES, random_state=10)
            
            # Community statistics
            community_counts = Counter(communities_df['cluster_id'])
            num_communities = len(community_counts)
            
            st.success(f"✅ Loaded {num_communities} communities from {len(communities_df):,} papers")
            
            # Summary panel
            with st.expander("📊 Community Summary", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Communities", f"{num_communities:,}")
                
                with col2:
                    st.metric("Total Papers", f"{len(communities_df):,}")
                
                with col3:
                    avg_size = len(communities_df) / num_communities
                    st.metric("Avg Community Size", f"{avg_size:.1f}")
                
                # Top communities
                st.markdown("**Top 10 Largest Communities:**")
                top_communities = community_counts.most_common(10)
                
                # Create a mini table
                comm_data = []
                for cluster_id, count in top_communities:
                    percentage = (count / len(communities_df)) * 100
                    comm_data.append({
                        "Community ID": cluster_id,
                        "Papers": count,
                        "Percentage": f"{percentage:.1f}%"
                    })
                st.dataframe(pd.DataFrame(comm_data), use_container_width=True, hide_index=True)
            
            # Create and display graph
            st.markdown("### 🎨 Interactive Network Visualization")
            
            # Add diagnostic info
            st.info(f"📊 Database has {len(papers_df):,} papers loaded. Community file has {len(communities_df):,} papers.")
            
            try:
                html_content = create_community_graph(communities_df, papers_df, edges_df)
                components.html(html_content, height=800, scrolling=False)
                
                # Download option
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    csv = communities_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Community Data (CSV)",
                        data=csv,
                        file_name=f"{algorithm.lower()}_communities.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Visualization (HTML)",
                        data=html_content,
                        file_name=f"{algorithm.lower()}_visualization.html",
                        mime="text/html"
                    )
                
            except ValueError as e:
                st.error(f"❌ Visualization Error: {str(e)}")
                
                # Provide helpful diagnostic information
                st.markdown("**Diagnostic Information:**")
                community_ids_sample = communities_df['paper_id'].head(10).tolist()
                paper_ids_sample = papers_df['id'].head(10).tolist()
                
                st.markdown(f"- Community paper IDs (first 10): `{community_ids_sample}`")
                st.markdown(f"- Database paper IDs (first 10): `{paper_ids_sample}`")
                st.markdown(f"- Total communities: {len(communities_df):,}")
                st.markdown(f"- Total papers in DB: {len(papers_df):,}")
                
                # Check for intersection
                common = set(communities_df['paper_id'].astype(str)) & set(papers_df['id'].astype(str))
                st.markdown(f"- **Papers in both: {len(common):,}**")
                
                if len(common) == 0:
                    st.warning(
                        "⚠️ **No matching papers found!** This could happen if:\n"
                        "1. You used `--filter-communities` during migration but the community files changed\n"
                        "2. The database was migrated with different topic filters\n"
                        "3. Paper ID formats don't match\n\n"
                        "**Solution:** Re-run the migration with `--filter-communities` flag:\n"
                        "```bash\n"
                        "python3 database/migrate.py --filter-communities\n"
                        "```"
                    )
                
            except Exception as e:
                st.error(f"❌ Error creating visualization: {str(e)}")
                st.exception(e)
