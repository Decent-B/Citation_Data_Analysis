"""
Citation Data Analysis Streamlit UI

A web interface for searching papers and visualizing community detection results.
"""

import streamlit as st
from pathlib import Path

# Import tab implementations
from ui.search_tab import render_search_tab
from ui.community_tab import render_community_tab
from ui.data_access import load_papers_metadata
from ui.config import TARGET_TOPICS

# Page configuration
st.set_page_config(
    page_title="Citation Data Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data(show_spinner="Loading papers metadata...")
def get_papers_data():
    """
    Load and cache papers metadata for the entire session.
    This is called once when the app starts and cached for performance.
    """
    return load_papers_metadata()

def main():
    """Main application entry point."""
    st.title("📚 Citation Data Analysis")
    
    # Load papers metadata once at app startup
    papers_df = get_papers_data()
    
    # Display database stats
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Total Papers", f"{len(papers_df):,}")
    with col2:
        # Show data source: 5 topics vs placeholder
        topics_loaded = len(papers_df['topic'].unique()) if 'topic' in papers_df.columns else 0
        data_source = f"{topics_loaded} Topics" if topics_loaded > 1 else "Placeholder"
        st.metric("Data Source", data_source)
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Search", "🌐 Community Detection"])
    
    with tab1:
        render_search_tab(papers_df)
    
    with tab2:
        render_community_tab(papers_df)

if __name__ == "__main__":
    main()
