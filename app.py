"""
Citation Data Analysis Streamlit UI

A web interface for searching papers and visualizing community detection results.
"""

import streamlit as st
from pathlib import Path

# Import tab implementations
from ui.search_tab import render_search_tab
from ui.community_tab import render_community_tab

# Page configuration
st.set_page_config(
    page_title="Citation Data Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    """Main application entry point."""
    st.title("📚 Citation Data Analysis")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Search", "🌐 Community Detection"])
    
    with tab1:
        render_search_tab()
    
    with tab2:
        render_community_tab()

if __name__ == "__main__":
    main()
