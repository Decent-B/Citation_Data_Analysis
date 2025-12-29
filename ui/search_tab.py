"""Search tab UI implementation with side-by-side algorithm comparison."""

import streamlit as st
import pandas as pd
from ui.data_access import get_papers_by_ids
from ui.search import run_both_searches


def render_result_column(results_df: pd.DataFrame):
    """Render search results in a compact format."""
    if len(results_df) == 0:
        st.info("No results found")
        return
    
    for rank, (idx, row) in enumerate(results_df.iterrows(), 1):
        # Compact format: Rank. Title
        title = row['title'][:70] + "..." if len(row['title']) > 70 else row['title']
        
        # Build info line
        date_str = str(row['publication_date'])[:10] if row['publication_date'] else "N/A"
        cited_count = int(row.get('cited_by_count', 0))
        
        # Display compact format
        st.markdown(f"**{rank}.** {title}")
        st.caption(f"📅 {date_str} | 📚 {cited_count:,} citations")


def render_search_tab(papers_df: pd.DataFrame):
    """
    Render the search tab interface with side-by-side algorithm comparison.
    
    Args:
        papers_df: Pre-loaded papers metadata DataFrame
    """
    
    # Initialize session state
    if 'keyword_results' not in st.session_state:
        st.session_state.keyword_results = None
    if 'pagerank_results' not in st.session_state:
        st.session_state.pagerank_results = None
    
    # Search form
    with st.form(key="search_form"):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="Enter keywords to search papers...",
                help="Press Enter or click Search to submit"
            )
        
        with col2:
            k = st.number_input(
                "Top-K",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of results"
            )
        
        submit = st.form_submit_button("🔍 Search", use_container_width=True)
    
    # Process search - run BOTH algorithms simultaneously
    if submit:
        if not query or query.strip() == "":
            st.warning("⚠️ Please enter a search query")
        else:
            with st.spinner("Searching with both algorithms simultaneously..."):
                # Run both searches at once (more efficient)
                keyword_ids, pagerank_ids = run_both_searches(query, k, papers_df)
                
                # Get paper details for display
                st.session_state.keyword_results = get_papers_by_ids(papers_df, keyword_ids)
                st.session_state.pagerank_results = get_papers_by_ids(papers_df, pagerank_ids)
    
    # Display results side by side
    if st.session_state.keyword_results is not None and st.session_state.pagerank_results is not None:
        keyword_df = st.session_state.keyword_results
        pagerank_df = st.session_state.pagerank_results
        
        st.success(f"Found {len(keyword_df)} results per algorithm")
        st.markdown("---")
        
        # Create two columns for side-by-side display
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.subheader("🔤 Keyword Matches")
            render_result_column(keyword_df)
        
        with right_col:
            st.subheader("📊 Keyword + PageRank")
            render_result_column(pagerank_df)
