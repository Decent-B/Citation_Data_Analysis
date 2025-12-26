"""Search tab UI implementation."""

import streamlit as st
from ui.data_access import load_papers_metadata, get_papers_by_ids
from ui.search import run_search

def render_search_tab():
    """Render the search tab interface."""
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    # Load papers metadata
    papers_df = load_papers_metadata()
    
    # Search controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        algorithm = st.selectbox(
            "Search Algorithm",
            ["BM25", "PageRank + BM25", "HITS"],
            help="Select the search algorithm to use"
        )
    
    with col2:
        k = st.number_input(
            "Top-K Results",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of results to return"
        )
    
    # Search form
    with st.form(key="search_form"):
        query = st.text_input(
            "Search Query",
            placeholder="Enter keywords to search papers...",
            help="Press Enter or click Search to submit"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit:
        if not query or query.strip() == "":
            st.warning("⚠️ Please enter a search query")
        else:
            with st.spinner(f"Searching with {algorithm}..."):
                result_ids = run_search(query, algorithm, k, papers_df)
                st.session_state.search_results = get_papers_by_ids(papers_df, result_ids)
    
    # Display results
    if st.session_state.search_results is not None:
        results_df = st.session_state.search_results
        
        if len(results_df) == 0:
            st.info("No results found")
        else:
            st.success(f"Found {len(results_df)} results")
            st.markdown("---")
            
            # Render each result as a card
            for idx, row in results_df.iterrows():
                with st.container():
                    # Title (large and bold)
                    st.markdown(f"### {row['title']}")
                    
                    # DOI and publication date (smaller, muted)
                    info_parts = []
                    
                    if row['doi'] and row['doi'].strip():
                        doi_link = f"https://doi.org/{row['doi']}"
                        info_parts.append(f"**DOI:** [{row['doi']}]({doi_link})")
                    else:
                        info_parts.append("**DOI:** _Not available_")
                    
                    if row['publication_date'] and row['publication_date'].strip():
                        info_parts.append(f"**Published:** {row['publication_date']}")
                    else:
                        info_parts.append("**Published:** _Unknown_")
                    
                    st.markdown(" | ".join(info_parts))
                    
                    # Paper ID (very small)
                    st.caption(f"Paper ID: {row['id']}")
                    
                    st.markdown("---")
