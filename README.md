# Citation Data Analysis

A citation data analysis project with web scraping, network analysis, and PageRank computation capabilities.

## Features

- OpenAlex API data scraping
- Citation network construction
- PageRank computation (CPU with NetworkX, GPU with cuGraph)
- Interactive Streamlit UI for search and community detection

## Installation

This project uses Python 3.12+. Install dependencies using:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Running the Streamlit UI

Launch the interactive web interface:

```bash
streamlit run app.py
```

The UI provides two main features:

### 🔍 Search Tab
- Search papers by keywords, title, paper ID, or DOI
- Choose between different ranking algorithms (BM25, PageRank + BM25, HITS)
- View results with title, DOI, and publication date
- Click DOI links to access papers directly

### 🌐 Community Detection Tab
- Run community detection algorithms (Girvan-Newman, Kernighan-Lin, Louvain)
- Visualize paper communities in an interactive graph
- Hover over nodes to see paper details
- View community statistics and sizes

## Data Files

The UI loads paper metadata from the SQLite database:

**Required:**
- `data/openalex_works.db` - SQLite database with paper metadata
  - Table: `works`
  - Key columns: `id`, `title`, `doi`, `publication_date`, `referenced_works`

**Optional CSV files for community detection:**
- `data/communities.csv` - Community assignments (columns: `id`, `community`)
- `data/communities_gn.csv` - Girvan-Newman communities (optional, falls back to `communities.csv`)
- `data/communities_kl.csv` - Kernighan-Lin communities (optional)
- `data/communities_louvain.csv` - Louvain communities (optional)
- `data/edges.csv` - Citation edges (columns: `source_id`, `target_id`) (optional)

**Note:** The database includes paper titles. Community CSV files should use `id` as the column name to match the database schema.

## Data Scraping

Scrape papers from OpenAlex API:

```bash
python data_scraping/openalex_scraper.py
```

This creates the SQLite database at `data/openalex_works.db` with paper metadata and citations.

The scraper:
- Fetches papers from OpenAlex API using cursor pagination
- Stores metadata: `id`, `title`, `doi`, `apc_list_price`, `topic`, `referenced_works`, `authors`, `cited_by_count`, `publication_date`
- Supports graceful shutdown (Ctrl+C) with cursor state persistence
- Auto-resumes from last cursor position

## Citation Network Analysis

See `citation_pagerank.ipynb` for examples of:
- Loading citation data from the SQLite database
- Building citation networks from `referenced_works`
- Computing PageRank rankings
- GPU-accelerated graph analysis with cuGraph

## Project Structure

```
.
├── app.py                      # Streamlit UI entry point
├── ui/                         # UI components
│   ├── search_tab.py          # Search interface
│   ├── community_tab.py       # Community detection interface
│   ├── search.py              # Search algorithms
│   └── data_access.py         # Data loading utilities (SQLite)
├── data_scraping/             # OpenAlex scraper
│   ├── openalex_scraper.py   # Main scraper
│   └── utils.py              # Helper functions (API calls)
├── data/                      # Data files (gitignored)
│   └── openalex_works.db     # SQLite database
├── citation_pagerank.ipynb   # Analysis notebook
└── pyproject.toml            # Dependencies
```

## License

MIT
