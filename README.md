# Citation Data Analysis

A comprehensive citation network analysis project featuring data scraping from OpenAlex, community detection algorithms, and quantitative evaluation metrics for bibliometric research.

## Features

- **OpenAlex API data scraping** - Automated collection of academic paper metadata and citations
- **Citation network construction** - Build directed citation graphs from reference data
- **Community detection evaluation** - Comprehensive metrics (internal & external indices)
- **PageRank computation** - CPU (NetworkX) and GPU-accelerated (cuGraph) ranking
- **Interactive Streamlit UI** - Search papers and visualize community structures
- **Database integration** - Efficient SQLite storage for large-scale paper collections

## Installations

This project uses Python 3.12+. Install dependencies using:

```bash
docker compose up

uv sync
```

### Key Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **networkx** - Graph analysis and community detection
- **scikit-learn** - Machine learning metrics
- **streamlit** - Interactive web UI
- **sqlite3** - Database operations (included in Python standard library)

## Quick Start

### 1. Scrape Citation Data

```bash
python data_scraping/openalex_scraper.py
```

### 2. Run Community Detection Evaluation

```bash
# Activate virtual environment
source .venv/bin/activate

# Calculate metrics
python metrics.py
```

### 3. Launch Interactive UI

```bash
streamlit run app.py
```

## Community Detection Evaluation Metrics

This project implements a comprehensive suite of metrics for evaluating community detection algorithms on citation networks. See `metrics.py` for implementation details.

### Internal Indices (No Ground Truth Required)

These metrics evaluate partition quality based solely on graph structure:

| Metric | Range | Interpretation | Description |
|--------|-------|----------------|-------------|
| **Modularity (Q)** | [-0.5, 1.0] | Higher is better | Measures how much more internally linked communities are compared to random expectation. Uses directed graph formulation for citation networks. |
| **Coverage** | [0, 1] | Higher is better | Fraction of edges within communities (intra-community edges / total edges). Perfect score of 1.0 means all edges are internal. |
| **Performance** | [0, 1] | Higher is better | Fraction of correctly classified node pairs (both connected within communities or disconnected between communities). |
| **Average Conductance** | [0, 1] | Lower is better | Average fraction of edges leaving each community. Lower values indicate better isolation of communities. |


### External Indices (Require Ground Truth)

These metrics compare predicted communities against reference partitions (e.g., journal categories, field labels):

| Metric | Range | Interpretation | Description |
|--------|-------|----------------|-------------|
| **Adjusted Mutual Information (AMI)** | [-1, 1] | Higher is better | Mutual information adjusted for chance. Recommended over NMI when cluster counts vary. |
| **Adjusted Rand Index (ARI)** | [-1, 1] | Higher is better | Pair-counting similarity adjusted for chance. Robust to label permutations. |
| **Variation of Information (VI)** | [0, log(N)] | Lower is better | Information-theoretic distance between partitions. Measures information lost/gained. |
| **Normalized Mutual Information (NMI)** | [0, 1] | Higher is better | Mutual information normalized to [0,1]. Not adjusted for chance (prefer AMI). |
| **Homogeneity** | [0, 1] | Higher is better | Each cluster contains only members of a single class. |
| **Completeness** | [0, 1] | Higher is better | All members of a class are assigned to the same cluster. |
| **V-measure** | [0, 1] | Higher is better | Harmonic mean of homogeneity and completeness. |
| **Fowlkes-Mallows Index (FMI)** | [0, 1] | Higher is better | Geometric mean of pairwise precision and recall. |
| **Purity** | [0, 1] | Higher is better | Extent to which clusters contain single classes. Biased toward many small clusters. |


## Streamlit UI


The UI provides two main features:

### 🔍 Search Tab
- **Search papers** by keywords, title, paper ID, or DOI
- **Choose ranking algorithms:**
  - BM25 (text similarity)
  - PageRank + BM25 (citation importance + text similarity)
  - HITS (authority/hub scores)
- **View results** with title, DOI, and publication date
- **Click DOI links** to access papers directly

### 🌐 Community Detection Tab
- **Run algorithms:** Girvan-Newman, Kernighan-Lin, Louvain
- **Visualize communities** in interactive network graphs
- **Hover over nodes** to see paper details
- **View statistics:** community sizes and distributions


## Data Scraping

Scrape papers from OpenAlex API:

```bash
python data_scraping/openalex_scraper.py
```

**Features:**
- Fetches papers using cursor-based pagination
- Stores comprehensive metadata in SQLite
- Supports graceful shutdown (Ctrl+C) with state persistence
- Auto-resumes from last cursor position
- Configurable topic filters

**Stored Fields:**
- `id` - OpenAlex work identifier
- `title` - Paper title
- `doi` - Digital Object Identifier
- `publication_date` - Publication date
- `referenced_works` - List of cited paper IDs
- `topic` - OpenAlex topic classification
- `authors` - Author information
- `cited_by_count` - Citation count
- `apc_list_price` - Article Processing Charge

### Database Schema

```sql
CREATE TABLE works (
    id TEXT PRIMARY KEY,
    title TEXT,
    doi TEXT,
    publication_date TEXT,
    referenced_works TEXT,
    topic TEXT,
    authors TEXT,
    cited_by_count INTEGER,
    apc_list_price REAL
);
```

## Project Structure

```
Citation_Data_Analysis/
├── app.py                          # Streamlit UI entry point
├── metrics.py                      # Community detection evaluation metrics
├── utils.py                        # Edge extraction and helper functions
├── calculate_metrics_demo.py       # Demo script for metrics calculation
├── METRICS_USAGE_GUIDE.md         # Detailed metrics documentation
│
├── ui/                             # Streamlit UI components
│   ├── __init__.py
│   ├── search_tab.py              # Search interface
│   ├── community_tab.py           # Community visualization
│   ├── search.py                  # Search algorithms (BM25, HITS)
│   └── data_access.py             # Database access layer
│
├── data_scraping/                 # OpenAlex data collection
│   ├── openalex_scraper.py       # Main scraper script
│   ├── test_openalex_scraper.py  # Scraper tests
│   └── utils.py                  # API helper functions
│
├── database/                      # Database utilities
│   ├── __init__.py
│   ├── connection.py             # Database connections
│   ├── migrate.py                # Schema migrations
│   ├── schema.sql                # Database schema
│   └── verify_migration.py       # Migration verification
│
├── ranking/                       # PageRank and ranking algorithms
│   ├── __init__.py
│   ├── data_loader.py            # Data loading utilities
│   ├── graph.py                  # Graph construction
│   ├── main.py                   # Main ranking script
│   ├── pagerank.py               # PageRank implementation
│   ├── search.py                 # Search functionality
│   ├── similarity.py             # Similarity metrics
│   └── utils.py                  # Helper functions
│
├── data/                          # Data files (gitignored)
│   ├── openalex_works.db         # SQLite database
│   └── edges.csv                 # Citation edges (optional)
│
├── results/                       # Community detection results
│   ├── leiden_communities.csv    # Leiden algorithm output
│   └── ground_truth_communities.csv  # Reference partition
│
├── citation_pagerank.ipynb        # Analysis notebook
├── ranking.ipynb                  # Ranking experiments
├── pyproject.toml                 # Project dependencies (uv)
├── setup.py                       # Setup script
└── README.md                      # This file
```

## License

MIT

