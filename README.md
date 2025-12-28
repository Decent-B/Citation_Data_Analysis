# Citation Data Analysis

A comprehensive citation network analysis project featuring data scraping from OpenAlex, community detection algorithms, and quantitative evaluation metrics for bibliometric research.

## Features

- **OpenAlex API data scraping** - Automated collection of academic paper metadata and citations
- **Citation network construction** - Build directed citation graphs from reference data
- **Community detection evaluation** - Comprehensive metrics (internal & external indices)
- **PageRank computation** - CPU (NetworkX) and GPU-accelerated (cuGraph) ranking
- **Interactive Streamlit UI** - Search papers and visualize community structures
- **Database integration** - Efficient SQLite storage for large-scale paper collections

## Installation

This project uses Python 3.12+. Install dependencies using:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
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
python calculate_metrics_demo.py
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

**Usage Example:**
```python
from utils import extract_edges_from_db
from metrics import calculate_internal_indices_from_dataframe

# Extract edges from database
edges_df = extract_edges_from_db(verbose=True)

# Calculate internal metrics
results = calculate_internal_indices_from_dataframe(
    communities_file='results/leiden_communities.csv',
    edges_df=edges_df,
    verbose=True
)

print(f"Modularity: {results['modularity']:.4f}")
print(f"Coverage: {results['coverage']:.4f}")
```

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

**Usage Example:**
```python
from metrics import calculate_external_indices

results = calculate_external_indices(
    preds_file='results/leiden_communities.csv',
    ground_truth_file='results/ground_truth_communities.csv',
    verbose=True
)

print(f"AMI: {results['ami']:.4f}")
print(f"ARI: {results['ari']:.4f}")
print(f"VI: {results['vi']:.4f}")
```

### Recommended Metrics for Citation Networks

**Internal evaluation (no ground truth):**
- **Primary:** Modularity (directed variant) + Coverage
- **Secondary:** Average Conductance (for boundary quality)

**External evaluation (with ground truth):**
- **Primary:** AMI + ARI + VI
- **Secondary:** NMI (for comparability with literature)

**Why these choices?**
- Citation networks are **directed** and **sparse** - modularity's directed formulation handles asymmetric links
- **AMI** handles variable cluster counts better than NMI (important when comparing algorithms)
- **VI** provides complementary distance-based perspective to similarity scores
- **Coverage** is simple, interpretable, and robust for sparse graphs

See the "Internal indices" and "External indices" sections at the end of this README for detailed rankings and theoretical background.

Launch the interactive web interface:

```bash
streamlit run app.py
```

## Running the Streamlit UI

Launch the interactive web interface:

```bash
streamlit run app.py
```

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

## Data Files

### Database (Required)

- **`data/openalex_works.db`** - SQLite database with paper metadata
  - **Table:** `works`
  - **Key columns:** `id`, `title`, `doi`, `publication_date`, `referenced_works`, `topic`, `authors`, `cited_by_count`, `apc_list_price`

### Community Detection Files (Optional)

- **`results/leiden_communities.csv`** - Community assignments from Leiden algorithm
  - Columns: `paper_id`, `cluster_id`
- **`results/ground_truth_communities.csv`** - Reference partition for external metrics
  - Columns: `paper_id`, `cluster_id`
- **`data/edges.csv`** - Citation edge list (can be generated from database)
  - Columns: `source_id`, `target_id`

### Alternative Community Files

- `data/communities_gn.csv` - Girvan-Newman results
- `data/communities_kl.csv` - Kernighan-Lin results
- `data/communities_louvain.csv` - Louvain results

**Note:** Community CSV files use `paper_id` (or `id`) to match the database schema.

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

## Citation Network Analysis

### Building Citation Networks

Extract edges from the database and build networks:

```python
from utils import extract_edges_from_db

# Extract citation edges
edges_df = extract_edges_from_db(
    db_path="/root/openalex_works.db",
    topics=['T10078', 'T10001', 'T10018', 'T10030', 'T10017'],
    verbose=True
)

# Result: DataFrame with source_id and target_id columns
print(f"Edges: {len(edges_df):,}")
```

### Computing PageRank

```python
import networkx as nx
from metrics import load_graph_from_dataframe

# Load graph
G = load_graph_from_dataframe(edges_df)

# Compute PageRank
pagerank_scores = nx.pagerank(G)

# Sort papers by importance
top_papers = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
```

### GPU-Accelerated Analysis

For large graphs (millions of edges), use cuGraph:

```python
import cugraph

# Convert to cuGraph
cu_G = cugraph.from_pandas_edgelist(edges_df, source='source_id', target='target_id')

# GPU-accelerated PageRank
pagerank_df = cugraph.pagerank(cu_G)
```

See `citation_pagerank.ipynb` for complete examples and visualizations.


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

## API Reference

### Core Functions

#### `utils.py`

```python
extract_edges_from_db(
    db_path: str = "/root/openalex_works.db",
    topics: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame
```
Extract citation edges from OpenAlex database. Returns DataFrame with `source_id` and `target_id` columns.

#### `metrics.py` - Internal Indices

```python
calculate_internal_indices_from_dataframe(
    communities_file: Union[str, Path],
    edges_df: pd.DataFrame,
    max_edges: Optional[int] = None,
    verbose: bool = True
) -> dict
```
Calculate modularity, coverage, performance, and conductance using in-memory edges.

```python
calculate_internal_indices(
    communities_file: Union[str, Path],
    edges_file: Optional[Union[str, Path]] = None,
    max_edges: Optional[int] = None,
    n_edges: Optional[int] = None,
    verbose: bool = True
) -> dict
```
Calculate internal indices from CSV file or with just edge count.

#### `metrics.py` - External Indices

```python
calculate_external_indices(
    preds_file: Union[str, Path],
    ground_truth_file: Union[str, Path],
    verbose: bool = True
) -> dict
```
Calculate AMI, ARI, VI, NMI, homogeneity, completeness, V-measure, FMI, and purity.

### Usage Examples

**Complete workflow:**
```python
from pathlib import Path
from utils import extract_edges_from_db
from metrics import (
    calculate_internal_indices_from_dataframe,
    calculate_external_indices
)

# 1. Extract edges
edges_df = extract_edges_from_db(verbose=True)

# 2. Internal evaluation
internal_results = calculate_internal_indices_from_dataframe(
    'results/leiden_communities.csv',
    edges_df
)

# 3. External evaluation (if ground truth available)
external_results = calculate_external_indices(
    'results/leiden_communities.csv',
    'results/ground_truth_communities.csv'
)

# 4. Print results
print(f"Modularity: {internal_results['modularity']:.4f}")
print(f"AMI: {external_results['ami']:.4f}")
```

## Performance Considerations

### Dataset Scale
- **Papers:** ~181,902 nodes
- **Citations:** ~5,438,287 directed edges
- **Communities:** Typically 50-100 clusters

### Memory Requirements
- Full graph: ~500MB - 1GB RAM
- Sampled (1M edges): ~100MB RAM
- Database: Varies by paper count

### Computation Time (Estimates)
| Operation | Full Graph | Sampled (1M edges) |
|-----------|------------|-------------------|
| Edge extraction | 30-60s | 30-60s |
| Graph loading | 60-120s | 10-20s |
| Internal metrics | 120-300s | 30-60s |
| External metrics | 5-10s | 5-10s |

### Optimization Tips

1. **Use sampling for development:**
   ```python
   results = calculate_internal_indices_from_dataframe(
       communities_file,
       edges_df,
       max_edges=1_000_000  # Faster, approximate results
   )
   ```

2. **Extract edges once, reuse multiple times:**
   ```python
   edges_df = extract_edges_from_db()
   
   # Evaluate multiple algorithms
   for algo in ['leiden', 'louvain', 'label_prop']:
       results = calculate_internal_indices_from_dataframe(
           f'results/{algo}_communities.csv',
           edges_df
       )
   ```

3. **For very large graphs, use graph databases:**
   - Neo4j for persistent graph storage
   - Graph-tool for efficient in-memory operations
   - cuGraph for GPU acceleration

## License

MIT

---

## Appendix: Metric Theory and Rankings

This section provides theoretical background for community detection metrics, including both **implemented metrics** and alternatives for advanced research.

### Internal Indices (No Ground Truth): Ranked by Suitability

**Legend:** ✅ = Implemented in this project

### 1) **Map equation / Infomap description length** (best fit for citation flow)

**Meaning:** Treats the network as a **flow** system (random walk). A partition is good if it gives a **short description length** for movements that tend to stay within communities (flow persists inside modules). ([MapEquation][1])
**Why it’s great for citations:** The map-equation literature explicitly motivates it for **bibliometric/citation networks** and directed flows. ([MapEquation][1])
**Use when:** Your citation edges are directed and you want communities that capture “navigation” / idea flow.

---

### 2) ✅ **(Directed) Modularity (Q)** (strong general-purpose baseline, but watch resolution)

**Meaning:** “How much more internally linked is the partition than expected by chance?” Formally: **fraction of within-community edges minus the expected fraction under a null model**. ([Khoury College of Computer Sciences][2])
**Directed suitability:** There’s a principled **directed modularity** generalization that uses **in/out degree structure** rather than ignoring direction. ([arXiv][3])
**Caveat (important):** Modularity has a known **resolution limit** (can miss smaller communities depending on network size). ([arXiv][4])
**Use when:** You need a widely recognized “overall quality” score; ideally use a directed variant if you keep direction.

---

### 3) ✅ **Conductance / Normalized cut** (best for "clear boundaries" and robustness checks)

**Meaning:** Roughly, “how many edges leave the community relative to how well-connected it is internally.” NetworkX defines conductance as **cut size divided by the smaller volume** of the two sides. ([NetworkX][5])
SNAP’s NCP materials describe it as a widely adopted community-goodness notion, closely tied to normalized cut. ([SNAP][6])
**Why useful for citation graphs:** Citation networks can produce partitions that look good under global objectives but still have “leaky” topic boundaries; conductance is a clean boundary check.

---

### 4) **Surprise / Significance (statistical partition quality)** (good for sparse graphs + small communities)

**Meaning:** Scores how **unlikely** the observed number of intra-community links is under a null model (Surprise is commonly framed via a **cumulative hypergeometric** idea). ([PLOS][7])
**Why useful here:** Citation graphs are sparse and often have many small/medium topical groups; these statistical scores can be informative when density-based intuition is weak.

---

### 5) ✅ **Coverage + Performance** (simple, fast "sanity check" metrics)

**Meaning (NetworkX):**

* **Coverage** = intra-community edges / total edges. ([NetworkX][8])
* **Performance** = (intra-community edges + inter-community non-edges) / all possible edges. ([NetworkX][8])
  **Why not higher:** Easy to compute, but can be **gamed** by many small communities and doesn’t “understand” direction/flow.

---

### 6) **Pure density/triangle-based cohesion metrics** (usually least suitable for raw citation graphs)

Examples: internal edge density, average internal degree, transitivity/triangles, triangle participation. (These exist as internal “fitness scores” in common libraries. ([GitHub][9]))
**Why low for citations:** Directed acyclic-ish citation structure tends to **suppress triangles/transitivity**, so triangle-heavy cohesion indices can be misleading unless you first transform the graph (e.g., co-citation or bibliographic coupling).

---

## External Indices (Requires a Reference Partition): Ranked by Suitability

**Legend:** ✅ = Implemented in this project

In citation work, "ground truth" is often **imperfect** (journal categories, field labels, venue tracks, curated taxonomies). Because cluster counts and sizes vary a lot, **chance-adjusted** measures are usually the safest headline numbers.

### 1) ✅ **AMI (Adjusted Mutual Information)** (best default external score)

**Meaning:** Mutual information between labelings **adjusted for chance**, explicitly recommended over raw NMI when chance inflation is a concern. ([Scikit-learn][10])
**Why #1:** Citation partitions often have many communities; **AMI handles the “more clusters ⇒ higher MI” pitfall** better than NMI. ([Scikit-learn][10])

---

### 2) ✅ **ARI (Adjusted Rand Index)** (strong pair-counting agreement)

**Meaning:** Pair-counting similarity (“same community vs different”) **adjusted for chance**. ([Scikit-learn][11])
**Why #2:** Very interpretable and robust when comparing partitions with different label IDs (permutation-invariant). ([Scikit-learn][11])

---

### 3) ✅ **VI (Variation of Information)** (best "distance between partitions")

**Meaning:** An information-theoretic **distance** between two partitions: “information lost and gained” when moving from one clustering to another. ([ScienceDirect][12])
**Why #3:** It’s a true partition-to-partition distance (still **not** node–node distances), and is often more stable/diagnostic than a single similarity score.

---

### 4) ✅ **NMI (Normalized Mutual Information)** (common, but not chance-adjusted)

**Meaning:** MI normalized to ([0,1]); scikit-learn notes it is **not adjusted for chance** and suggests AMI may be preferred. ([Scikit-learn][13])
**Why #4:** Still widely reported, but can look artificially good when the number of clusters grows.

---

### 5) ✅ **Homogeneity / Completeness / V-measure** (label-focused diagnostics)

**Meaning (scikit-learn):**

* **Homogeneity:** each cluster contains members of a single class.
* **Completeness:** each class’s members mostly fall into a single cluster.
* **V-measure:** trades off the two. ([Scikit-learn][14])
  **Why #5:** Very interpretable for “topic purity vs topic fragmentation” comparisons against a taxonomy, but less standard in network community detection papers than AMI/ARI/NMI/VI.

---

### 6) ✅ **Fowlkes–Mallows (FMI)** (pairwise precision/recall flavor)

**Meaning:** Based on counts of node pairs that are co-clustered in both vs mismatched; scikit-learn describes it via TP/FP/FN over pairs. ([Scikit-learn][15])
**Why #6:** Fine as an additional pairwise perspective, but usually not the primary metric in bibliometric community detection.

---

### 7) ✅ **Purity / (pairwise) F-measure** (use only as supporting numbers)

**Meaning:** Purity is a simple external criterion and F-measure can weight error types; both are discussed as external clustering evaluation criteria. ([Stanford NLP][16])
**Why last:** Purity especially is biased toward producing **many small clusters**.

---

## Summary: What to Report in a Citation Network Paper

### Implemented in This Project

**Internal headline metrics:**
- ✅ **Directed Modularity (Q)** - General-purpose quality score
- ✅ **Coverage** - Simple, interpretable edge classification
- ✅ **Average Conductance** - Boundary quality check

**External headline metrics (if ground truth available):**
- ✅ **AMI** - Chance-adjusted mutual information
- ✅ **ARI** - Chance-adjusted pair-counting similarity
- ✅ **VI** - Information-theoretic partition distance
- ✅ **NMI** - For comparability with older literature

### Additional Recommended Metrics (Not Implemented)

**For advanced bibliometric analysis:**
- **Map Equation / Infomap** - Best for capturing citation flow and idea navigation
- **Surprise / Significance** - Statistical quality for sparse graphs

If you tell us what your "ground truth" is (e.g., **venue categories**, **arXiv subject**, **MAG/Fields of Study**, **journal taxonomy**), we can suggest the best external trio and how to interpret disagreements (interdisciplinary papers tend to break "flat" ground truths).

---

## References

[1]: https://www.mapequation.org/assets/publications/mapequationtutorial.pdf?utm_source=chatgpt.com "Community detection and visualization of networks with the map equation ..."
[2]: https://www.khoury.northeastern.edu/home/vip/teach/DMcourse/6_graph_analysis/notes_slides/Lect10_community_R.pdf?utm_source=chatgpt.com "Socialnetworkanalysis: communitydetection"
[3]: https://arxiv.org/abs/0709.4500?utm_source=chatgpt.com "Community structure in directed networks"
[4]: https://arxiv.org/abs/physics/0607100?utm_source=chatgpt.com "Resolution limit in community detection"
[5]: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cuts.conductance.html?utm_source=chatgpt.com "conductance — NetworkX 3.6.1 documentation"
[6]: https://snap.stanford.edu/ncp/?utm_source=chatgpt.com "NCP: Network Community Profile - Stanford University"
[7]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0024195&utm_source=chatgpt.com "Deciphering Network Community Structure by Surprise - PLOS"
[8]: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.partition_quality.html?utm_source=chatgpt.com "partition_quality — NetworkX 3.6.1 documentation"
[9]: https://github.com/GiulioRossetti/cdlib/blob/master/docs/reference/evaluation.rst?utm_source=chatgpt.com "cdlib/docs/reference/evaluation.rst at master - GitHub"
[10]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html?utm_source=chatgpt.com "adjusted_mutual_info_score — scikit-learn 1.8.0 documentation"
[11]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html?utm_source=chatgpt.com "adjusted_rand_score — scikit-learn 1.8.0 documentation"
[12]: https://www.sciencedirect.com/science/article/pii/S0047259X06002016?utm_source=chatgpt.com "Comparing clusterings—an information based distance"
[13]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html?utm_source=chatgpt.com "normalized_mutual_info_score — scikit-learn 1.8.0 documentation"
[14]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html?utm_source=chatgpt.com "homogeneity_completeness_v_measure - scikit-learn"
[15]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html?utm_source=chatgpt.com "fowlkes_mallows_score — scikit-learn 1.8.0 documentation"
[16]: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html?utm_source=chatgpt.com "Evaluation of clustering - Stanford University"
