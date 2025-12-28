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
- **Girvan-Newman**: Can run live in the UI or load pre-computed CSV
- Visualize paper communities in an interactive graph
- Hover over nodes to see paper details
- View community statistics and sizes

## Girvan-Newman Community Detection

The Girvan-Newman algorithm is implemented in `scripts/community_detection.py`.
The algorithm is run via command line to generate CSV files, which are then loaded by the Streamlit UI for visualization.

### Via Command Line

```bash
# Run with defaults
python scripts/community_detection.py

# Quick test on 200 papers
python scripts/community_detection.py --limit 200 --test

# Custom database path
python scripts/community_detection.py --db-path /path/to/custom/works.db

# Custom parameters
python scripts/community_detection.py --max-levels 100 --max-nodes 5000

# Filter by topic
python scripts/community_detection.py --topic T10181

# Verbose output
python scripts/community_detection.py --verbose
```

### Via Streamlit UI

After running the algorithm via command line, the results can be visualized in the UI:
1. Go to the "Community Detection" tab
2. Select "Girvan-Newman" algorithm
3. Click "🔍 Load Communities"

### Algorithm Details

- Uses NetworkX's `girvan_newman` implementation
- Selects best partition by maximizing **modularity**
- Builds **undirected** citation graph from `referenced_works`
- Handles disconnected graphs by extracting largest component
- Outputs: `data/communities_gn.csv`, `data/edges.csv`, `data/communities_gn_meta.json`

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

## Community Detection Evaluation

The `metrics.py` module provides tools to evaluate community detection algorithms by comparing predicted clusters against ground truth.

### Available Metrics

1. **AMI (Adjusted Mutual Information)**: Measures agreement between clusterings, adjusted for chance
   - Range: [0, 1] (typically)
   - 1.0 = perfect agreement, 0.0 = random

2. **ARI (Adjusted Rand Index)**: Measures similarity between clusterings, adjusted for chance
   - Range: [-1, 1]
   - 1.0 = identical clustering, 0.0 = random

3. **VI (Variation of Information)**: Measures distance between clusterings based on entropy
   - Range: [0, log(N)]
   - 0.0 = perfect agreement, higher = more different

### Usage

```python
from metrics import evaluate_clustering

# Evaluate a community detection algorithm
results = evaluate_clustering(
    preds_file='data/communities_louvain.csv',
    ground_truth_file='data/communities_ground_truth.csv'
)

print(f"AMI: {results['ami']:.4f}")
print(f"ARI: {results['ari']:.4f}")
print(f"VI:  {results['vi']:.4f}")
```

**Individual metric functions:**

```python
from metrics import calculate_ami, calculate_ari, calculate_vi

# Calculate metrics from numpy arrays
ami = calculate_ami(predictions, ground_truth)
ari = calculate_ari(predictions, ground_truth)
vi = calculate_vi(predictions, ground_truth)
```

**CSV Format:**
Community CSV files should have columns: `paper_id` (or `id`) and `cluster_id` (or `community`):
```csv
paper_id,cluster_id
W1234567,0
W2345678,0
W3456789,1
```

See `example_metrics.py` for detailed examples including comparing multiple algorithms.

## Project Structure

```
.
├── app.py                      # Streamlit UI entry point
├── main.py                     # Main application entry
├── scripts/                    # Utility scripts
│   ├── community_detection.py # Girvan-Newman community detection (CLI)
│   └── metrics.py             # Community detection evaluation metrics
├── notebooks/                  # Jupyter notebooks for analysis
│   └── ranking.ipynb          # Ranking algorithms exploration
├── ui/                         # Streamlit UI components
│   ├── search_tab.py          # Search interface
│   ├── community_tab.py       # Community detection visualization
│   ├── search.py              # Search algorithms
│   └── data_access.py         # Data loading utilities (SQLite, CSV)
├── data_scraping/             # OpenAlex data collection
│   ├── openalex_scraper.py   # Main scraper
│   └── utils.py              # API helper functions
├── ranking/                   # Ranking algorithms module
│   ├── pagerank.py           # PageRank implementations
│   ├── search.py             # Search functions
│   └── utils.py              # Utility functions
├── data/                      # Data files (gitignored)
│   └── openalex_works.db     # SQLite database
└── pyproject.toml            # Dependencies
```

## License

MIT


## Internal indices (no ground truth): ranked by suitability

### 1) **Map equation / Infomap description length (best fit for citation flow)**

**Meaning:** Treats the network as a **flow** system (random walk). A partition is good if it gives a **short description length** for movements that tend to stay within communities (flow persists inside modules). ([MapEquation][1])
**Why it’s great for citations:** The map-equation literature explicitly motivates it for **bibliometric/citation networks** and directed flows. ([MapEquation][1])
**Use when:** Your citation edges are directed and you want communities that capture “navigation” / idea flow.

---

### 2) **(Directed) Modularity (Q)** (strong general-purpose baseline, but watch resolution)

**Meaning:** “How much more internally linked is the partition than expected by chance?” Formally: **fraction of within-community edges minus the expected fraction under a null model**. ([Khoury College of Computer Sciences][2])
**Directed suitability:** There’s a principled **directed modularity** generalization that uses **in/out degree structure** rather than ignoring direction. ([arXiv][3])
**Caveat (important):** Modularity has a known **resolution limit** (can miss smaller communities depending on network size). ([arXiv][4])
**Use when:** You need a widely recognized “overall quality” score; ideally use a directed variant if you keep direction.

---

### 3) **Conductance / Normalized cut** (best for “clear boundaries” and robustness checks)

**Meaning:** Roughly, “how many edges leave the community relative to how well-connected it is internally.” NetworkX defines conductance as **cut size divided by the smaller volume** of the two sides. ([NetworkX][5])
SNAP’s NCP materials describe it as a widely adopted community-goodness notion, closely tied to normalized cut. ([SNAP][6])
**Why useful for citation graphs:** Citation networks can produce partitions that look good under global objectives but still have “leaky” topic boundaries; conductance is a clean boundary check.

---

### 4) **Surprise / Significance (statistical partition quality)** (good for sparse graphs + small communities)

**Meaning:** Scores how **unlikely** the observed number of intra-community links is under a null model (Surprise is commonly framed via a **cumulative hypergeometric** idea). ([PLOS][7])
**Why useful here:** Citation graphs are sparse and often have many small/medium topical groups; these statistical scores can be informative when density-based intuition is weak.

---

### 5) **Coverage + Performance** (simple, fast “sanity check” metrics)

**Meaning (NetworkX):**

* **Coverage** = intra-community edges / total edges. ([NetworkX][8])
* **Performance** = (intra-community edges + inter-community non-edges) / all possible edges. ([NetworkX][8])
  **Why not higher:** Easy to compute, but can be **gamed** by many small communities and doesn’t “understand” direction/flow.

---

### 6) **Pure density/triangle-based cohesion metrics** (usually least suitable for raw citation graphs)

Examples: internal edge density, average internal degree, transitivity/triangles, triangle participation. (These exist as internal “fitness scores” in common libraries. ([GitHub][9]))
**Why low for citations:** Directed acyclic-ish citation structure tends to **suppress triangles/transitivity**, so triangle-heavy cohesion indices can be misleading unless you first transform the graph (e.g., co-citation or bibliographic coupling).

---

## External indices (requires a reference partition): ranked by suitability

In citation work, “ground truth” is often **imperfect** (journal categories, field labels, venue tracks, curated taxonomies). Because cluster counts and sizes vary a lot, **chance-adjusted** measures are usually the safest headline numbers.

### 1) **AMI (Adjusted Mutual Information)** (best default external score)

**Meaning:** Mutual information between labelings **adjusted for chance**, explicitly recommended over raw NMI when chance inflation is a concern. ([Scikit-learn][10])
**Why #1:** Citation partitions often have many communities; **AMI handles the “more clusters ⇒ higher MI” pitfall** better than NMI. ([Scikit-learn][10])

---

### 2) **ARI (Adjusted Rand Index)** (strong pair-counting agreement)

**Meaning:** Pair-counting similarity (“same community vs different”) **adjusted for chance**. ([Scikit-learn][11])
**Why #2:** Very interpretable and robust when comparing partitions with different label IDs (permutation-invariant). ([Scikit-learn][11])

---

### 3) **VI (Variation of Information)** (best “distance between partitions”)

**Meaning:** An information-theoretic **distance** between two partitions: “information lost and gained” when moving from one clustering to another. ([ScienceDirect][12])
**Why #3:** It’s a true partition-to-partition distance (still **not** node–node distances), and is often more stable/diagnostic than a single similarity score.

---

### 4) **NMI (Normalized Mutual Information)** (common, but not chance-adjusted)

**Meaning:** MI normalized to ([0,1]); scikit-learn notes it is **not adjusted for chance** and suggests AMI may be preferred. ([Scikit-learn][13])
**Why #4:** Still widely reported, but can look artificially good when the number of clusters grows.

---

### 5) **Homogeneity / Completeness / V-measure** (label-focused diagnostics)

**Meaning (scikit-learn):**

* **Homogeneity:** each cluster contains members of a single class.
* **Completeness:** each class’s members mostly fall into a single cluster.
* **V-measure:** trades off the two. ([Scikit-learn][14])
  **Why #5:** Very interpretable for “topic purity vs topic fragmentation” comparisons against a taxonomy, but less standard in network community detection papers than AMI/ARI/NMI/VI.

---

### 6) **Fowlkes–Mallows (FMI)** (pairwise precision/recall flavor)

**Meaning:** Based on counts of node pairs that are co-clustered in both vs mismatched; scikit-learn describes it via TP/FP/FN over pairs. ([Scikit-learn][15])
**Why #6:** Fine as an additional pairwise perspective, but usually not the primary metric in bibliometric community detection.

---

### 7) **Purity / (pairwise) F-measure** (use only as supporting numbers)

**Meaning:** Purity is a simple external criterion and F-measure can weight error types; both are discussed as external clustering evaluation criteria. ([Stanford NLP][16])
**Why last:** Purity especially is biased toward producing **many small clusters**.

---

## What I’d report in a citation-network paper

* **Internal headline:** **Map equation description length** (or its score) + **directed modularity** + **conductance/normalized cut** as a boundary sanity check. ([MapEquation][1])
* **External headline (if you have labels):** **AMI + ARI + VI** (and optionally NMI for comparability with older literature). ([Scikit-learn][10])

If you tell me what your “ground truth” is (e.g., **venue categories**, **arXiv subject**, **MAG/Fields of Study**, **journal taxonomy**), I’ll suggest the best external trio and how to interpret disagreements (interdisciplinary papers tend to break “flat” ground truths).

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
