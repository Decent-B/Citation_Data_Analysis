#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-file Girvan–Newman runner for OpenAlex citation data (SQLite).

- Loads up to PAPERS_PER_TOPIC papers per topic (top by cited_by_count) using a window function.
- Builds a citation graph (undirected by default) from referenced_works.
- Optional: keep only largest connected component.
- Optional: sample to MAX_NODES to prevent memory/time blowups.
- Runs Girvan–Newman up to MAX_LEVELS and selects the best modularity partition.
- Writes:
    - gn_communities.csv  (paper_id, cluster_id, topic)
    - edges.csv           (source_id, target_id)
    - gn_meta.json        (run metadata)
"""

import argparse
import json
import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import networkx as nx
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity


# ============================================================
# DEFAULT CONFIG (edit here if you want hard-coded defaults)
# ============================================================

DB_PATH_DEFAULT = "/data/openalex_works.db"
OUTPUT_DIR_DEFAULT = "results"
PAPERS_PER_TOPIC_DEFAULT = 200

# Balanced settings (your chosen configuration)
MAX_NODES_DEFAULT = 2500
MAX_LEVELS_DEFAULT = 25
USE_LARGEST_COMPONENT_DEFAULT = True
UNDIRECTED_DEFAULT = True
FILTER_REFERENCES_TO_LOADED_PAPERS_DEFAULT = True
RANDOM_SEED_DEFAULT = 42

# Your topics list (paste/edit as needed)
TARGET_TOPICS_DEFAULT: List[str] = [
    "T10036", "T10320", "T10775", "T11273", "T11714", "T11307", "T11652", "T11512",
    "T10203", "T10538", "T12016", "T10028", "T12026", "T11396", "T11636", "T14414",
    "T10906", "T10100", "T10963", "T10848", "T10050", "T11975", "T12101", "T11612",
    "T10181", "T11710", "T11550", "T10664", "T12031", "T10201", "T10860", "T13083",
    "T13629", "T12380", "T12262", "T10531", "T11105", "T12923", "T14339", "T10627",
    "T10824", "T10601", "T10057", "T11448", "T11094", "T10331", "T10052", "T11659",
    "T10688", "T11019", "T11165", "T12549", "T10719", "T10812", "T10260", "T10430",
    "T12490", "T10743", "T11450", "T11675", "T12127", "T12423", "T10317", "T10101",
    "T14067", "T11181", "T11106", "T11719", "T11986", "T13650", "T13398", "T10799",
    "T11891", "T14280", "T11937", "T14201", "T11478", "T10715", "T10772", "T10054",
    "T10273", "T12222", "T10575", "T10125", "T11458", "T11409", "T10080", "T10711",
    "T10651", "T10714", "T10742", "T10796", "T11896", "T11932", "T12791", "T12024",
    "T10138", "T12216", "T12326", "T10237", "T10951", "T10400", "T10734", "T11045",
    "T10764", "T11504", "T11800", "T11644", "T11241", "T12122", "T11598", "T13999",
    "T10927", "T10270", "T13913", "T10462", "T10586", "T10653", "T10879", "T10191",
    "T10326", "T10709", "T10868", "T10888", "T10648", "T10789", "T11024", "T11938",
    "T13985", "T11572", "T11499", "T12885", "T10953", "T10350", "T10912", "T11446",
    "T14380", "T10712", "T13673", "T13166", "T12289", "T12863", "T10102", "T13976",
]


# ============================================================
# Core helpers
# ============================================================

def parse_referenced_works(val: Any) -> List[str]:
    """
    Parse 'referenced_works' field into a list of paper IDs.
    Supports:
      - JSON string of list
      - Python-like list string
      - Already-a-list
      - None/NaN
    """
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x is not None]

    s = str(val).strip()
    if not s or s.lower() in {"none", "nan"}:
        return []

    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj if x is not None]
        return []
    except Exception:
        pass

    # Fallback: try to interpret common formats like "['a', 'b']"
    # without using eval on arbitrary code; do a minimal cleanup.
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        # split by comma, strip quotes/spaces
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
        return [p for p in parts if p]

    return []


def load_papers_by_topics_snippet_style(
    conn: sqlite3.Connection,
    topics: Sequence[str],
    papers_per_topic: int,
) -> pd.DataFrame:
    """
    Load up to N papers per topic (ranked by cited_by_count DESC) using a single query
    with ROW_NUMBER() window function, matching the approach in the user's snippet.
    """
    if not topics:
        raise ValueError("topics list is empty")

    placeholders = ", ".join(["?" for _ in topics])

    df = pd.read_sql_query(
        f"""
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY topic ORDER BY cited_by_count DESC) as row_num
            FROM works
            WHERE topic IN ({placeholders})
        )
        WHERE row_num <= ?
        """,
        conn,
        params=(*topics, papers_per_topic),
    )

    if "row_num" in df.columns:
        df = df.drop("row_num", axis=1)

    return df


def build_citation_graph_from_df(
    df: pd.DataFrame,
    undirected: bool = True,
    filter_references_to_loaded_papers: bool = True,
) -> nx.Graph:
    """
    Build a citation graph from df with columns:
      - id (paper id)
      - referenced_works_parsed (list[str])
    """
    required_cols = {"id", "referenced_works_parsed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for graph build: {sorted(missing)}")

    if undirected:
        G: nx.Graph = nx.Graph()
    else:
        G = nx.DiGraph()

    df_ids = df["id"].astype(str).tolist()
    paper_ids = set(df_ids)

    G.add_nodes_from(df_ids)

    edges: List[Tuple[str, str]] = []
    for src, refs in zip(df_ids, df["referenced_works_parsed"].tolist()):
        if not refs:
            continue
        for dst in refs:
            d = str(dst)
            if filter_references_to_loaded_papers and d not in paper_ids:
                continue
            if d == src:
                continue
            edges.append((src, d))

    G.add_edges_from(edges)
    return G


def largest_connected_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    if isinstance(G, nx.DiGraph):
        # for directed graphs, use weakly connected components
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    if not comps:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()


def sample_graph_nodes(G: nx.Graph, max_nodes: int, seed: int) -> nx.Graph:
    if G.number_of_nodes() <= max_nodes:
        return G
    random.seed(seed)
    sampled_nodes = random.sample(list(G.nodes()), max_nodes)
    return G.subgraph(sampled_nodes).copy()


@dataclass
class GNResult:
    best_level: int
    best_modularity: float
    num_communities: int
    num_nodes: int
    num_edges: int
    elapsed_seconds: float


def girvan_newman_best_modularity(
    G: nx.Graph,
    max_levels: int,
) -> Tuple[List[Tuple[str, ...]], GNResult]:
    """
    Runs Girvan–Newman, evaluates modularity at each level up to max_levels,
    and returns the best partition (highest modularity).
    """
    import time

    t0 = time.time()

    # Edge cases
    if G.number_of_nodes() == 0:
        part: List[Tuple[str, ...]] = []
        meta = GNResult(0, float("nan"), 0, 0, 0, time.time() - t0)
        return part, meta

    if G.number_of_edges() == 0:
        # each node as its own community
        part = [(str(n),) for n in G.nodes()]
        meta = GNResult(
            best_level=0,
            best_modularity=0.0,
            num_communities=len(part),
            num_nodes=G.number_of_nodes(),
            num_edges=0,
            elapsed_seconds=time.time() - t0,
        )
        return part, meta

    comp_gen = girvan_newman(G)

    best_partition: Optional[List[Tuple[str, ...]]] = None
    best_Q = float("-inf")
    best_level = 0

    # Level indexing: first yielded partition is level=1 (2 communities typically)
    for level in range(1, max_levels + 1):
        try:
            communities = next(comp_gen)
        except StopIteration:
            break

        part_list = [tuple(map(str, c)) for c in communities]
        # networkx modularity expects list of sets
        part_sets = [set(c) for c in part_list]
        try:
            Q = modularity(G, part_sets)
        except Exception:
            # if modularity computation fails for some reason, skip
            continue

        if Q > best_Q:
            best_Q = Q
            best_partition = part_list
            best_level = level

    if best_partition is None:
        # fallback: single community
        best_partition = [tuple(map(str, G.nodes()))]
        best_Q = 0.0
        best_level = 0

    meta = GNResult(
        best_level=best_level,
        best_modularity=float(best_Q),
        num_communities=len(best_partition),
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        elapsed_seconds=time.time() - t0,
    )
    return best_partition, meta


def partition_to_df(partition: List[Tuple[str, ...]]) -> pd.DataFrame:
    rows = []
    for cid, comm in enumerate(partition):
        for pid in comm:
            rows.append({"paper_id": str(pid), "cluster_id": int(cid)})
    return pd.DataFrame(rows)


def edges_to_df(G: nx.Graph) -> pd.DataFrame:
    return pd.DataFrame(
        [{"source_id": str(u), "target_id": str(v)} for u, v in G.edges()],
        columns=["source_id", "target_id"],
    )


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Girvan–Newman on OpenAlex citation data (single-file).")
    p.add_argument("--db-path", default=DB_PATH_DEFAULT, help="Path to SQLite DB (default: %(default)s)")
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Output directory (default: %(default)s)")
    p.add_argument("--papers-per-topic", type=int, default=PAPERS_PER_TOPIC_DEFAULT, help="Top N papers per topic (default: %(default)s)")
    p.add_argument("--max-nodes", type=int, default=MAX_NODES_DEFAULT, help="Sample graph down to this many nodes if larger (default: %(default)s)")
    p.add_argument("--max-levels", type=int, default=MAX_LEVELS_DEFAULT, help="Max GN levels to evaluate (default: %(default)s)")
    p.add_argument("--largest-component", action="store_true", default=USE_LARGEST_COMPONENT_DEFAULT, help="Keep only largest connected component (default: enabled)")
    p.add_argument("--no-largest-component", action="store_false", dest="largest_component", help="Disable largest connected component filter")
    p.add_argument("--directed", action="store_true", default=not UNDIRECTED_DEFAULT, help="Use directed graph (default: undirected)")
    p.add_argument("--no-filter-refs", action="store_true", default=not FILTER_REFERENCES_TO_LOADED_PAPERS_DEFAULT,
                   help="Do NOT filter references to only loaded papers (default: filter enabled)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT, help="Random seed for sampling (default: %(default)s)")
    p.add_argument("--topics-file", default="", help="Optional: path to a text file of topics (one per line). If set, overrides default topics list.")
    p.add_argument("--topics", default="", help="Optional: comma-separated topics. If set, overrides default topics list.")
    return p.parse_args()


def load_topics(args: argparse.Namespace) -> List[str]:
    if args.topics_file:
        path = Path(args.topics_file)
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln]
    if args.topics:
        return [t.strip() for t in args.topics.split(",") if t.strip()]
    return list(TARGET_TOPICS_DEFAULT)


def main() -> None:
    args = parse_args()
    topics = load_topics(args)
    if not topics:
        raise SystemExit("No topics provided.")

    undirected = not args.directed
    filter_refs = not args.no_filter_refs

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"DB: {args.db_path}")
    print(f"Output: {out_dir.resolve()}")
    print(f"Topics: {len(topics)}")
    print(f"Papers per topic: {args.papers_per_topic}")
    print(f"Undirected: {undirected}")
    print(f"Largest component: {args.largest_component}")
    print(f"Max nodes: {args.max_nodes}")
    print(f"Max levels: {args.max_levels}")
    print(f"Filter refs to loaded papers: {filter_refs}")
    print("-" * 60)

    # Connect DB
    conn = sqlite3.connect(args.db_path)

    # Load data (snippet-style window function)
    print(f"Loading {args.papers_per_topic} papers for each of {len(topics)} topics...")
    df = load_papers_by_topics_snippet_style(conn, topics, args.papers_per_topic)
    conn.close()

    if df.empty:
        raise SystemExit("Loaded DataFrame is empty. Check topics/db/table schema.")

    # Basic sanity info
    if "topic" not in df.columns:
        raise SystemExit("Expected 'topic' column not found in works table.")
    if "id" not in df.columns:
        raise SystemExit("Expected 'id' column not found in works table.")
    if "referenced_works" not in df.columns:
        raise SystemExit("Expected 'referenced_works' column not found in works table.")

    print(f"Total papers loaded: {len(df)}")
    # per-topic counts
    counts = df["topic"].value_counts().to_dict()
    print("Papers per topic (loaded):")
    for t in topics:
        print(f"  {t}: {counts.get(t, 0)}")

    # Parse references
    df["id"] = df["id"].astype(str)
    df["referenced_works_parsed"] = df["referenced_works"].apply(parse_referenced_works)

    # Build graph
    G = build_citation_graph_from_df(
        df[["id", "referenced_works_parsed"]],
        undirected=undirected,
        filter_references_to_loaded_papers=filter_refs,
    )

    print("-" * 60)
    print(f"Graph built: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # Keep largest component
    if args.largest_component:
        G = largest_connected_component(G)
        print(f"After largest component: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # Sample if too big
    if args.max_nodes and G.number_of_nodes() > args.max_nodes:
        G = sample_graph_nodes(G, args.max_nodes, args.seed)
        print(f"After sampling to max_nodes: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # Run GN
    print("-" * 60)
    print("Running Girvan–Newman...")
    partition, meta = girvan_newman_best_modularity(G, max_levels=args.max_levels)
    print(f"Done. Best level={meta.best_level} best modularity={meta.best_modularity:.6f} "
          f"communities={meta.num_communities} elapsed={meta.elapsed_seconds:.2f}s")

    # Communities DF + add topic label
    comm_df = partition_to_df(partition)
    id_to_topic = dict(zip(df["id"], df["topic"]))
    comm_df["topic"] = comm_df["paper_id"].map(id_to_topic)

    # Edges DF (for the final G used in GN)
    edges_df = edges_to_df(G)

    # Write outputs
    comm_path = out_dir / "gn_communities.csv"
    edges_path = out_dir / "edges.csv"
    meta_path = out_dir / "gn_meta.json"

    comm_df.to_csv(comm_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    meta_payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": args.db_path,
        "topics_count": len(topics),
        "papers_per_topic": args.papers_per_topic,
        "undirected": undirected,
        "largest_component": args.largest_component,
        "max_nodes": args.max_nodes,
        "max_levels": args.max_levels,
        "filter_refs_to_loaded_papers": filter_refs,
        "seed": args.seed,
        "graph_nodes_final": meta.num_nodes,
        "graph_edges_final": meta.num_edges,
        "gn_result": asdict(meta),
        "output_files": {
            "communities_csv": str(comm_path),
            "edges_csv": str(edges_path),
            "meta_json": str(meta_path),
        },
    }
    write_json(meta_path, meta_payload)

    print("-" * 60)
    print(f"Wrote: {comm_path}")
    print(f"Wrote: {edges_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
