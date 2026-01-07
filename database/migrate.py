#!/usr/bin/env python3
"""
Migration script: SQLite to PostgreSQL

Migrates paper citation data from SQLite database to PostgreSQL.
Handles the transformation of JSON array fields to normalized tables.

Usage:
    python database/migrate.py [--batch-size BATCH_SIZE] [--filter-communities]
    
Options:
    --batch-size: Number of rows to process per batch (default: 10000)
    --skip-indexes: Skip creating indexes after migration
    --filter-communities: Only migrate papers in community detection results
                         (results/checkpoint_level_10.csv or results/leiden_communities.csv)
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime
from typing import Generator, List, Tuple, Any, Set
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Configuration
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'data/openalex_works.db')
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'citation_db'),
    'user': os.getenv('POSTGRES_USER', 'citation_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'citation_pass'),
}

# Default batch size for inserts
DEFAULT_BATCH_SIZE = 10000


def get_sqlite_connection() -> sqlite3.Connection:
    """Create SQLite connection."""
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"Error: SQLite database not found at {SQLITE_DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(SQLITE_DB_PATH)


def get_postgres_connection():
    """Create PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        conn.autocommit = False
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("Make sure the PostgreSQL container is running: docker compose up -d")
        sys.exit(1)


def parse_json_list(json_str: str) -> List[str]:
    """Parse a JSON array string, returning empty list on failure."""
    if not json_str:
        return []
    try:
        result = json.loads(json_str)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def parse_date(date_str: str) -> str | None:
    """Parse date string to PostgreSQL-compatible format."""
    if not date_str:
        return None
    try:
        # Handle various date formats
        for fmt in ('%Y-%m-%d', '%Y-%m', '%Y'):
            try:
                dt = datetime.strptime(date_str[:len(fmt.replace('%', '').replace('-', '') + '-' * fmt.count('-'))], fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return date_str[:10] if len(date_str) >= 10 else None
    except Exception:
        return None


def load_community_paper_ids() -> Set[str]:
    """
    Load paper IDs from community detection result files.
    
    Looks for:
    - results/checkpoint_level_10.csv (Girvan-Newman)
    - results/leiden_communities.csv (Leiden)
    
    Returns:
        Set of paper IDs that appear in either community file
    """
    import pandas as pd
    
    paper_ids = set()
    community_files = [
        Path('results/checkpoint_level_10.csv'),
        Path('results/leiden_communities.csv')
    ]
    
    for file_path in community_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                # Look for paper_id column
                if 'paper_id' in df.columns:
                    ids = df['paper_id'].astype(str).tolist()
                    paper_ids.update(ids)
                    print(f"  Loaded {len(ids):,} paper IDs from {file_path}")
                else:
                    print(f"  Warning: {file_path} does not have 'paper_id' column")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        else:
            print(f"  File not found: {file_path}")
    
    return paper_ids


def fetch_sqlite_rows(conn: sqlite3.Connection, batch_size: int, filter_ids: Set[str] | None = None) -> Generator[List[Tuple], None, None]:
    """
    Fetch rows from SQLite in batches.
    
    Args:
        conn: SQLite connection
        batch_size: Number of rows to fetch per batch
        filter_ids: Optional set of paper IDs to filter by. If provided, only these papers are fetched.
    """
    cursor = conn.cursor()
    
    if filter_ids:
        # Create a temporary table with the IDs we want
        cursor.execute("CREATE TEMP TABLE filter_ids (id TEXT PRIMARY KEY)")
        cursor.executemany("INSERT INTO filter_ids VALUES (?)", [(id,) for id in filter_ids])
        
        cursor.execute("""
            SELECT w.id, w.doi, w.title, w.apc_list_price, w.topic, 
                   w.referenced_works_count, w.referenced_works, w.authors,
                   w.cited_by_count, w.publication_date, w.related_works
            FROM works w
            INNER JOIN filter_ids f ON w.id = f.id
        """)
    else:
        cursor.execute("""
            SELECT id, doi, title, apc_list_price, topic, 
                   referenced_works_count, referenced_works, authors,
                   cited_by_count, publication_date, related_works
            FROM works
        """)
    
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def migrate_batch(
    pg_conn,
    rows: List[Tuple],
    papers_cursor,
    refs_cursor,
    related_cursor
) -> Tuple[int, int, int]:
    """
    Migrate a batch of rows to PostgreSQL.
    
    Returns:
        Tuple of (papers_count, refs_count, related_count)
    """
    papers_data = []
    refs_data = []
    related_data = []
    
    for row in rows:
        (paper_id, doi, title, apc_price, topic, 
         ref_count, referenced_works, authors_json,
         cited_by, pub_date, related_works) = row
        
        # Parse JSON fields
        authors = parse_json_list(authors_json)
        refs = parse_json_list(referenced_works)
        related = parse_json_list(related_works)
        
        # Prepare papers data
        papers_data.append((
            paper_id,
            doi,
            title,
            apc_price,
            topic,
            ref_count or 0,
            cited_by or 0,
            parse_date(pub_date),
            json.dumps(authors) if authors else None
        ))
        
        # Prepare references data
        for cited_id in refs:
            if cited_id:  # Skip empty strings
                refs_data.append((paper_id, cited_id))
        
        # Prepare related works data
        for related_id in related:
            if related_id:  # Skip empty strings
                related_data.append((paper_id, related_id))
    
    # Bulk insert papers
    execute_values(
        papers_cursor,
        """
        INSERT INTO papers (id, doi, title, apc_list_price, topic, 
                           referenced_works_count, cited_by_count, 
                           publication_date, authors)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
        """,
        papers_data,
        page_size=1000
    )
    
    # Bulk insert references
    if refs_data:
        execute_values(
            refs_cursor,
            """
            INSERT INTO paper_references (citing_paper_id, cited_paper_id)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            refs_data,
            page_size=1000
        )
    
    # Bulk insert related works
    if related_data:
        execute_values(
            related_cursor,
            """
            INSERT INTO related_works (paper_id, related_paper_id)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            related_data,
            page_size=1000
        )
    
    return len(papers_data), len(refs_data), len(related_data)


def create_indexes(pg_conn):
    """Create indexes after migration for better performance."""
    print("\nCreating indexes...")
    cursor = pg_conn.cursor()
    
    indexes = [
        ("idx_papers_topic", "CREATE INDEX IF NOT EXISTS idx_papers_topic ON papers(topic)"),
        ("idx_papers_publication_date", "CREATE INDEX IF NOT EXISTS idx_papers_publication_date ON papers(publication_date)"),
        ("idx_papers_cited_by_count", "CREATE INDEX IF NOT EXISTS idx_papers_cited_by_count ON papers(cited_by_count DESC)"),
        ("idx_paper_references_cited", "CREATE INDEX IF NOT EXISTS idx_paper_references_cited ON paper_references(cited_paper_id)"),
        ("idx_related_works_related", "CREATE INDEX IF NOT EXISTS idx_related_works_related ON related_works(related_paper_id)"),
    ]
    
    for name, sql in indexes:
        print(f"  Creating {name}...")
        cursor.execute(sql)
    
    pg_conn.commit()
    print("Indexes created successfully.")


def get_counts(pg_conn) -> Tuple[int, int, int]:
    """Get current row counts from PostgreSQL tables."""
    cursor = pg_conn.cursor()
    counts = []
    for table in ['papers', 'paper_references', 'related_works']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        counts.append(cursor.fetchone()[0])
    return tuple(counts)


def main():
    parser = argparse.ArgumentParser(description='Migrate SQLite to PostgreSQL')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for inserts (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--skip-indexes', action='store_true',
                        help='Skip creating indexes after migration')
    parser.add_argument('--filter-communities', action='store_true',
                        help='Only migrate papers that appear in community detection results')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SQLite to PostgreSQL Migration")
    print("=" * 60)
    print(f"Source: {SQLITE_DB_PATH}")
    print(f"Target: {POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}")
    print(f"Batch size: {args.batch_size}")
    print(f"Filter by communities: {args.filter_communities}")
    print()
    
    # Load community paper IDs if filtering is enabled
    filter_ids = None
    if args.filter_communities:
        print("Loading paper IDs from community detection results...")
        filter_ids = load_community_paper_ids()
        if not filter_ids:
            print("Warning: No paper IDs found in community files!")
            print("Expected files:")
            print("  - results/checkpoint_level_10.csv")
            print("  - results/leiden_communities.csv")
            response = input("Continue without filtering? (y/n): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                sys.exit(0)
        else:
            print(f"Found {len(filter_ids):,} unique paper IDs to migrate")
            print()
    
    # Connect to databases
    print("Connecting to SQLite...")
    sqlite_conn = get_sqlite_connection()
    
    print("Connecting to PostgreSQL...")
    pg_conn = get_postgres_connection()
    
    # Get total count for progress reporting
    sqlite_cursor = sqlite_conn.cursor()
    if filter_ids:
        # Count only filtered papers
        total_rows = len(filter_ids)
    else:
        sqlite_cursor.execute("SELECT COUNT(*) FROM works")
        total_rows = sqlite_cursor.fetchone()[0]
    print(f"Total rows to migrate: {total_rows:,}")
    print()
    
    # Migration
    print("Starting migration...")
    start_time = datetime.now()
    
    papers_cursor = pg_conn.cursor()
    refs_cursor = pg_conn.cursor()
    related_cursor = pg_conn.cursor()
    
    total_papers = 0
    total_refs = 0
    total_related = 0
    batch_num = 0
    
    try:
        for rows in fetch_sqlite_rows(sqlite_conn, args.batch_size, filter_ids):
            batch_num += 1
            papers, refs, related = migrate_batch(
                pg_conn, rows, papers_cursor, refs_cursor, related_cursor
            )
            
            total_papers += papers
            total_refs += refs
            total_related += related
            
            # Commit every batch
            pg_conn.commit()
            
            # Progress report
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = total_papers / elapsed if elapsed > 0 else 0
            progress = (total_papers / total_rows) * 100 if total_rows > 0 else 0
            
            print(f"\rBatch {batch_num}: {total_papers:,}/{total_rows:,} papers "
                  f"({progress:.1f}%) | {total_refs:,} refs | {total_related:,} related | "
                  f"{rate:.0f} papers/sec", end="", flush=True)
        
        print()  # New line after progress
        
        # Create indexes
        if not args.skip_indexes:
            create_indexes(pg_conn)
        
        # Final stats
        elapsed = (datetime.now() - start_time).total_seconds()
        print()
        print("=" * 60)
        print("Migration Complete!")
        print("=" * 60)
        print(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Papers migrated: {total_papers:,}")
        print(f"References inserted: {total_refs:,}")
        print(f"Related works inserted: {total_related:,}")
        
        if args.filter_communities:
            print(f"\n✓ Filtered to community papers only")
        
        # Verify counts
        pg_papers, pg_refs, pg_related = get_counts(pg_conn)
        print()
        print("PostgreSQL table counts:")
        print(f"  papers: {pg_papers:,}")
        print(f"  paper_references: {pg_refs:,}")
        print(f"  related_works: {pg_related:,}")
        
    except Exception as e:
        pg_conn.rollback()
        print(f"\nError during migration: {e}")
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == '__main__':
    main()
