#!/usr/bin/env python3
"""
Verification script for SQLite to PostgreSQL migration.

Compares row counts and samples data between the two databases.
"""

import os
import sys
import json
import sqlite3

import psycopg2
from dotenv import load_dotenv


load_dotenv()

SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'data/openalex_works-ver2.db')
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'citation_db'),
    'user': os.getenv('POSTGRES_USER', 'citation_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'citation_pass'),
}


def main():
    print("=" * 60)
    print("Migration Verification")
    print("=" * 60)
    
    # Connect to SQLite
    print("\nConnecting to SQLite...")
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    sqlite_cursor = sqlite_conn.cursor()
    
    # Connect to PostgreSQL
    print("Connecting to PostgreSQL...")
    try:
        pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
        pg_cursor = pg_conn.cursor()
    except psycopg2.OperationalError as e:
        print(f"Error: Cannot connect to PostgreSQL: {e}")
        sys.exit(1)
    
    # Compare paper counts
    print("\n--- Row Count Comparison ---")
    
    sqlite_cursor.execute("SELECT COUNT(*) FROM works")
    sqlite_count = sqlite_cursor.fetchone()[0]
    
    pg_cursor.execute("SELECT COUNT(*) FROM papers")
    pg_papers = pg_cursor.fetchone()[0]
    
    pg_cursor.execute("SELECT COUNT(*) FROM paper_references")
    pg_refs = pg_cursor.fetchone()[0]
    
    pg_cursor.execute("SELECT COUNT(*) FROM related_works")
    pg_related = pg_cursor.fetchone()[0]
    
    print(f"SQLite works:        {sqlite_count:>12,}")
    print(f"PostgreSQL papers:   {pg_papers:>12,}")
    print(f"PostgreSQL refs:     {pg_refs:>12,}")
    print(f"PostgreSQL related:  {pg_related:>12,}")
    
    match = sqlite_count == pg_papers
    print(f"\nPaper count match: {'✓ PASS' if match else '✗ FAIL'}")
    
    # Sample verification
    print("\n--- Sample Data Verification ---")
    
    # Get a sample paper from SQLite
    sqlite_cursor.execute("""
        SELECT id, title, referenced_works, related_works 
        FROM works 
        WHERE referenced_works IS NOT NULL 
          AND referenced_works != ''
          AND related_works IS NOT NULL
          AND related_works != ''
        LIMIT 1
    """)
    sample = sqlite_cursor.fetchone()
    
    if sample:
        paper_id, title, sqlite_refs, sqlite_related = sample
        sqlite_ref_list = json.loads(sqlite_refs) if sqlite_refs else []
        sqlite_related_list = json.loads(sqlite_related) if sqlite_related else []
        
        print(f"Sample paper: {paper_id}")
        print(f"Title: {title[:60]}...")
        
        # Check PostgreSQL
        pg_cursor.execute("SELECT title FROM papers WHERE id = %s", (paper_id,))
        pg_title = pg_cursor.fetchone()
        
        pg_cursor.execute("SELECT COUNT(*) FROM paper_references WHERE citing_paper_id = %s", (paper_id,))
        pg_ref_count = pg_cursor.fetchone()[0]
        
        pg_cursor.execute("SELECT COUNT(*) FROM related_works WHERE paper_id = %s", (paper_id,))
        pg_related_count = pg_cursor.fetchone()[0]
        
        print(f"\nSQLite references: {len(sqlite_ref_list)}")
        print(f"PostgreSQL references: {pg_ref_count}")
        refs_match = len(sqlite_ref_list) == pg_ref_count
        print(f"References match: {'✓ PASS' if refs_match else '✗ FAIL'}")
        
        print(f"\nSQLite related works: {len(sqlite_related_list)}")
        print(f"PostgreSQL related works: {pg_related_count}")
        related_match = len(sqlite_related_list) == pg_related_count
        print(f"Related works match: {'✓ PASS' if related_match else '✗ FAIL'}")
    
    # Overall result
    print("\n" + "=" * 60)
    all_pass = match and (not sample or (refs_match and related_match))
    print(f"Overall verification: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 60)
    
    sqlite_conn.close()
    pg_conn.close()
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
