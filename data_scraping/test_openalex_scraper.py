import random

from database.connection import get_connection


def connect_db():
    """Connect to PostgreSQL database."""
    conn = get_connection()
    return conn


def count_rows(conn):
    """Count total rows in papers table."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS total FROM papers")
    total = cur.fetchone()[0]
    print(f"Total rows in database: {total:,}")
    return total


def show_sample(conn, n=10):
    """Show sample papers from the database."""
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            p.id,
            p.doi,
            p.apc_list_price,
            p.topic,
            p.referenced_works_count,
            (SELECT json_agg(pr.cited_paper_id) 
             FROM paper_references pr 
             WHERE pr.citing_paper_id = p.id) as referenced_works,
            p.authors,
            p.publication_date,
            p.cited_by_count
        FROM papers p
        LIMIT 1000
    """)
    rows = cur.fetchall()
    if not rows:
        print("No data found.")
        return

    sample_rows = random.sample(rows, min(n, len(rows)))
    column_names = ['id', 'doi', 'apc_list_price', 'topic', 'referenced_works_count', 
                    'referenced_works', 'authors', 'publication_date', 'cited_by_count']
    
    for i, row in enumerate(sample_rows, 1):
        print(f"\n--- Sample #{i} ---")
        print(f"id: {row[0]}")                           # OpenAlex work ID (e.g., W1976841490)
        print(f"doi: {row[1]}")                          # DOI of the work (if available)
        print(f"apc_list_price: {row[2]}")               # Article Processing Charge list price (if available in USD)
        print(f"topic: {row[3]}")                        # Topic of the work (if available)
        print(f"referenced_works_count: {row[4]}")       # Number of works referenced by this work
        print(f"referenced_works: {row[5]}")             # List of referenced work IDs
        print(f"authors: {row[6]}")                      # List of author IDs
        print(f"publication_date: {row[7]}")             # Publication date of the work (ISO 8601 format)
        print(f"cited_by_count: {row[8]}")               # Number of times this work has been cited


def main():
    conn = connect_db()
    count_rows(conn)
    show_sample(conn, n=10)
    conn.close()


if __name__ == "__main__":
    main()
