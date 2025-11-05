import sqlite3
import random

DB_FILE = "dataset/openalex_works-13m.db"

def connect_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # makes rows behave like dicts
    return conn

def count_rows(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS total FROM works")
    total = cur.fetchone()["total"]
    print(f"Total rows in database: {total:,}")
    return total

def show_sample(conn, n=10):
    cur = conn.cursor()
    cur.execute("SELECT * FROM works LIMIT 1000")
    rows = cur.fetchall()
    if not rows:
        print("No data found.")
        return

    sample_rows = random.sample(rows, min(n, len(rows)))
    for i, row in enumerate(sample_rows, 1):
        print(f"\n--- Sample #{i} ---")
        print(f"id: {row['id']}")                                           # OpenAlex work ID (e.g., W1976841490)
        print(f"doi: {row['doi']}")                                         # DOI of the work (if available)
        print(f"apc_list_price: {row['apc_list_price']}")                   # Article Processing Charge list price (if available in USD)
        print(f"topic: {row['topic']}")                                     # Topic of the work (if available)
        print(f"referenced_works_count: {row['referenced_works_count']}")   # Number of works referenced by this work
        print(f"referenced_works: {row['referenced_works']}")               # List of referenced work IDs
        print(f"authors: {row['authors']}")                                 # List of author IDs
        print(f"publication_date: {row['publication_date']}")               # Publication date of the work (ISO 8601 format)
        print(f"cited_by_count: {row['cited_by_count']}")                   # Number of times this work has been cited


def main():
    conn = connect_db()
    count_rows(conn)
    show_sample(conn, n=10)
    conn.close()

if __name__ == "__main__":
    main()
