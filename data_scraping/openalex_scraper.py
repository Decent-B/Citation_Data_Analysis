import requests
import json
import time
import threading
from queue import Queue
import signal

from database.connection import get_connection

API_URL = "https://api.openalex.org/works"
FIELDS = ["id", "doi", "title", "apc_list_price", "topic", "topic_name",
          "referenced_works_count", "referenced_works",
          "authors", "cited_by_count", "publication_date", "related_works"]

PER_PAGE_MAX = 200
BUFFER_LIMIT = 5000  # number of works to buffer before flush
CURSOR_FILE = "cursor_state.json"
stop_scraping = False


# --- graceful shutdown handling ---
def handle_interrupt(sig, frame): # pyright: ignore[reportUnusedParameter]
    global stop_scraping
    print("\nKeyboardInterrupt detected. Stopping gracefully...")
    stop_scraping = True
    
_ = signal.signal(signal.SIGINT, handle_interrupt)


# --- cursor persistence ---
def load_cursor():
    try:
        with open(CURSOR_FILE, "r") as f:
            data = json.load(f)
            return data.get("cursor", "*")
    except FileNotFoundError:
        return "*"


def save_cursor(cursor:str):
    with open(CURSOR_FILE, "w") as f:
        json.dump({"cursor": cursor}, f)


def insert_batch(conn, batch):
    """
    Insert a batch of papers into PostgreSQL.
    Uses normalized schema: papers, paper_references, related_works tables.
    """
    cur = conn.cursor()
    
    # Prepare data for papers table
    papers_data = []
    references_data = []
    related_works_data = []
    
    for row in batch:
        paper_id = row[0]
        doi = row[1]
        title = row[2]
        apc_list_price = row[3]
        topic = row[4]
        # topic_name = row[5]  # Not in PostgreSQL schema
        referenced_works_count = row[6]
        referenced_works = json.loads(row[7]) if row[7] else []
        authors = row[8]  # Already JSON string
        cited_by_count = row[9]
        publication_date = row[10]
        related_works = json.loads(row[11]) if row[11] else []
        
        # Papers table data
        papers_data.append((
            paper_id,
            doi,
            title,
            apc_list_price,
            topic,
            referenced_works_count,
            cited_by_count,
            publication_date,
            authors  # JSONB in PostgreSQL
        ))
        
        # Paper references data
        for ref_id in referenced_works:
            references_data.append((paper_id, ref_id))
        
        # Related works data
        for related_id in related_works:
            related_works_data.append((paper_id, related_id))
    
    # Insert papers with UPSERT
    cur.executemany("""
        INSERT INTO papers (id, doi, title, apc_list_price, topic, 
                           referenced_works_count, cited_by_count, publication_date, authors)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            doi = EXCLUDED.doi,
            title = EXCLUDED.title,
            apc_list_price = EXCLUDED.apc_list_price,
            topic = EXCLUDED.topic,
            referenced_works_count = EXCLUDED.referenced_works_count,
            cited_by_count = EXCLUDED.cited_by_count,
            publication_date = EXCLUDED.publication_date,
            authors = EXCLUDED.authors
    """, papers_data)
    
    # Insert paper references (ignore conflicts)
    if references_data:
        cur.executemany("""
            INSERT INTO paper_references (citing_paper_id, cited_paper_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        """, references_data)
    
    # Insert related works (ignore conflicts)
    if related_works_data:
        cur.executemany("""
            INSERT INTO related_works (paper_id, related_paper_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        """, related_works_data)
    
    conn.commit()


# --- worker threads ---
def fetcher(q):
    cursor = load_cursor()
    per_page = PER_PAGE_MAX
    while not stop_scraping:
        print(f"Fetching works with cursor: {cursor}")
        params = {"per-page": per_page, "cursor": cursor}
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("Fetch error:", e)
            per_page -= 1
            time.sleep(10)
            continue
        
        works = data.get("results", [])
        next_cursor = data["meta"].get("next_cursor")

        if not works or not next_cursor:
            print("No more results. Done.")
            break

        q.put((works, next_cursor))
        save_cursor(next_cursor)
        cursor = next_cursor
        
        per_page = PER_PAGE_MAX  # reset per_page on success

        time.sleep(0.001)  # gentle delay to avoid hammering API

    q.put(None)  # sentinel to stop processor


def process_authors(authorships:list[dict]) -> list[str]: # pyright: ignore[reportMissingTypeArgument]
    author_ids = []
    for authorship in authorships:
        author = authorship.get("author")
        if author:
            if author.get("id"):
                # Check string, string length and strip prefix
                if isinstance(author["id"], str) and len(author["id"]) > 21:
                    author_ids.append(author["id"][21:])  # strip prefix
    if len(author_ids) == 0:
        print("Warning: No authors found in authorships:", authorships)
    return author_ids


def processor(q, conn):
    buffer = []
    number_processed = 0
    while True:
        print(f"{number_processed * PER_PAGE_MAX} works processed.")
        item = q.get()
        if item is None:
            if buffer:
                insert_batch(conn, buffer)
            print("Processor exiting.")
            break

        works, cursor = item # pyright: ignore[reportUnusedVariable]
        for w in works:
            # Trick to handle nested dictionaries: result = (inner_dict := a.get('key1')) and inner_dict.get('key2') => This will return None if any key is missing, else a['key1']['key2']
            primary_topic = w.get("primary_topic")
            row = [
                w.get("id") and w.get("id")[21:],  # strip prefix
                w.get("doi") and w.get("doi")[16:], # strip prefix
                w.get("title"),
                (apc_list_price := w.get("apc_list")) and apc_list_price.get("value_usd"),
                primary_topic and primary_topic.get("id") and primary_topic.get("id")[21:], # strip prefix
                primary_topic and primary_topic.get("display_name"),  # topic name
                w.get("referenced_works_count"),
                json.dumps(w.get("referenced_works") and [rw[21:] for rw in w.get("referenced_works")]),
                json.dumps(w.get("authorships") and process_authors(w.get("authorships"))),
                w.get("cited_by_count"),
                w.get("publication_date"), # ISO 8601 string
                json.dumps(w.get("related_works") and [rw[21:] for rw in w.get("related_works")]),
            ]
            buffer.append(row)

        if len(buffer) >= BUFFER_LIMIT:
            insert_batch(conn, buffer)
            print(f"Inserted {len(buffer)} works, cursor saved.")
            buffer.clear()
        number_processed += 1

    conn.close()


def main():
    print("Connecting to PostgreSQL database...")
    conn = get_connection()
    print("Connected successfully.")
    
    q = Queue(maxsize=10)

    t_fetch = threading.Thread(target=fetcher, args=(q,))
    t_proc = threading.Thread(target=processor, args=(q, conn))

    t_fetch.start()
    t_proc.start()

    t_fetch.join()
    t_proc.join()

    print("All done!")


if __name__ == "__main__":
    main()
