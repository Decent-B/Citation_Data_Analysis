"""
PostgreSQL database connection utilities.

Provides centralized connection management for the Citation Data Analysis project.
"""

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection parameters from environment
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'citation_db'),
    'user': os.getenv('POSTGRES_USER', 'citation_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'citation_pass'),
}

# SQLAlchemy connection URL
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Connection pool for production use
_connection_pool = None
_sqlalchemy_engine = None


def get_engine() -> Engine:
    """
    Get SQLAlchemy engine for pandas compatibility.
    
    Returns:
        SQLAlchemy Engine instance (for use with pandas.read_sql_query)
    """
    global _sqlalchemy_engine
    if _sqlalchemy_engine is None:
        _sqlalchemy_engine = create_engine(DATABASE_URL)
    return _sqlalchemy_engine


def get_connection_pool(minconn: int = 1, maxconn: int = 10) -> pool.ThreadedConnectionPool:
    """
    Get or create a connection pool.
    
    Args:
        minconn: Minimum number of connections in the pool
        maxconn: Maximum number of connections in the pool
    
    Returns:
        ThreadedConnectionPool instance
    """
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn, maxconn, **DB_CONFIG
        )
    return _connection_pool


def get_connection():
    """
    Get a new database connection.
    
    Returns:
        psycopg2 connection object
    """
    return psycopg2.connect(**DB_CONFIG)


@contextmanager
def get_db_connection() -> Generator:
    """
    Context manager for database connections.
    
    Automatically handles connection cleanup.
    
    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers")
    
    Yields:
        Database connection
    """
    conn = None
    try:
        conn = get_connection()
        yield conn
    finally:
        if conn is not None:
            conn.close()


@contextmanager
def get_db_cursor(commit: bool = False) -> Generator:
    """
    Context manager for database cursors.
    
    Automatically handles cursor and connection cleanup.
    Optionally commits the transaction.
    
    Args:
        commit: If True, commit the transaction before closing
    
    Usage:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM papers")
            rows = cursor.fetchall()
    
    Yields:
        Database cursor
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        finally:
            cursor.close()


def test_connection() -> bool:
    """
    Test the database connection.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return result is not None and result[0] == 1
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection
    print("Testing PostgreSQL connection...")
    print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"Database: {DB_CONFIG['database']}")
    print(f"User: {DB_CONFIG['user']}")
    
    if test_connection():
        print("✅ Connection successful!")
    else:
        print("❌ Connection failed!")
