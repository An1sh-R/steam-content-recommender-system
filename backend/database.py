import os
from contextlib import contextmanager
from typing import Generator, Optional
from typing import cast

import psycopg2
from psycopg2.extras import RealDictCursor


# Read connection settings from environment variables.
# Defaults are Docker-friendly and can be overridden without code changes.
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "games")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def get_connection():
    """Create and return a PostgreSQL connection."""
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return psycopg2.connect(database_url, cursor_factory=RealDictCursor)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
        sslmode="require"  # supabase requires SSL connections
    )


@contextmanager
def get_db_cursor(commit: bool = False) -> Generator[RealDictCursor, None, None]:
    """
    Context manager for safe database cursor usage.
    - Opens connection + cursor
    - Commits when requested
    - Rolls back on failures
    - Always closes resources
    """
    conn = get_connection()
    cur = cast(RealDictCursor, conn.cursor())
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# Centralized SQL keeps query logic clean and maintainable.
CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);
"""

CREATE_USER_PROFILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exploration REAL NOT NULL DEFAULT 0,
    story REAL NOT NULL DEFAULT 0,
    challenge REAL NOT NULL DEFAULT 0,
    strategy REAL NOT NULL DEFAULT 0,
    social REAL NOT NULL DEFAULT 0,
    relaxation REAL NOT NULL DEFAULT 0
);
"""


def initialize_database() -> None:
    """Create required auth tables if they do not already exist."""
    with get_db_cursor(commit=True) as cur:
        cur.execute(CREATE_USERS_TABLE_SQL)
        cur.execute(CREATE_USER_PROFILES_TABLE_SQL)
