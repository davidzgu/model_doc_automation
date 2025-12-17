import sqlite3
import json
import datetime
from typing import Optional, List, Dict, Any

# Lightweight SQLite wrapper for chat history, code history and test data locations.
class SQLiteDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self._ensure_pragmas()
        # self.initialize_schema()

    def _ensure_pragmas(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.close()

    def initialize_schema(self):
        cur = self.conn.cursor()
        # Sessions and messages (one conversation/session -> many messages)
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT UNIQUE,
            description TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT CHECK(role IN ('system', 'human', 'ai', 'tool')),
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """)
        # Code history (optionally linked to a session)
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS code_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            file_path TEXT NOT NULL,
            version INTEGER NOT NULL,
            submitted INTEGER NOT NULL CHECK (submitted IN (0, 1)),
            content TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(session_id, version),
            UNIQUE(file_path)
        );
        """)

        # Explicit table for test data locations (optionally linked to a session)
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS test_data_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            description TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(session_id),
            UNIQUE(name),
            UNIQUE(path)
        );
        """)
        cur.close()


    # --- Generic helpers ---
    def _row_to_dict(self, row: sqlite3.Row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # --- Session/Conversation APIs ---
    def create_session(self, session_key: Optional[str] = None, description: Optional[str] = None) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sessions(session_key,  description) VALUES (?, ?)",
            (session_key, description)
        )
        sid = cur.lastrowid
        cur.close()
        return sid


    def get_session(self, session_id: Optional[int] = None, session_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        if (session_id is not None) and (session_key is None):
            cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        elif (session_id is None) and (session_key is not None):
            cur.execute("SELECT * FROM sessions WHERE session_key = ?", (session_key,))
        row = cur.fetchone()
        cur.close()
        return self._row_to_dict(row)

    def list_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        sql = "SELECT * FROM sessions ORDER BY created_at DESC" + (" LIMIT ?" if limit else "")
        cur.execute(sql, (limit,) if limit else ())
        rows = [self._row_to_dict(r) for r in cur.fetchall()]
        cur.close()
        return rows


    def add_message(self, session_id: int, role: str, content: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        msg_id = cur.lastrowid
        cur.close()
        return msg_id


    def get_session_messages(self, session_id: int) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
        rows = []
        for r in cur.fetchall():
            d = self._row_to_dict(r)
            rows.append(d)
        cur.close()
        return rows



    # --- Code history APIs ---
    def add_code_entry(self, file_path: str, version: int, content: str, submitted: Optional[bool] = False,
                        metadata: Optional[Dict[str, Any]] = None, session_id: Optional[int] = None) -> int:
        submitted_to_db = 1 if submitted else 0
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO code_history(session_id, file_path, version, submitted, content)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, file_path, version, submitted_to_db, content))
        entry_id = cur.lastrowid
        cur.close()
        return entry_id

    def get_code_recent_history(self, session_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        sql = "SELECT * FROM code_history WHERE session_id = ? ORDER BY version DESC" + (" LIMIT ?" if limit else "")
        cur.execute(sql, (session_id, limit) if limit else (session_id,))
        rows = []
        for r in cur.fetchall():
            d = self._row_to_dict(r)
            rows.append(d)
        cur.close()
        return rows

    # --- Test data locations APIs ---
    def set_test_data_location(self, session_id: int, name: str, path: str, description: Optional[str] = None) -> int:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO test_data_locations(session_id, name, path, description)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET name = excluded.name, path=excluded.path, description=excluded.description
        """, (session_id, name, path, description))
        entry_id = cur.lastrowid
        cur.close()
        return entry_id

    def get_test_data_locations(self, session_id: int, ) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM test_data_locations WHERE session_id = ? ORDER BY name", (session_id,))
        rows = [self._row_to_dict(r) for r in cur.fetchall()]
        cur.close()
        return rows



if __name__ == "__main__":
    db = SQLiteDB("src/auto_code/db_tables/code_generator.db")
    # db.initialize_schema()
    # sid = db.create_session(session_key="greeks_test", description="code_generation_for_greeks_test")
    # sid=1
    # db.add_message(session_id=sid, role='human', content='Please generate another function')
    fetched_message = db.get_session_messages(1)
    print(fetched_message)
    # print(db.get_session(session_key='gamma_test2'))
    # # Convenience function
    # def open_db(path: str) -> SQLiteDB:
    #     return SQLiteDB(path)




