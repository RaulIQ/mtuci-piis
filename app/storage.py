import sqlite3
from pathlib import Path


class RequestLogStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    filename TEXT,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    model_version TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def write(
        self,
        created_at: str,
        filename: str | None,
        predicted_class: str,
        confidence: float,
        latency_ms: float,
        model_version: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO requests
                (created_at, filename, predicted_class, confidence, latency_ms, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    filename,
                    predicted_class,
                    confidence,
                    latency_ms,
                    model_version,
                ),
            )
            conn.commit()

