import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


class Database:
    def __init__(self):
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
            (DATA_DIR / "generations").mkdir(parents=True, exist_ok=True)
            db_path = DATA_DIR / "metaai.db"
            self._local.conn = sqlite3.connect(str(db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._init_db()
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                mime_type TEXT,
                file_size INTEGER,
                media_id TEXT,
                upload_session_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                generation_type TEXT NOT NULL,
                input_media_ids TEXT,
                result_json TEXT,
                video_urls TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()

    def add_upload(self, filename: str, original_name: str, mime_type: str,
                   file_size: int, media_id: Optional[str] = None,
                   upload_session_id: Optional[str] = None) -> int:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO uploads (filename, original_name, mime_type, file_size, media_id, upload_session_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (filename, original_name, mime_type, file_size, media_id, upload_session_id, now)
        )
        conn.commit()
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_uploads(self) -> list:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM uploads ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_upload(self, upload_id: int) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        return dict(row) if row else None

    def delete_upload(self, upload_id: int) -> bool:
        conn = self._get_conn()
        upload = self.get_upload(upload_id)
        if not upload:
            return False
        filepath = DATA_DIR / "uploads" / upload["filename"]
        if filepath.exists():
            filepath.unlink()
        conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
        conn.commit()
        return True

    def update_upload_media_id(self, upload_id: int, media_id: str,
                               upload_session_id: Optional[str] = None):
        conn = self._get_conn()
        if upload_session_id:
            conn.execute(
                "UPDATE uploads SET media_id = ?, upload_session_id = ? WHERE id = ?",
                (media_id, upload_session_id, upload_id)
            )
        else:
            conn.execute(
                "UPDATE uploads SET media_id = ? WHERE id = ?",
                (media_id, upload_id)
            )
        conn.commit()

    def add_generation(self, prompt: str, generation_type: str,
                       input_media_ids: Optional[list] = None,
                       result_json: Optional[dict] = None,
                       video_urls: Optional[list] = None,
                       status: str = "pending") -> int:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO generations (prompt, generation_type, input_media_ids, result_json, video_urls, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (prompt, generation_type,
             json.dumps(input_media_ids) if input_media_ids else None,
             json.dumps(result_json) if result_json else None,
             json.dumps(video_urls) if video_urls else None,
             status, now)
        )
        conn.commit()
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_generations(self) -> list:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM generations ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d["input_media_ids"]:
                d["input_media_ids"] = json.loads(d["input_media_ids"])
            if d["result_json"]:
                d["result_json"] = json.loads(d["result_json"])
            if d["video_urls"]:
                d["video_urls"] = json.loads(d["video_urls"])
            result.append(d)
        return result

    def get_generation(self, gen_id: int) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        if d["input_media_ids"]:
            d["input_media_ids"] = json.loads(d["input_media_ids"])
        if d["result_json"]:
            d["result_json"] = json.loads(d["result_json"])
        if d["video_urls"]:
            d["video_urls"] = json.loads(d["video_urls"])
        return d

    def update_generation(self, gen_id: int, **kwargs):
        conn = self._get_conn()
        fields = []
        values = []
        for key, value in kwargs.items():
            if key in ("prompt", "generation_type", "status"):
                fields.append(f"{key} = ?")
                values.append(value)
            elif key in ("input_media_ids", "result_json", "video_urls"):
                fields.append(f"{key} = ?")
                values.append(json.dumps(value) if value else None)
        if fields:
            values.append(gen_id)
            conn.execute(
                f"UPDATE generations SET {', '.join(fields)} WHERE id = ?", values
            )
            conn.commit()

    def delete_generation(self, gen_id: int) -> bool:
        conn = self._get_conn()
        gen = self.get_generation(gen_id)
        if not gen:
            return False
        conn.execute("DELETE FROM generations WHERE id = ?", (gen_id,))
        conn.commit()
        return True


db = Database()
