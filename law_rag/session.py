"""SQLite case snapshots; source documents remain in the existing evidence store."""
from pathlib import Path
from contextlib import closing
import json
import sqlite3


class SessionConflict(RuntimeError):
    pass


class CaseNotFound(LookupError):
    pass


class SessionStore:
    def __init__(self, filename):
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.filename)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                payload TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def load(self, case_id):
        with closing(sqlite3.connect(self.filename)) as connection, connection:
            row = connection.execute("SELECT payload FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise CaseNotFound(case_id)
        return json.loads(row[0])

    def save(self, payload):
        revision = payload["revision"]
        saved = {**payload, "revision": revision + 1}
        serialized = json.dumps(saved, ensure_ascii=False)
        with closing(sqlite3.connect(self.filename)) as connection, connection:
            if revision == 0:
                connection.execute("INSERT INTO cases (case_id, revision, payload) VALUES (?, ?, ?)",
                                   (saved["case_id"], 1, serialized))
            else:
                cursor = connection.execute("""UPDATE cases SET revision = ?, payload = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE case_id = ? AND revision = ?""", (revision + 1, serialized, saved["case_id"], revision))
                if cursor.rowcount != 1:
                    raise SessionConflict("案件已被另一请求更新，请重新加载后续办。")
        return revision + 1
