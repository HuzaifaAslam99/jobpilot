"""Multi-user storage for the web app.

  data/app.db            shared: users (+ their encrypted Google token), scan jobs and their progress,
                         pending sign-ins, and the template cache
  data/users/<id>.db     one private pipeline database per user (same schema as the CLI's jobpilot.db)

Google refresh tokens are encrypted with TOKEN_KEY (a Fernet key kept in your secrets), so a copy of the
database alone can't be used to read anyone's Gmail.
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from . import db

DATA = Path(os.getenv("JOBPILOT_HOME", Path(__file__).resolve().parent.parent / "data"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT UNIQUE NOT NULL,
    token_enc   TEXT,
    created_at  TEXT,
    last_seen   TEXT
);
CREATE TABLE IF NOT EXISTS oauth_states (      -- sign-ins in progress (Google redirects to a NEW browser session)
    state       TEXT PRIMARY KEY,
    verifier    TEXT,
    created     REAL
);
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    label       TEXT,                 -- "Last 7 days: 20 Sep 2026 - 26 Sep 2026 (7 days)"
    query       TEXT,                 -- Gmail date filter: after:... before:...
    status      TEXT NOT NULL,        -- queued | counting | running | done | cancelled | error
    total       INTEGER DEFAULT 0,    -- emails in the period (after filters)
    already     INTEGER DEFAULT 0,    -- of those, done by earlier scans (skipped)
    done        INTEGER DEFAULT 0,    -- processed by this job
    deferred    INTEGER DEFAULT 0,    -- left for later because every AI model was at its limit
    message     TEXT,
    created_at  TEXT,
    updated_at  TEXT
);
CREATE TABLE IF NOT EXISTS job_items (
    job_id      INTEGER,
    gmail_id    TEXT,
    status      TEXT DEFAULT 'pending',   -- pending | done | deferred | error
    PRIMARY KEY (job_id, gmail_id)
);
CREATE TABLE IF NOT EXISTS template_cache (    -- hashed template fingerprint -> label; no personal text
    key         TEXT PRIMARY KEY,
    label       TEXT,
    confidence  REAL,
    reasoning   TEXT,
    hits        INTEGER DEFAULT 0,
    created_at  TEXT
);
"""

_lock = threading.Lock()


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path or DATA / "app.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    for name, typ in (("phase", "TEXT DEFAULT 'scan'"), ("refine_total", "INTEGER DEFAULT 0"),
                      ("refine_done", "INTEGER DEFAULT 0")):
        if name not in have:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {typ}")
    conn.commit()
    return conn


# ------------------------------------------------------------------ users and tokens

def _fernet():
    from cryptography.fernet import Fernet
    key = os.getenv("TOKEN_KEY")
    if not key:  # local development: make one and keep it next to the data (hosted: set TOKEN_KEY in secrets)
        f = DATA / ".token_key"
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            f.write_text(Fernet.generate_key().decode())
        key = f.read_text().strip()
    return Fernet(key.encode())


def upsert_user(conn, email: str, token_json: str | None) -> int:
    email = email.lower()
    row = conn.execute("SELECT id, token_enc FROM users WHERE email=?", (email,)).fetchone()
    enc = _fernet().encrypt(token_json.encode()).decode() if token_json else None
    if row:
        conn.execute("UPDATE users SET token_enc=COALESCE(?, token_enc), last_seen=? WHERE id=?", (enc, db.now(), row["id"]))
        uid = row["id"]
    else:
        uid = conn.execute("INSERT INTO users(email, token_enc, created_at, last_seen) VALUES (?,?,?,?)",
                           (email, enc, db.now(), db.now())).lastrowid
    conn.commit()
    return uid


def user_token(conn, user_id: int) -> str | None:
    row = conn.execute("SELECT token_enc FROM users WHERE id=?", (user_id,)).fetchone()
    if not row or not row["token_enc"]:
        return None
    return _fernet().decrypt(row["token_enc"].encode()).decode()


def save_token(conn, user_id: int, token_json: str):
    conn.execute("UPDATE users SET token_enc=? WHERE id=?", (_fernet().encrypt(token_json.encode()).decode(), user_id))
    conn.commit()


def user_db(user_id: int) -> sqlite3.Connection:
    """The user's own pipeline database (applications, messages, drafts)."""
    return db.connect(DATA / "users" / f"{int(user_id)}.db")


def delete_user(conn, user_id: int):
    """Sign out AND forget: removes the stored Google token, scan history and the user's pipeline database."""
    conn.execute("DELETE FROM job_items WHERE job_id IN (SELECT id FROM jobs WHERE user_id=?)", (user_id,))
    conn.execute("DELETE FROM jobs WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    path = DATA / "users" / f"{int(user_id)}.db"
    try:
        for suffix in ("", "-wal", "-shm"):
            f = Path(f"{path}{suffix}")
            if f.exists():
                f.unlink()
    except PermissionError:
        # Windows won't delete a file another connection still has open: empty it instead (same result for the
        # user), and the file itself goes the next time someone deletes or the server restarts
        c = sqlite3.connect(path, timeout=30)
        for table in ("drafts", "messages", "events", "applications", "settings"):
            try:
                c.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        c.commit()
        c.close()


# ------------------------------------------------------------------ sign-in state (CSRF + PKCE)

def new_oauth_state(conn) -> tuple[str, str]:
    state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(64)
    conn.execute("DELETE FROM oauth_states WHERE created < ?", (time.time() - 900,))
    conn.execute("INSERT INTO oauth_states(state, verifier, created) VALUES (?,?,?)", (state, verifier, time.time()))
    conn.commit()
    return state, verifier


def pop_oauth_state(conn, state: str) -> str | None:
    """The PKCE verifier for a sign-in we started (within 15 min), used once; None for unknown/forged states."""
    row = conn.execute("SELECT verifier, created FROM oauth_states WHERE state=?", (state or "",)).fetchone()
    conn.execute("DELETE FROM oauth_states WHERE state=?", (state or "",))
    conn.commit()
    if not row or row["created"] < time.time() - 900:
        return None
    return row["verifier"]


# ------------------------------------------------------------------ template cache

class TemplateCache:
    """Shared across users: fingerprint of a portal email template -> the AI's label for it."""

    def __init__(self, conn):
        self.conn = conn

    def get(self, key):
        row = self.conn.execute("SELECT label, confidence, reasoning FROM template_cache WHERE key=?", (key,)).fetchone()
        if row:
            self.conn.execute("UPDATE template_cache SET hits=hits+1 WHERE key=?", (key,))
            self.conn.commit()
            return dict(row)
        return None

    def put(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO template_cache(key, label, confidence, reasoning, hits, created_at) "
                          "VALUES (?,?,?,?,0,?)", (key, value["label"], value["confidence"], value["reasoning"], db.now()))
        self.conn.commit()


# ------------------------------------------------------------------ jobs

def create_job(conn, user_id: int, label: str, query: str) -> int:
    with _lock:
        active = conn.execute("SELECT id FROM jobs WHERE user_id=? AND status IN ('queued','counting','running')",
                              (user_id,)).fetchone()
        if active:
            return active["id"]  # one scan per user at a time
        jid = conn.execute("INSERT INTO jobs(user_id, label, query, status, created_at, updated_at) "
                           "VALUES (?,?,?,'queued',?,?)", (user_id, label, query, db.now(), db.now())).lastrowid
        conn.commit()
        return jid


def update_job(conn, job_id: int, **fields):
    if not fields:
        return
    fields["updated_at"] = db.now()
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
    conn.commit()


def get_job(conn, job_id: int):
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def latest_job(conn, user_id: int):
    row = conn.execute("SELECT * FROM jobs WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
    return dict(row) if row else None


def cancel_job(conn, job_id: int):
    conn.execute("UPDATE jobs SET status='cancelled', message='Stopped by you. Start again any time: finished "
                 "emails are kept and skipped.', updated_at=? WHERE id=? AND status IN ('queued','counting','running')",
                 (db.now(), job_id))
    conn.commit()


def progress(job: dict | None) -> dict:
    """Numbers for the loader. percent = (already done before + done now) / total emails in the period."""
    if not job:
        return {"percent": 0, "text": ""}
    total, already, done, deferred = job["total"] or 0, job["already"] or 0, job["done"] or 0, job["deferred"] or 0
    finished = min(total, already + done)
    pct = 100 if job["status"] == "done" or total == 0 and job["status"] not in ("queued", "counting") \
        else int(finished * 100 / total) if total else 0
    if job["status"] in ("queued",):
        text = "Waiting for a free slot…"
    elif job["status"] == "counting":
        text = "Counting emails in this period…"
    else:
        text = f"{finished} of {total} emails checked ({pct}%)"
        if already:
            text += f" · {already} were already done earlier"
        if deferred:
            text += f" · {deferred} left for later (AI limit)"
    out = {"percent": pct, "text": text, "finished": finished, "total": total, "refining": False}
    if job.get("phase") == "refine" and job["status"] == "running":
        rt, rd = job.get("refine_total") or 0, job.get("refine_done") or 0
        out.update(refining=True, refine_percent=int(rd * 100 / rt) if rt else 100,
                   refine_text=f"AI double-checking the emails the rules weren't sure about: {rd} of {rt}")
    return out