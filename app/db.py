"""SQLite pipeline store. One file, zero setup, inspectable with any SQLite viewer."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("JOBPILOT_DB", Path(__file__).resolve().parent.parent / "data" / "jobpilot.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    company         TEXT NOT NULL,
    role            TEXT NOT NULL,
    contact_email   TEXT,
    contact_domain  TEXT,
    thread_id       TEXT,            -- Gmail thread of the original outreach
    posting_url     TEXT,
    scam_score      INTEGER,         -- from the detector, if the posting was checked
    status          TEXT NOT NULL DEFAULT 'APPLIED',
    priority        TEXT NOT NULL DEFAULT 'normal',   -- normal | high
    next_action     TEXT,
    applied_at      TEXT,
    last_update     TEXT
);
CREATE INDEX IF NOT EXISTS ix_app_thread ON applications(thread_id);
CREATE INDEX IF NOT EXISTS ix_app_email  ON applications(contact_email);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    gmail_id        TEXT UNIQUE,     -- idempotency: never process the same email twice
    thread_id       TEXT,
    application_id  INTEGER REFERENCES applications(id),
    from_addr       TEXT,
    subject         TEXT,
    snippet         TEXT,
    received_at     TEXT,
    label           TEXT,
    confidence      REAL,
    reasoning       TEXT,
    match_method    TEXT,
    processed_at    TEXT
);

CREATE TABLE IF NOT EXISTS drafts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id  INTEGER REFERENCES applications(id),
    message_id      INTEGER REFERENCES messages(id),
    to_addr         TEXT,
    subject         TEXT,
    body            TEXT,
    status          TEXT NOT NULL DEFAULT 'PENDING_REVIEW',  -- PENDING_REVIEW | APPROVED | DISCARDED
    gmail_draft_id  TEXT,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key             TEXT PRIMARY KEY,
    value           TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id  INTEGER,
    ts              TEXT,
    kind            TEXT,
    detail          TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # the web app reads while the background scanner writes
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn):
    """Add columns introduced after the first version, without losing data."""
    cols = {
        "applications": {"subject": "TEXT", "last_contact_at": "TEXT", "followups_sent": "INTEGER DEFAULT 0",
                         "outreach_kind": "TEXT",   # application | cold
                         "kind_locked": "INTEGER DEFAULT 0"},  # 1 = set by you with set-kind; never re-guessed
        "drafts": {"thread_id": "TEXT", "kind": "TEXT DEFAULT 'reply'"},
        # 1 = decided by the rules alone while the AI wasn't used; the AI double-checks it later (web app)
        "messages": {"ai_pending": "INTEGER DEFAULT 0"},
    }
    for table, wanted in cols.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, typ in wanted.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
    conn.commit()


def get_setting(conn, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row[0] if row and row[0] not in (None, "") else default


def set_setting(conn, key: str, value: str | None):
    conn.execute("INSERT INTO settings(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))
    conn.commit()


def candidate_name(conn) -> str:
    """Name used to sign drafts: the user's own setting (web app), else CANDIDATE_NAME from .env (CLI)."""
    return get_setting(conn, "name") or os.getenv("CANDIDATE_NAME", "Huzaifa")


def log_event(conn, application_id, kind: str, detail: dict | str):
    conn.execute("INSERT INTO events(application_id, ts, kind, detail) VALUES (?,?,?,?)",
                 (application_id, now(), kind, detail if isinstance(detail, str) else json.dumps(detail)))


def add_application(conn, company, role, contact_email=None, thread_id=None, posting_url=None,
                    scam_score=None, applied_at=None, subject=None, kind="application") -> int:
    domain = contact_email.split("@")[-1].lower() if contact_email else None
    cur = conn.execute(
        """INSERT INTO applications(company, role, contact_email, contact_domain, thread_id, posting_url,
               scam_score, status, next_action, applied_at, last_update)
           VALUES (?,?,?,?,?,?,?,'APPLIED','Wait for reply (follow up after 7 days)',?,?)""",
        (company, role, contact_email.lower() if contact_email else None, domain, thread_id, posting_url,
         scam_score, applied_at or now(), now()))
    conn.execute("UPDATE applications SET subject=?, last_contact_at=?, outreach_kind=? WHERE id=?",
                 (subject, applied_at or now(), kind, cur.lastrowid))
    log_event(conn, cur.lastrowid, "applied", {"company": company, "role": role})
    conn.commit()
    return cur.lastrowid


def pipeline(conn):
    return [dict(r) for r in conn.execute(
        """SELECT a.*, (SELECT label FROM messages m WHERE m.application_id=a.id AND m.label != 'IGNORED'
                   ORDER BY m.received_at DESC, m.id DESC LIMIT 1) AS last_label,
                  (SELECT MAX(received_at) FROM messages m WHERE m.application_id=a.id AND m.label != 'IGNORED')
                   AS last_reply_at,
                  (SELECT COUNT(*) FROM drafts d WHERE d.application_id=a.id AND d.status='PENDING_REVIEW') AS pending_drafts
           FROM applications a
           ORDER BY CASE a.priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END, a.last_update DESC""")]


def pending_drafts(conn):
    return [dict(r) for r in conn.execute(
        """SELECT d.*, a.company, a.role, m.label, m.subject AS reply_subject
           FROM drafts d JOIN applications a ON a.id=d.application_id LEFT JOIN messages m ON m.id=d.message_id
           WHERE d.status='PENDING_REVIEW' ORDER BY d.id""")]


def repair_statuses(conn) -> int:
    """Fix rows an older email wrongly re-opened before that was prevented: if the NEWEST email for an
    application is a rejection, the application is rejected (unless you set its status yourself)."""
    n = 0
    for a in conn.execute("SELECT id, status, next_action FROM applications WHERE status != 'REJECTED'").fetchall():
        if (a["next_action"] or "").startswith("Set by you"):
            continue
        latest = conn.execute("""SELECT label FROM messages WHERE application_id=? AND label != 'IGNORED'
                                 ORDER BY received_at DESC, id DESC LIMIT 1""", (a["id"],)).fetchone()
        if latest and latest["label"] == "REJECTED":
            conn.execute("UPDATE applications SET status='REJECTED', priority='normal', next_action=? WHERE id=?",
                         ("Closed (rejected: their latest email was a rejection).", a["id"]))
            log_event(conn, a["id"], "status_repaired", {"from": a["status"], "to": "REJECTED"})
            n += 1
    conn.commit()
    return n


def merge_applications(conn, source_id: int, target_id: int):
    """Two rows that are really one application (e.g. found once from your sent email and once from their
    reply): move everything to target, keep the earliest date and any missing details, delete source.
    The newest email then decides the status again."""
    src = conn.execute("SELECT * FROM applications WHERE id=?", (source_id,)).fetchone()
    if not src or source_id == target_id:
        return
    for t in ("messages", "drafts", "events"):
        conn.execute(f"UPDATE {t} SET application_id=? WHERE application_id=?", (target_id, source_id))
    conn.execute("""UPDATE applications SET
            contact_email=COALESCE(contact_email, ?), contact_domain=COALESCE(contact_domain, ?),
            thread_id=COALESCE(thread_id, ?), subject=COALESCE(subject, ?),
            applied_at=CASE WHEN ? IS NOT NULL AND ? < applied_at THEN ? ELSE applied_at END,
            role=CASE WHEN role LIKE '(role%' THEN ? ELSE role END WHERE id=?""",
                 (src["contact_email"], src["contact_domain"], src["thread_id"], src["subject"],
                  src["applied_at"], src["applied_at"], src["applied_at"], src["role"], target_id))
    conn.execute("DELETE FROM applications WHERE id=?", (source_id,))
    latest = conn.execute("""SELECT label FROM messages WHERE application_id=? AND label != 'IGNORED'
                             ORDER BY received_at DESC LIMIT 1""", (target_id,)).fetchone()
    if latest and latest["label"] == "REJECTED":
        conn.execute("UPDATE applications SET status='REJECTED', priority='normal', "
                     "next_action='Closed (rejected).' WHERE id=?", (target_id,))
    log_event(conn, target_id, "merged", {"from": source_id})
    conn.commit()


def reset(conn):
    conn.executescript("DELETE FROM drafts; DELETE FROM messages; DELETE FROM events; DELETE FROM applications;")
    conn.commit()


def clear_replies(conn):
    conn.executescript("""DELETE FROM drafts; DELETE FROM messages;
        UPDATE applications SET status='APPLIED', priority='normal',
               next_action='Wait for reply (follow up after 7 days)';""")
    conn.commit()