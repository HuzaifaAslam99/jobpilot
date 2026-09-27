"""Background scanner shared by all web users.

One thread works through everyone's scan jobs in turns (a small batch per user, round-robin), so one
person's 800-email backlog never blocks another person's 7-day scan.

A job:
  1. counting  - import applications you SENT in the period, sync thread dates, then list the IDs of every
                 relevant email in the period (IDs only: fast). total = all of them; already = those an
                 earlier scan finished; the rest become job_items, oldest first.
  2. running   - FAST PASS: fetch emails in bulk (Gmail batch requests) and decide everything with the rules
                 alone, so the board is complete in minutes. Emails the rules weren't sure about are marked
                 ai_pending and shown with the rules' best guess.
     refine    - AI PASS: the AI double-checks only those uncertain emails, several per request (template
                 cache, Groq -> Gemini failover). When every free model is at its limit it stops and the
                 rest are double-checked by a later scan. The user never waits on the AI to see a board.
  3. done / cancelled / error.
Cancel = stop after the current batch. Starting again recounts and skips everything already done, so
progress resumes where it stopped. If the server restarts, running jobs continue from their job_items.
"""
from __future__ import annotations

import threading
import time
import traceback

from . import accounts, db

BATCH = 25              # emails per turn in the fast (rules) pass
AI_BATCH = 6            # emails per turn (and per AI request) in the AI pass
MAX_EMAILS = 1500       # per job; a year of a busy inbox
SENT_QUERY = ("in:sent subject:(application OR apply OR applying OR internship OR intern OR position OR role OR job "
              "OR developer OR engineer OR opportunity OR cv OR resume)")

_worker = None
_wake = threading.Event()


def start() -> "Worker":
    """Start the one background worker for this server process (safe to call on every page load)."""
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = Worker()
        _worker.start()
    return _worker


def enqueue(conn, user_id: int, label: str, query: str) -> int:
    jid = accounts.create_job(conn, user_id, label, query)
    start()
    _wake.set()
    return jid


class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="jobpilot-scanner")
        self.conn = accounts.connect()
        self.services = {}
        # resume after a restart: counting starts over, running continues from its remaining items
        self.conn.execute("UPDATE jobs SET status='queued' WHERE status='counting'")
        self.conn.execute("DELETE FROM job_items WHERE job_id IN (SELECT id FROM jobs WHERE status='queued')")
        self.conn.commit()

    # -------------------------------------------------------------- loop

    def run(self):
        while True:
            jobs = self.conn.execute("SELECT * FROM jobs WHERE status IN ('queued','running') "
                                     "ORDER BY updated_at").fetchall()
            if not jobs:
                _wake.wait(timeout=5)
                _wake.clear()
                continue
            for job in jobs:  # one turn each, then the next user
                try:
                    if job["status"] == "queued":
                        self.count(dict(job))
                    else:
                        self.step(dict(job))
                except Exception as exc:  # one user's problem must not stop everyone else's scans
                    traceback.print_exc()
                    accounts.update_job(self.conn, job["id"], status="error",
                                        message=f"Stopped by an error: {exc.__class__.__name__}: {str(exc)[:200]}")

    def service(self, user_id):
        from .auth import gmail_service
        if user_id not in self.services:
            self.services[user_id] = gmail_service(self.conn, user_id)
        return self.services[user_id]

    def cancelled(self, job_id) -> bool:
        row = self.conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        return not row or row["status"] == "cancelled"

    # -------------------------------------------------------------- phase 1: count

    def count(self, job):
        from .gmail_client import _x, import_sent_applications, sync_thread_dates
        from .inbox_scan import DEFAULT_QUERY
        jid, uid, window = job["id"], job["user_id"], job["query"]
        accounts.update_job(self.conn, jid, status="counting", message="Counting emails in this period…")
        svc, udb = self.service(uid), accounts.user_db(uid)
        try:
            self._count(jid, window, svc, udb)
        finally:
            udb.close()  # never keep a user's file open between turns (Windows can't delete open files)

    def _count(self, jid, window, svc, udb):
        from .gmail_client import import_sent_applications, sync_thread_dates
        from .inbox_scan import DEFAULT_QUERY

        # applications you sent in the period, and real dates / follow-ups for the threads you started
        import_sent_applications(svc, udb, f"{SENT_QUERY} {window}", db.add_application)
        sync_thread_dates(svc, udb)
        # an email skipped earlier as "not about an application" gets another look if it turns out to be in a
        # thread / from a person you applied to (e.g. your sent email was only found in this scan)
        udb.execute("""DELETE FROM messages WHERE label='IGNORED' AND (
                           thread_id IN (SELECT thread_id FROM applications WHERE thread_id IS NOT NULL)
                           OR from_addr IN (SELECT contact_email FROM applications WHERE contact_email IS NOT NULL))""")
        udb.commit()
        if self.cancelled(jid):
            return

        ids = self.list_ids(svc, f"{DEFAULT_QUERY} {window}")
        # replies from people you wrote to, even when the subject has no job words ("Re: quick question")
        contacts = sorted({r["contact_email"] for r in udb.execute(
            "SELECT contact_email FROM applications WHERE contact_email IS NOT NULL")})
        for i in range(0, len(contacts), 30):
            q = "-in:sent -in:spam -in:trash " + window + " (" + " OR ".join(f"from:{c}" for c in contacts[i:i + 30]) + ")"
            ids += [m for m in self.list_ids(svc, q) if m not in set(ids)]
        ids = list(dict.fromkeys(ids))[:MAX_EMAILS]

        done_before = {r["gmail_id"] for r in udb.execute("SELECT gmail_id FROM messages")}
        todo = [m for m in reversed(ids) if m not in done_before]  # Gmail lists newest first; work oldest first
        self.conn.execute("DELETE FROM job_items WHERE job_id=?", (jid,))
        self.conn.executemany("INSERT OR IGNORE INTO job_items(job_id, gmail_id) VALUES (?,?)", [(jid, m) for m in todo])
        self.conn.commit()
        accounts.update_job(self.conn, jid, status="running", total=len(ids), already=len(ids) - len(todo), done=0,
                            deferred=0, message=f"{len(ids)} emails in this period, {len(todo)} to check.")
        if not todo:
            self.finish(jid)

    def list_ids(self, svc, q) -> list[str]:
        from .gmail_client import _x
        out, page = [], None
        while len(out) < MAX_EMAILS:
            resp = _x(svc.users().messages().list(userId="me", q=q, maxResults=500, pageToken=page))
            out += [m["id"] for m in resp.get("messages", [])]
            page = resp.get("nextPageToken")
            if not page:
                break
        return out

    # -------------------------------------------------------------- phase 2: fast pass, then AI pass

    def step(self, job):
        jid, uid = job["id"], job["user_id"]
        if self.cancelled(jid):
            return
        items = [r["gmail_id"] for r in self.conn.execute(
            "SELECT gmail_id FROM job_items WHERE job_id=? AND status='pending' ORDER BY rowid LIMIT ?", (jid, BATCH))]
        if items:
            return self.fast_step(jid, uid, items)
        return self.refine_step(job)

    def fast_step(self, jid, uid, items):
        udb = accounts.user_db(uid)
        try:
            results = process_batch(self.service(uid), udb, items, cache=accounts.TemplateCache(self.conn),
                                    use_llm=False)
        finally:
            udb.close()
        for gid, outcome in results.items():
            self.conn.execute("UPDATE job_items SET status=? WHERE job_id=? AND gmail_id=?", (outcome, jid, gid))
        self.conn.commit()
        cur = accounts.get_job(self.conn, jid)
        if cur["status"] == "running":  # (not if you pressed Stop meanwhile)
            accounts.update_job(self.conn, jid, done=cur["done"] + len(results))
            left = self.conn.execute("SELECT 1 FROM job_items WHERE job_id=? AND status='pending' LIMIT 1",
                                     (jid,)).fetchone()
            if not left:
                self.refine_step(accounts.get_job(self.conn, jid))  # finishes at once if nothing needs the AI

    def refine_step(self, job):
        from .llm import llm_available
        jid, uid = job["id"], job["user_id"]
        if not llm_available():
            return self.finish(jid)  # no AI key configured: the rules' answers are final
        udb = accounts.user_db(uid)
        try:
            pending = [r["gmail_id"] for r in udb.execute(
                "SELECT gmail_id FROM messages WHERE ai_pending=1 ORDER BY received_at LIMIT ?", (AI_BATCH,))]
            if job.get("phase") != "refine":
                if not pending:
                    return self.finish(jid)
                total = udb.execute("SELECT COUNT(*) FROM messages WHERE ai_pending=1").fetchone()[0]
                accounts.update_job(self.conn, jid, phase="refine", refine_total=total, refine_done=0)
                return
            if not pending:
                return self.finish(jid)
            results = process_batch(self.service(uid), udb, pending, cache=accounts.TemplateCache(self.conn),
                                    use_llm=True, mode="refine")
        finally:
            udb.close()
        done = sum(1 for o in results.values() if o != "deferred")
        cur = accounts.get_job(self.conn, jid)
        if cur["status"] != "running":
            return
        accounts.update_job(self.conn, jid, refine_done=(cur["refine_done"] or 0) + done)
        if done == 0:
            self.finish(jid)  # every free model is at its limit: the rest waits for a later scan

    def finish(self, jid):
        job = accounts.get_job(self.conn, jid)
        if job["status"] != "running":
            return
        ids = [r["gmail_id"] for r in self.conn.execute("SELECT gmail_id FROM job_items WHERE job_id=?", (jid,))]
        udb = accounts.user_db(job["user_id"])
        try:
            ignored = sum(1 for i in range(0, len(ids), 500) for _ in udb.execute(
                f"SELECT 1 FROM messages WHERE label='IGNORED' AND gmail_id IN ({','.join('?' * len(ids[i:i + 500]))})",
                ids[i:i + 500])) if ids else 0
            waiting = udb.execute("SELECT COUNT(*) FROM messages WHERE ai_pending=1").fetchone()[0]
        finally:
            udb.close()
        msg = f"Done: {job['total']} emails in this period checked."
        if ignored:
            msg += (f" {ignored} of them weren't about an application you made (job alerts, newsletters, "
                    "promotions), so they're not on your board.")
        from .llm import llm_available
        if waiting and llm_available():
            msg += (f" {waiting} uncertain email(s) still need an AI double-check (the free AI limit was reached); "
                    "the board shows the rules' best guess, and the next scan finishes the check.")
        accounts.update_job(self.conn, jid, status="done", message=msg)


def _fetch(svc, gmail_ids: list[str]) -> dict:
    """Download several emails at once (Gmail batch requests: one round trip for ~20 emails instead of 20)."""
    from .gmail_client import _x, to_email
    raw = {}
    if hasattr(svc, "new_batch_http_request"):
        for i in range(0, len(gmail_ids), 20):
            chunk, got = gmail_ids[i:i + 20], {}

            def keep(request_id, response, exception, got=got):
                if exception is None:
                    got[request_id] = response
            batch = svc.new_batch_http_request(callback=keep)
            for gid in chunk:
                batch.add(svc.users().messages().get(userId="me", id=gid, format="full"), request_id=gid)
            try:
                batch.execute()
            except Exception:
                pass
            raw.update(got)
    for gid in gmail_ids:  # anything a batch missed (e.g. Gmail said "slow down" for one of them): one by one
        if gid not in raw:
            try:
                raw[gid] = _x(svc.users().messages().get(userId="me", id=gid, format="full"))
                time.sleep(0.05)
            except Exception:
                pass
    return {gid: to_email(m) for gid, m in raw.items()}


def _forget(udb, gmail_id: str):
    """Before the AI re-decides an email: drop the rules' earlier record of it (and its unsent draft)."""
    row = udb.execute("SELECT id, application_id FROM messages WHERE gmail_id=?", (gmail_id,)).fetchone()
    if not row:
        return None
    udb.execute("DELETE FROM drafts WHERE message_id=? AND status='PENDING_REVIEW'", (row["id"],))
    udb.execute("DELETE FROM messages WHERE id=?", (row["id"],))
    return row["application_id"]


def process_batch(svc, udb, gmail_ids: list[str], cache=None, use_llm: bool = True,
                  mode: str = "scan") -> dict[str, str]:
    """Process a few emails for one user. -> {gmail_id: 'done' | 'deferred' | 'error'}

    mode 'scan' with use_llm=False: rules only (fast); uncertain emails are saved with the rules' best guess and
    marked ai_pending. mode 'refine': the AI re-decides ai_pending emails; if the AI isn't available the earlier
    record is kept untouched ('deferred')."""
    from .gmail_client import _as_dt, _thread_my_last, _x
    from .inbox_scan import extract, ingest, is_noreply, is_platform
    from .pipeline import process_email
    from .reply_classifier import RULES_FIRST, classify_many

    me = _x(svc.users().getProfile(userId="me"))["emailAddress"].lower()
    threads = {r["thread_id"] for r in udb.execute("SELECT thread_id FROM applications WHERE thread_id IS NOT NULL")}
    contacts = {r["contact_email"] for r in udb.execute("SELECT contact_email FROM applications WHERE contact_email IS NOT NULL")}
    emails = _fetch(svc, gmail_ids)
    outcome, staged, cache_mine, touched = {}, [], {}, set()
    for gid in gmail_ids:
        e = emails.get(gid)
        if e is None:
            outcome[gid] = "error"
            continue
        if e.from_addr == me:
            outcome[gid] = "done"
            continue
        platform = is_platform(e.from_addr) or is_noreply(e.from_addr)
        if not platform:
            mine = _thread_my_last(svc, e.thread_id, me, cache_mine)
            if mine and e.received_at and _as_dt(mine) > _as_dt(e.received_at):
                e.you_replied_at = mine
        tracked = not platform and (e.thread_id in threads or e.from_addr in contacts)
        info, unsure = None, False
        if not tracked:
            info = extract(e, use_llm)  # is it about an application at all? (rules; AI only if unsure and allowed)
            if info.get("deferred"):
                outcome[gid] = "deferred"
                continue
            unsure = bool(info.get("unsure")) and not use_llm
            if not info["related"]:
                if mode == "refine":
                    touched.add(_forget(udb, gid))
                ingest(udb, e, use_llm=use_llm, info=info)  # recorded as ignored
                _mark(udb, gid, unsure)
                outcome[gid] = "done"
                continue
        staged.append((gid, e, tracked, info, unsure))

    classes = classify_many([s[1] for s in staged], use_llm=use_llm, cache=cache) if staged else []
    for (gid, e, tracked, info, unsure), cls in zip(staged, classes):
        if cls.key_details.get("ai_missing"):
            outcome[gid] = "deferred"
            continue
        needs_ai = unsure or (not use_llm and cls.source == "rules" and
                              (cls.label == "UNCLEAR" or cls.confidence < RULES_FIRST))
        try:
            if mode == "refine":
                touched.add(_forget(udb, gid))
            res = process_email(udb, e, use_llm=use_llm, cls=cls) if tracked else \
                ingest(udb, e, use_llm=use_llm, cls=cls, info=info)
            if str(res.get("skipped", "")).startswith("deferred"):
                outcome[gid] = "deferred"
                continue
            _mark(udb, gid, needs_ai)
            if needs_ai and res.get("label") == "UNCLEAR" and res.get("application_id"):
                udb.execute("UPDATE applications SET next_action=? WHERE id=?",
                            ("The AI is double-checking this email; check back in a few minutes.",
                             res["application_id"]))
            outcome[gid] = "done"
        except Exception:
            traceback.print_exc()
            outcome[gid] = "error"
    # a row the rules created that the AI now says isn't an application at all: remove it
    for app_id in touched - {None}:
        left = udb.execute("SELECT COUNT(*) FROM messages WHERE application_id=?", (app_id,)).fetchone()[0]
        if not left:
            udb.execute("DELETE FROM applications WHERE id=? AND posting_url LIKE 'via %'", (app_id,))
    udb.commit()
    return outcome


def _mark(udb, gmail_id: str, pending: bool):
    udb.execute("UPDATE messages SET ai_pending=? WHERE gmail_id=?", (1 if pending else 0, gmail_id))