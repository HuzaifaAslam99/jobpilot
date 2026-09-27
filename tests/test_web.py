"""Multi-user web version: accounts, batching/caching, and the background scanner against a fake Gmail."""
import base64
import os

import pytest

os.environ["JOBPILOT_RULES_ONLY"] = "1"
os.environ["JOBPILOT_OFFLINE"] = "1"

from app import accounts, db, worker  # noqa: E402
from app.reply_classifier import Email, classify_many  # noqa: E402

ME = "student@gmail.com"


# ------------------------------------------------------------------ a tiny fake Gmail

def _msg(mid, frm, subject, body, date, thread=None, to=ME):
    return {"id": mid, "threadId": thread or f"t-{mid}", "snippet": body[:120],
            "payload": {"mimeType": "text/plain", "headers": [
                {"name": "From", "value": frm}, {"name": "To", "value": to},
                {"name": "Subject", "value": subject}, {"name": "Date", "value": date}],
                "body": {"data": base64.urlsafe_b64encode(body.encode()).decode()}}}


class _R:
    def __init__(self, v):
        self.v = v

    def execute(self):
        return self.v


class FakeGmail:
    def __init__(self, inbox, sent=()):
        self.inbox, self.sent, self.fetches = list(inbox), list(sent), 0

    def users(self):
        return self

    def getProfile(self, userId):
        return _R({"emailAddress": ME})

    def messages(self):
        return self

    def threads(self):
        return self

    def list(self, userId, q, maxResults=100, pageToken=None):
        import re
        pool = self.sent if re.search(r"(^|\s)in:sent", q) else self.inbox
        return _R({"messages": [{"id": m["id"]} for m in reversed(pool)]})  # newest first, like Gmail

    def get(self, userId, id, format="full", **kw):
        every = {m["id"]: m for m in self.inbox + self.sent}
        if id in every:
            self.fetches += 1
            return _R(every[id])
        return _R({"messages": [m for m in self.inbox + self.sent if m["threadId"] == id]})  # threads().get


LINKEDIN_REJECT = ("jobs-noreply@linkedin.com", "Your application to Web3 Product Intern at Gate",
                   "Your update from Gate\nThank you for your interest. Unfortunately, we will not be moving forward "
                   "with your application.")
NEWSLETTER = ("news@cryptotalentlabs.io", "Your job search filter is costing you", "Plus the hack that finds roles.")


def _inbox(n_extra=0):
    msgs = [_msg("m1", f"LinkedIn <{LINKEDIN_REJECT[0]}>", LINKEDIN_REJECT[1], LINKEDIN_REJECT[2],
                 "Mon, 18 May 2026 05:35:00 +0000"),
            _msg("m2", f"Doris <{NEWSLETTER[0]}>", NEWSLETTER[1], NEWSLETTER[2], "Fri, 11 Sep 2026 10:00:00 +0000")]
    for i in range(n_extra):
        msgs.append(_msg(f"x{i}", "LinkedIn <jobs-noreply@linkedin.com>", f"Your application to Intern {i} at Acme{i}",
                         f"Your update from Acme{i}\nUnfortunately, we will not be moving forward.",
                         "Tue, 19 May 2026 05:35:00 +0000"))
    return msgs


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DATA", tmp_path)
    monkeypatch.setenv("TOKEN_KEY", "")
    return tmp_path


# ------------------------------------------------------------------ accounts

def test_tokens_are_encrypted_and_users_are_separate(home):
    c = accounts.connect()
    a = accounts.upsert_user(c, "A@gmail.com", '{"refresh_token": "secret-a"}')
    b = accounts.upsert_user(c, "b@gmail.com", '{"refresh_token": "secret-b"}')
    assert a != b and accounts.upsert_user(c, "a@gmail.com", None) == a      # same person, same id
    raw = c.execute("SELECT token_enc FROM users WHERE id=?", (a,)).fetchone()[0]
    assert "secret-a" not in raw and "secret-a" in accounts.user_token(c, a)  # stored encrypted
    ua, ub = accounts.user_db(a), accounts.user_db(b)
    db.add_application(ua, "Gate", "Intern")
    assert db.pipeline(ub) == []                                             # b can't see a's board
    ua.close(); ub.close()
    accounts.delete_user(c, a)
    assert not (home / "users" / f"{a}.db").exists() and accounts.user_token(c, a) is None


def test_delete_still_wipes_data_when_the_file_is_open(home, monkeypatch):
    from pathlib import Path
    c = accounts.connect()
    a = accounts.upsert_user(c, "a@gmail.com", None)
    ua = accounts.user_db(a)
    db.add_application(ua, "Gate", "Intern")
    real_unlink = Path.unlink

    def windows_like(self, *args, **kw):  # simulate Windows refusing to delete an open file
        if self.name.startswith(f"{a}.db"):
            raise PermissionError("in use")
        return real_unlink(self, *args, **kw)
    monkeypatch.setattr(Path, "unlink", windows_like)
    accounts.delete_user(c, a)
    assert db.pipeline(ua) == [] and accounts.user_token(c, a) is None


def test_sign_in_state_is_one_time(home):
    c = accounts.connect()
    state, verifier = accounts.new_oauth_state(c)
    assert accounts.pop_oauth_state(c, state) == verifier
    assert accounts.pop_oauth_state(c, state) is None          # can't be replayed
    assert accounts.pop_oauth_state(c, "forged") is None


def test_progress_counts_earlier_work():
    job = {"status": "running", "total": 200, "already": 60, "done": 40, "deferred": 0}
    p = accounts.progress(job)
    assert p["percent"] == 50 and "100 of 200" in p["text"] and "60 were already done" in p["text"]


# ------------------------------------------------------------------ rules first, cache, batching

def test_classify_many_rules_first_then_one_batched_ai_call(monkeypatch):
    from app import reply_classifier as rc
    calls = []

    def fake_ai(system, user, **kw):
        calls.append(user.count("### EMAIL"))
        n = user.count("### EMAIL")
        return {"results": [{"i": i, "label": "INTERESTED", "confidence": 0.9, "reasoning": "x"} for i in range(1, n + 1)]}
    monkeypatch.setattr(rc, "chat_json", fake_ai)
    clear = Email("a", "a", "hr@x.example", "HR", "Update", "Unfortunately, we will not be moving forward.")
    vague = [Email(f"v{i}", f"v{i}", f"p{i}@y.example", "P", "Quick note", "We liked chatting with you.") for i in range(8)]
    out = classify_many([clear] + vague, use_llm=True)
    assert out[0].label == "REJECTED" and out[0].source == "rules"           # no AI needed
    assert calls == [6, 2]                                                   # 8 unclear emails -> 2 requests, not 8
    assert all(o.label == "INTERESTED" for o in out[1:])


def test_template_cache_is_shared_between_users(monkeypatch, home):
    from app import reply_classifier as rc
    calls = []
    monkeypatch.setattr(rc, "chat_json", lambda s, u, **k: calls.append(1) or
                        {"results": [{"i": 1, "label": "AUTO_REPLY", "confidence": 0.95, "reasoning": "status note"}]})
    cache = accounts.TemplateCache(accounts.connect())
    note = lambda who, co: Email(who, who, "noreply@portal.example", "Portal", f"Status for {co}",
                                 f"Hello {who.title()}, the team at {co} is still reviewing profiles.")
    first = classify_many([note("ali", "Acme")], use_llm=True, cache=cache)[0]
    second = classify_many([note("sara", "Globex")], use_llm=True, cache=cache)[0]  # different user, same template
    assert first.source == "llm" and second.source == "cache" and len(calls) == 1


# ------------------------------------------------------------------ the scanner

def _run_job(home, svc, user_id, window="after:0 before:9999999999"):
    c = accounts.connect()
    jid = accounts.create_job(c, user_id, "test", window)
    w = worker.Worker.__new__(worker.Worker)      # the worker's logic, without starting its thread
    w.conn, w.services = c, {user_id: svc}
    job = accounts.get_job(c, jid)
    w.count(job)
    while accounts.get_job(c, jid)["status"] == "running":
        w.step(accounts.get_job(c, jid))
    return c, jid, w


def test_scan_counts_first_then_processes_and_ignores_newsletters(home):
    c = accounts.connect()
    uid = accounts.upsert_user(c, ME, None)
    c, jid, _ = _run_job(home, FakeGmail(_inbox()), uid)
    job = accounts.get_job(c, jid)
    assert job["status"] == "done" and job["total"] == 2 and job["done"] == 2
    u = accounts.user_db(uid)
    board = db.pipeline(u)
    u.close()
    assert [(r["company"], r["status"]) for r in board] == [("Gate", "REJECTED")]


def test_cancel_then_restart_resumes_instead_of_starting_over(home, monkeypatch):
    monkeypatch.setattr(worker, "BATCH", 6)
    c = accounts.connect()
    uid = accounts.upsert_user(c, ME, None)
    svc = FakeGmail(_inbox(n_extra=10))            # 12 emails
    jid = accounts.create_job(c, uid, "test", "after:0 before:9999999999")
    w = worker.Worker.__new__(worker.Worker)
    w.conn, w.services = c, {uid: svc}
    w.count(accounts.get_job(c, jid))
    w.step(accounts.get_job(c, jid))               # one batch (6 emails) ...
    accounts.cancel_job(c, jid)                    # ... then the user presses Stop
    fetched_before = svc.fetches
    c, jid2, _ = _run_job(home, svc, uid)          # start again
    job = accounts.get_job(c, jid2)
    assert job["already"] == 6 and job["done"] == 6 and job["status"] == "done"
    assert svc.fetches - fetched_before == 6       # only the remaining emails were downloaded again
    assert accounts.progress(job)["percent"] == 100


def test_round_robin_keeps_a_small_scan_fast(home):
    c = accounts.connect()
    big = accounts.upsert_user(c, "big@gmail.com", None)
    small = accounts.upsert_user(c, "small@gmail.com", None)
    w = worker.Worker.__new__(worker.Worker)
    w.conn, w.services = c, {big: FakeGmail(_inbox(n_extra=30)), small: FakeGmail(_inbox())}
    j_big = accounts.create_job(c, big, "big", "after:0")
    j_small = accounts.create_job(c, small, "small", "after:0")
    for j in (j_big, j_small):
        w.count(accounts.get_job(c, j))
    turns = 0
    while accounts.get_job(c, j_small)["status"] == "running":
        for j in (j_big, j_small):                 # the worker loop: one batch each, in turn
            if accounts.get_job(c, j)["status"] == "running":
                w.step(accounts.get_job(c, j))
        turns += 1
    assert turns == 1 and accounts.get_job(c, j_big)["status"] == "running"  # small finished while big still going


def test_one_scan_per_user_at_a_time(home):
    c = accounts.connect()
    uid = accounts.upsert_user(c, ME, None)
    assert accounts.create_job(c, uid, "a", "q") == accounts.create_job(c, uid, "b", "q")


def test_sent_application_joins_the_row_found_from_their_reply(home):
    from app.gmail_client import import_sent_applications
    udb = db.connect(home / "u.db")
    db.add_application(udb, "Menzies-Ras", "Menzies-RAS Summer", contact_email="hr@menzies-ras.com.pk",
                       thread_id="t-hr", applied_at="2026-07-24T10:00:00+00:00")
    sent = _msg("s1", f"Me <{ME}>", "Application for IT & IS Intern", "Hi MENZIES-RAS Pvt Ltd team, I'm applying...",
                "Mon, 01 Jun 2026 09:00:00 +0000", to="hr@menzies-ras.com.pk", thread="s1")
    n = import_sent_applications(FakeGmail([], sent=[sent]), udb, "in:sent", db.add_application)
    rows = db.pipeline(udb)
    assert n == 0 and len(rows) == 1 and rows[0]["applied_at"].startswith("2026-06-01")


def test_board_filters_by_period_type_and_date_basis():
    from datetime import date, datetime, timedelta, timezone
    from app.board_filters import custom_window, in_window, kind_matches
    now = datetime.now(timezone.utc)
    iso = lambda days: (now - timedelta(days=days)).isoformat()
    old_app_new_reply = {"applied_at": iso(100), "last_reply_at": iso(1), "outreach_kind": "application"}
    cold = {"applied_at": iso(3), "outreach_kind": "cold"}
    frm, to = custom_window(date.today() - timedelta(days=6), date.today())
    assert in_window(old_app_new_reply, frm, to, "activity")          # recent activity counts
    assert not in_window(old_app_new_reply, frm, to, "applied")       # but it wasn't applied this week
    assert in_window(cold, None, None)                                # "All time"
    assert kind_matches(cold, "cold") and not kind_matches(cold, "application") and kind_matches(cold, "all")


def test_reply_to_a_promo_email_is_not_imported_as_an_application(home):
    from app.gmail_client import import_sent_applications
    udb = db.connect(home / "p.db")
    reply = _msg("r2", f"Me <{ME}>", "Re: Ready to make $60-120k/yr as a blockchain developer?", "Thanks, interested",
                 "Mon, 01 Apr 2026 09:00:00 +0000", thread="promo-thread", to="gregory@dappuniversity.com")
    started = _msg("s2", f"Me <{ME}>", "Application for Blockchain Developer", "Hi Acme team, I'm applying...",
                   "Mon, 01 Apr 2026 10:00:00 +0000", thread="s2", to="hr@acme.example")
    promo = _msg("p1", "Gregory <gregory@dappuniversity.com>", "Ready to make $60-120k/yr as a blockchain developer?",
                 "Join my bootcamp", "Sun, 31 Mar 2026 09:00:00 +0000", thread="promo-thread")
    # Gmail gives a thread the id of your first saved DRAFT, so an email you composed usually has id != threadId
    drafted = _msg("s3", f"Me <{ME}>", "Web3 Developer", "Hi Krypto Hive team, I couldn't find open roles...",
                   "Mon, 01 Apr 2026 11:00:00 +0000", thread="draft-123", to="hello@kryptohive.example")
    n = import_sent_applications(FakeGmail([promo], sent=[reply, started, drafted]), udb, "in:sent", db.add_application)
    assert n == 2 and sorted(r["company"] for r in db.pipeline(udb)) == ["Acme", "Krypto Hive"]


def test_company_name_from_a_career_phrase():
    from app.inbox_scan import _clean_company
    assert _clean_company("pursuing a career with LantroTech") == "LantroTech"
    assert _clean_company("joining Binance") == "Binance"


def test_done_message_explains_ignored_emails(home):
    c = accounts.connect()
    uid = accounts.upsert_user(c, ME, None)
    c, jid, _ = _run_job(home, FakeGmail(_inbox()), uid)
    assert "1 of them weren't about an application" in accounts.get_job(c, jid)["message"]


def test_ignored_reply_gets_rechecked_once_its_application_is_known(home):
    c = accounts.connect()
    uid = accounts.upsert_user(c, ME, None)
    udb = accounts.user_db(uid)
    udb.execute("INSERT INTO messages(gmail_id, thread_id, from_addr, label) VALUES ('old', 't-x', 'hr@x.example', 'IGNORED')")
    db.add_application(udb, "X", "Intern", contact_email="hr@x.example", thread_id="t-x")
    udb.close()
    _run_job(home, FakeGmail([]), uid)
    udb = accounts.user_db(uid)
    assert udb.execute("SELECT COUNT(*) FROM messages WHERE gmail_id='old'").fetchone()[0] == 0  # will be read again
    udb.close()


def _human_reply_inbox():
    return [_msg("h1", "Sara <sara@acme.example>", "Re: Application for Intern", "We enjoyed your profile; let's talk soon.",
                 "Tue, 15 Sep 2026 10:00:00 +0000", thread="t-acme")]


def _acme_user(c):
    uid = accounts.upsert_user(c, ME, None)
    u = accounts.user_db(uid)
    db.add_application(u, "Acme", "Intern", contact_email="sara@acme.example", thread_id="t-acme",
                       applied_at="2026-09-10T10:00:00+00:00")
    u.close()
    return uid


def test_fast_rules_pass_then_ai_double_checks_uncertain_emails(home, monkeypatch):
    from app import llm, reply_classifier as rc
    monkeypatch.setattr(llm, "llm_available", lambda: True)
    calls = []
    monkeypatch.setattr(rc, "chat_json", lambda s, u, **k: calls.append(u) or
                        {"results": [{"i": 1, "label": "INTERVIEW_REQUEST", "confidence": 0.93, "reasoning": "let's talk"}]})
    c = accounts.connect()
    uid = _acme_user(c)
    c, jid, _ = _run_job(home, FakeGmail(_human_reply_inbox()), uid)
    u = accounts.user_db(uid)
    msg = u.execute("SELECT label, ai_pending FROM messages WHERE gmail_id='h1'").fetchone()
    status = u.execute("SELECT status FROM applications WHERE company='Acme'").fetchone()[0]
    u.close()
    assert sum("### EMAIL" in u for u in calls) == 1          # one classification call, only for the unsure email
    assert msg["label"] == "INTERVIEW_REQUEST" and msg["ai_pending"] == 0 and status == "INTERVIEWING"
    assert accounts.get_job(c, jid)["refine_done"] == 1


def test_when_ai_is_exhausted_the_rules_guess_stays_and_waits(home, monkeypatch):
    from app import llm, reply_classifier as rc
    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(rc, "chat_json", lambda *a, **k: None)
    monkeypatch.setattr(rc, "last_call_failed", lambda: True)
    c = accounts.connect()
    uid = _acme_user(c)
    c, jid, _ = _run_job(home, FakeGmail(_human_reply_inbox()), uid)
    u = accounts.user_db(uid)
    msg = u.execute("SELECT label, ai_pending FROM messages WHERE gmail_id='h1'").fetchone()
    u.close()
    job = accounts.get_job(c, jid)
    assert job["status"] == "done" and msg["label"] == "UNCLEAR" and msg["ai_pending"] == 1
    assert "still need an AI double-check" in job["message"]


def test_emails_are_downloaded_in_batches():
    class Batch:
        def __init__(self, svc, callback):
            self.svc, self.cb, self.reqs = svc, callback, []

        def add(self, req, request_id):
            self.reqs.append(request_id)

        def execute(self):
            self.svc.batches += 1
            for rid in self.reqs:
                self.cb(rid, self.svc.every[rid], None)

    class BatchGmail(FakeGmail):
        batches = 0

        @property
        def every(self):
            return {m["id"]: m for m in self.inbox}

        def new_batch_http_request(self, callback):
            return Batch(self, callback)

    svc = BatchGmail(_inbox(n_extra=38))                     # 40 emails
    got = worker._fetch(svc, [m["id"] for m in svc.inbox])
    assert len(got) == 40 and svc.batches == 2   # 2 round trips instead of 40