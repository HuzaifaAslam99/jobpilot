import json
import os
from pathlib import Path

import pytest

os.environ["JOBPILOT_OFFLINE"] = "1"
os.environ["JOBPILOT_RULES_ONLY"] = "1"

from app import db, scam_detector, reply_classifier  # noqa: E402
from app.pipeline import process_email  # noqa: E402
from app.reply_classifier import Email, classify  # noqa: E402
from app.scam_detector import analyze_posting  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
POSTINGS = {p["id"]: p for p in json.loads((DATA / "sample_postings.json").read_text())}
REPLIES = json.loads((DATA / "sample_replies.json").read_text())


def _post(pid, **kw):
    p = POSTINGS[pid]
    return analyze_posting(p["text"], p["company"], p["website"], p["contact_email"], **kw)


def _email(i):
    e = dict(REPLIES[i])
    return Email(headers=e.pop("headers", None), **e)


@pytest.mark.parametrize("pid,verdict", [
    ("northwind-backend", "looks_legit"),
    ("lumen-ml-intern", "looks_legit"),
    ("global-remote-data-entry", "likely_scam"),
])
def test_posting_verdicts(pid, verdict):
    assert _post(pid, use_llm=False).verdict == verdict


def test_pixelforge_is_at_least_suspicious():
    r = _post("pixelforge-junior-dev", use_llm=False)
    assert r.verdict in ("suspicious", "likely_scam")
    assert any("trial" in f.title.lower() for f in r.flags)


@pytest.mark.parametrize("text,title", [
    ("Hi, we saw your profile. Part-time online tasks: like YouTube videos and earn Rs 500 per task. "
     "To withdraw your earnings pay a small upgrade fee.", "Task scam"),
    ("Please share the 6-digit code you just received so we can verify your account.", "OTP"),
    ("Call centre jobs in Thailand, free air tickets and guaranteed visa, apply on WhatsApp.", "Overseas"),
])
def test_real_world_pk_scam_patterns(text, title):
    r = analyze_posting(text, use_llm=False, check_domain_age=False)
    assert any(title.lower() in f.title.lower() for f in r.flags)
    assert r.verdict != "looks_legit"


def test_every_flag_quotes_evidence():
    for f in _post("global-remote-data-entry", use_llm=False).flags:
        assert f.evidence and f.why


def test_llm_hallucinated_evidence_is_dropped(monkeypatch):
    fake = {"risk_score": 5, "summary": "fine", "green_flags": [],
            "red_flags": [{"category": "content", "severity": "high", "title": "Invented",
                           "evidence": "requires a Bitcoin wallet and a notarised letter", "why": "x"},
                          {"category": "process", "severity": "medium", "title": "Chat app",
                           "evidence": "All communication after the first email happens on WhatsApp", "why": "y"}]}
    monkeypatch.setattr(scam_detector, "chat_json", lambda *a, **k: fake)
    r = _post("pixelforge-junior-dev", use_llm=True)
    titles = {f.title for f in r.flags}
    assert "Invented" not in titles and r.mode == "rules+llm"


def test_critical_rule_floor_beats_lenient_llm(monkeypatch):
    monkeypatch.setattr(scam_detector, "chat_json",
                        lambda *a, **k: {"risk_score": 0, "red_flags": [], "green_flags": [], "summary": "ok"})
    assert _post("global-remote-data-entry", use_llm=True).risk_score >= 60


@pytest.mark.parametrize("i,label", [(0, "INTERVIEW_REQUEST"), (1, "REJECTED"), (2, "NEEDS_INFO"),
                                     (3, "AUTO_REPLY"), (4, "UNCLEAR")])
def test_reply_labels_rules_only(i, label):
    assert classify(_email(i), use_llm=False).label == label


def test_low_confidence_llm_becomes_unclear(monkeypatch):
    monkeypatch.setattr(reply_classifier, "chat_json",
                        lambda *a, **k: {"label": "INTERESTED", "confidence": 0.5, "reasoning": "maybe"})
    assert classify(_email(4)).label == "UNCLEAR"


def test_pipeline_end_to_end(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    from demo import seed
    seed(conn)
    results = [process_email(conn, _email(i), use_llm=False) for i in range(len(REPLIES))]
    by_company = {r["company"]: r for r in results}
    status = {r["company"]: r["status"] for r in db.pipeline(conn)}
    assert status["Northwind Analytics"] == "INTERVIEWING"
    assert status["Lumen Health AI"] == "REJECTED"
    assert status["Kestrel Systems"] == "APPLIED"            # auto-reply never changes status
    assert status["Quanta Retail"] == "NEEDS_REVIEW"         # scam reply overrides
    assert by_company["Quanta Retail"]["draft"] is None      # and never gets a draft
    assert by_company["Northwind Analytics"]["draft"] is not None
    # idempotent: second run processes nothing
    again = [process_email(conn, _email(i), use_llm=False) for i in range(len(REPLIES))]
    assert all(r.get("skipped") for r in again)


def test_bounce_is_detected_and_marks_application(tmp_path):
    conn = db.connect(tmp_path / "b.db")
    app_id = db.add_application(conn, "New Software", "Intern", "hr@newsoftware.example", thread_id="t-b")
    e = Email(gmail_id="b1", thread_id="t-b", from_addr="mailer-daemon@googlemail.com",
              from_name="Mail Delivery Subsystem", subject="Delivery Status Notification (Failure)",
              body="Address not found. Your message wasn't delivered to hr@newsoftware.example")
    res = process_email(conn, e, use_llm=False)
    row = conn.execute("SELECT status FROM applications WHERE id=?", (app_id,)).fetchone()
    assert res["label"] == "AUTO_REPLY" and row["status"] == "BOUNCED" and res["draft"] is None


def test_delay_notice_is_not_a_bounce():
    e = Email(gmail_id="d1", thread_id="t", from_addr="mailer-daemon@googlemail.com", from_name="Mail Delivery Subsystem",
              subject="Delivery Status Notification (Delay)", body="Delivery incomplete. Gmail will keep trying.")
    c = classify(e, use_llm=False)
    assert c.label == "AUTO_REPLY" and not c.key_details.get("bounce")


def test_phase_two_selection_is_positive():
    e = Email(gmail_id="p1", thread_id="t", from_addr="people@acme.example", from_name="People Team",
              subject="Selection for Phase 2 - Internship Program", body="Hi, please see the details below.")
    assert classify(e, use_llm=False).label == "INTERESTED"


@pytest.mark.parametrize("subject,role", [
    (" Summer Internship Application – Software Engineering (MERN + Web3)", "Summer Internship – Software Engineering (MERN + Web3)"),
    ("Re: Web3 Developer — Local Candidate, Open to Opportunities", "Web3 Developer"),
    ("Blockchain Developer Intern Application - Huzaifa", "Blockchain Developer Intern"),
    ("Application for Summer Internship 2026 – Full Stack Development", "Summer Internship 2026 – Full Stack Development"),
    ("Intern - Huzaifa", "Intern"),
])
def test_clean_role(subject, role):
    assert reply_classifier.clean_role(subject) == role


def test_stale_reply_gets_still_open_draft(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    db.add_application(conn, "BCDApps", "Web3 Developer", "info@bcdapps.example", thread_id="t-s")
    e = Email(gmail_id="s1", thread_id="t-s", from_addr="info@bcdapps.example", from_name="BCDApps",
              subject="Re: Web3 Developer", received_at="2026-07-02T10:00:00+05:00",
              body="Hi Huzaifa, can we schedule a call tomorrow? Please share your availability.")
    res = process_email(conn, e, use_llm=False)
    assert res["label"] == "INTERVIEW_REQUEST" and res["age_days"] > 14
    assert "still open" in res["draft"]["body"]
    row = conn.execute("SELECT next_action FROM applications").fetchone()
    assert "days old" in row["next_action"]


def _ago(days):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def test_followup_policy_by_kind_and_age(tmp_path):
    from app import followups
    conn = db.connect(tmp_path / "f.db")
    add = lambda name, kind, days, **kw: db.add_application(conn, name, "Web3 Developer", f"x@{name.lower()}.example",
                                                            thread_id=name, applied_at=_ago(days), subject="Web3 Developer",
                                                            kind=kind, **kw)
    add("Fresh", "application", 3)
    add("AppDue", "application", 10)
    add("AppOld", "application", 80)
    add("ColdDue", "cold", 9)
    add("ColdOld", "cold", 86)
    made = followups.create_drafts(conn, use_llm=False)
    st = {r["company"]: r for r in db.pipeline(conn)}
    assert sorted(m["company"] for m in made) == ["AppDue", "ColdDue"]
    assert all(r["priority"] != "high" for r in st.values())          # silence is never HIGH
    assert st["ColdOld"]["status"] == "NO_RESPONSE" and st["ColdOld"]["priority"] == "low"
    assert "LinkedIn" in st["ColdOld"]["next_action"]
    assert st["AppOld"]["status"] == "NO_RESPONSE" and st["AppOld"]["priority"] == "low"
    assert st["Fresh"]["status"] == "APPLIED" and "Wait" in st["Fresh"]["next_action"]
    cold_body = next(m["body"] for m in made if m["company"] == "ColdDue")
    assert "application" not in cold_body.lower()                       # a cold nudge isn't an 'application'
    d = conn.execute("SELECT * FROM drafts WHERE application_id=(SELECT id FROM applications WHERE company='AppDue')").fetchone()
    assert d["kind"] == "followup" and d["thread_id"] == "AppDue" and d["subject"] == "Re: Web3 Developer"
    assert followups.create_drafts(conn, use_llm=False) == []           # no duplicate drafts
    assert {m["company"] for m in followups.create_drafts(conn, use_llm=False, include_old=True)} == {"AppOld", "ColdOld"}


def test_reply_you_already_answered_is_waiting_on_them(tmp_path):
    conn = db.connect(tmp_path / "w.db")
    db.add_application(conn, "BCDApps", "Web3 Developer", "info@bcdapps.example", thread_id="t-w",
                       applied_at=_ago(86), kind="cold")
    e = Email(gmail_id="w1", thread_id="t-w", from_addr="info@bcdapps.example", from_name="BCDApps",
              subject="Re: Web3 Developer", received_at=_ago(85), body="Can I call you tomorrow?",
              you_replied_at=_ago(84))
    res = process_email(conn, e, use_llm=False)
    row = conn.execute("SELECT * FROM applications").fetchone()
    assert res["label"] == "INTERVIEW_REQUEST" and res["answered"] and res["draft"] is None
    assert row["status"] == "WAITING_ON_THEM" and row["priority"] == "low" and "went quiet" in row["next_action"]


@pytest.mark.parametrize("subject,snippet,kind", [
    ("Web3 Developer — Local Candidate, Open to Opportunities", "Hi Zindex Team, I'm Huzaifa, a Final Year student based in Islamabad", "cold"),
    ("Application for Summer Internship 2026", "Hi Dexterz Technologies Team, I'm writing to apply", "application"),
    ("Intern - Huzaifa", "I am writing to express my strong interest in the Internship position at New Software", "application"),
    # the tricky one: says 'apply' but there is no advertised role
    ("Web3 Developer – Open to Opportunities at Zindex",
     "Hi Zindex Team, I'm Huzaifa, a final-year Software Engineering student in Islamabad. I couldn't find any open "
     "roles on your site, but I'd love to apply if anything comes up for a Web3 or full-stack developer.", "cold"),
    ("Hello", "I'd love to apply for your 6-week Summer Internship Program 2026.", "application"),
    # a real advertised role, applied to late, even with a cold-sounding subject
    ("Web3 Developer – Open to Opportunities at Zindex",
     "Hi Zindex Team, I'm Huzaifa, a final-year Software Engineering student in Islamabad. I find a open role on your "
     "site, but i was late to apply. Its been a week but i still hoping for an opportunity from you", "application"),
    ("Web3 Developer", "Hi team, I came across the Web3 Developer opening on your careers page and wanted to reach out.",
     "application"),
    ("Hello", "I didn't see a matching role on your careers page, but I'd love to apply if one opens.", "cold"),
    ("Quick question", "Not sure if you're hiring, but I'd love to apply for an internship with your team.", "cold"),
    ("Application for Summer Internship 2026 – Full Stack Development",
     "Hi Dexterz Technologies Team, I'm writing to apply for the Full Stack Development track", "application"),
])
def test_guess_kind(subject, snippet, kind):
    from app.naming import guess_kind
    assert guess_kind(subject, snippet) == kind


def test_migration_adds_columns_to_old_db(tmp_path):
    import sqlite3
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.executescript(db.SCHEMA)  # the original schema, before the new columns existed
    c.execute("INSERT INTO applications(company, role, status) VALUES ('Keep Me', 'Intern', 'APPLIED')")
    c.commit()
    c.close()
    conn = db.connect(p)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(applications)")}
    assert {"subject", "last_contact_at", "followups_sent"} <= cols
    assert conn.execute("SELECT company FROM applications").fetchone()["company"] == "Keep Me"  # data kept


@pytest.mark.parametrize("snippet,name", [
    ("Hi Dexterz Technologies Team, I'm Huzaifa Aslam, a 3rd-year", "Dexterz Technologies"),
    ("Hi Vizz Web Solutions, I'm Huzaifa, a Software Engineering student", "Vizz Web Solutions"),
    ("Hi Zindex Team, I'm Huzaifa", "Zindex"),
    ("Dear Hiring Manager, I am writing to", None),
    ("Hi Zindex Team I&#39;m Huzaifa, a final-year student", "Zindex"),
    ("Hi Zindex Team, I&#39;m Huzaifa", "Zindex"),
    ("I am writing to express my interest", None),
])
def test_company_from_greeting(snippet, name):
    from app.naming import company_from_greeting
    assert company_from_greeting(snippet) == name


def test_free_mail_company_is_never_named(tmp_path):
    from app import followups
    from app.naming import domain_company, unfilled_placeholders
    assert domain_company("someone@gmail.com") == "Unknown"
    conn = db.connect(tmp_path / "g.db")
    db.add_application(conn, "Gmail", "Blockchain Developer", "someone@gmail.com", thread_id="t9",
                       applied_at=_ago(10), subject="Blockchain Developer Application")
    conn.execute("UPDATE applications SET company='Unknown'")
    body = followups.create_drafts(conn, use_llm=False)[0]["body"]
    assert "Gmail" not in body and "Unknown" not in body and body.startswith("Hi there,")
    assert unfilled_placeholders(body) == []
    assert unfilled_placeholders("I'm free [your availability].") == ["[your availability]"]


def _li(gid, subject, body, frm="jobs-noreply@linkedin.com", name="LinkedIn", days=20, tid=None):
    return Email(gmail_id=gid, thread_id=tid or gid, from_addr=frm, from_name=name, subject=subject,
                 body=body, received_at=_ago(days))


def test_inbox_scan_linkedin_and_portal_emails(tmp_path):
    from app.inbox_scan import ingest
    conn = db.connect(tmp_path / "i.db")
    r1 = ingest(conn, _li("m1", "Huzaifa, your application was sent to Acme Corp",
                          "Your application was sent to Acme Corp\nSoftware Engineer Intern · Islamabad"), use_llm=False)
    r2 = ingest(conn, _li("m2", "Huzaifa, your application was sent to Beta Labs",
                          "Your application was sent to Beta Labs\nFrontend Intern · Remote", days=18), use_llm=False)
    r3 = ingest(conn, _li("m3", "Software Engineer at Gamma is hiring", "Jobs you may be interested in: ...",
                          frm="jobalerts-noreply@linkedin.com", days=15), use_llm=False)
    r4 = ingest(conn, _li("m4", "Your application to Acme Corp",
                          "Thank you for applying to Acme Corp. Unfortunately, we have decided to move forward with "
                          "other candidates for the Software Engineer Intern position.",
                          frm="no-reply@us.greenhouse-mail.io", name="Acme Corp Recruiting", days=5), use_llm=False)
    rows = {r["company"]: r for r in db.pipeline(conn)}
    assert set(rows) == {"Acme Corp", "Beta Labs"}                     # alert ignored, LinkedIn not lumped together
    assert r1["created"] and r2["created"] and r3["skipped"] and not r4["created"]
    assert r1["label"] == "AUTO_REPLY" and r4["label"] == "REJECTED"   # 'thank you for applying...unfortunately'
    assert rows["Acme Corp"]["status"] == "REJECTED" and rows["Acme Corp"]["contact_email"] is None
    assert r4["draft"] is None and r1["draft"] is None                 # nobody to reply to at a no-reply address
    assert "Application confirmed" in rows["Beta Labs"]["next_action"]
    from app import followups
    assert followups.create_drafts(conn, use_llm=False) == []          # no follow-ups to no-reply platforms
    b = [r for r in db.pipeline(conn) if r["company"] == "Beta Labs"][0]
    assert "No direct contact" in b["next_action"] and "linkedin.com" in b["next_action"]
    again = ingest(conn, _li("m1", "x", "y"), use_llm=False)
    assert again.get("skipped") == "already processed"


def test_thank_you_for_applying_alone_is_acknowledgement():
    e = Email(gmail_id="a", thread_id="a", from_addr="careers@x.example", from_name="X",
              subject="Thank you for applying", body="Thank you for applying to X. Our team will review your profile.")
    assert classify(e, use_llm=False).label == "AUTO_REPLY"


def test_gmail_rate_limit_is_retried(monkeypatch):
    import time
    import httplib2
    from googleapiclient.errors import HttpError
    from app import gmail_client
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Req:
        def execute(self):
            calls["n"] += 1
            if calls["n"] < 3:
                resp = httplib2.Response({"status": 403})
                raise HttpError(resp, b'{"error": {"errors": [{"reason": "rateLimitExceeded"}], '
                                      b'"message": "Quota exceeded for quota metric"}}')
            return {"ok": True}

    assert gmail_client._x(Req()) == {"ok": True} and calls["n"] == 3


def _fresh_llm(monkeypatch):
    import app.llm as L
    L._state.update({"model": None, "dead": set(), "disabled": False, "no_json_mode": set(),
                     "cooldown_until": 0.0, "last_failed": False, "model_cooldown": {}})
    monkeypatch.setattr(L, "_client", lambda *a: object())
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    return L


class _Resp:
    def __init__(self, c):
        self.choices = [type("C", (), {"message": type("M", (), {"content": c})()})()]


def test_llm_waits_out_per_minute_limit(monkeypatch):
    L = _fresh_llm(monkeypatch)
    n = {"c": 0}
    def call(*a):
        n["c"] += 1
        if n["c"] == 1:
            raise Exception("Error code: 429 - Rate limit reached for model. Please try again in 7.5s.")
        return _Resp('{"ok": 1}')
    monkeypatch.setattr(L, "_call", call)
    assert L.chat_json("s", "u") == {"ok": 1} and not L.last_call_failed()


def test_llm_daily_limit_cools_down_without_hammering(monkeypatch):
    L = _fresh_llm(monkeypatch)
    n = {"c": 0}
    def call(*a):
        n["c"] += 1
        raise Exception("Error code: 429 - Rate limit reached ... tokens per day (TPD). Please try again in 21m30s.")
    monkeypatch.setattr(L, "_call", call)
    assert L.chat_json("s", "u") is None and L.last_call_failed()
    tried = n["c"]                                            # each model tried once (limits are per model)
    assert tried == len(L._candidates())
    assert L.chat_json("s", "u") is None and n["c"] == tried  # second call doesn't hit the API at all


def test_unsure_email_is_deferred_not_ignored_when_ai_is_limited(tmp_path, monkeypatch):
    from app import inbox_scan
    monkeypatch.setattr(inbox_scan, "chat_json", lambda *a, **k: None)
    monkeypatch.setattr(inbox_scan, "last_call_failed", lambda: True)
    conn = db.connect(tmp_path / "d.db")
    e = _li("d1", "Update from the Delta team", "Hi Huzaifa, we wanted to share an update with you.",
            frm="talent@delta.example", name="Delta Talent")
    res = inbox_scan.ingest(conn, e, use_llm=True)
    assert res["skipped"].startswith("deferred")
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0   # retried on the next scan


def test_gpt_oss_calls_use_low_reasoning_effort(monkeypatch):
    import app.llm as L
    seen = {}
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    return _Resp('{"ok": 1}')
    L._state.setdefault("no_effort", set())
    L._call(Client, "openai/gpt-oss-120b", "s", "u", 0.1)
    assert seen["reasoning_effort"] == "low" and seen["max_tokens"] <= 1000


def test_platform_legal_name_is_not_the_employer():
    from app.inbox_scan import _clean_company
    assert _clean_company("Indeed Ireland Operations Limited") is None
    assert _clean_company("Clever Solutions") == "Clever Solutions"      # whole words only


def test_greeting_for_hr_mailbox_is_team():
    from app.reply_classifier import Classification, draft_reply
    e = Email(gmail_id="g", thread_id="g", from_addr="hr@menzies.example", from_name="Human Resources",
              subject="Update", body="We regret to inform you...", received_at=_ago(2))
    d = draft_reply(e, Classification("REJECTED", 0.9, "", {}, "rules"), "Huzaifa", "MENZIES-RAS", "IT Intern", use_llm=False)
    assert d["body"].startswith("Hi MENZIES-RAS team,")


def test_newer_email_supersedes_old_drafts_and_link_actions_get_no_draft(tmp_path):
    conn = db.connect(tmp_path / "s2.db")
    db.add_application(conn, "micro1", "Fullstack Developer", "talent@micro1.example", thread_id="t-m",
                       applied_at=_ago(20))
    e1 = Email(gmail_id="m1", thread_id="t-m", from_addr="talent@micro1.example", from_name="Ali",
               subject="Next steps", received_at=_ago(10),
               body="Thanks for applying! Could you share your availability for a quick call this week?")
    r1 = process_email(conn, e1, use_llm=False)
    assert r1["draft"] is not None
    e2 = Email(gmail_id="m2", thread_id="t-m", from_addr="talent@micro1.example", from_name="Ali",
               subject="Complete your AI interview", received_at=_ago(8),
               body="Please complete the AI interview using the link below: https://micro1.example/interview")
    r2 = process_email(conn, e2, use_llm=False)
    assert r2["draft"] is None
    pending = conn.execute("SELECT COUNT(*) FROM drafts WHERE status='PENDING_REVIEW'").fetchone()[0]
    assert pending == 0                                                    # the old call-draft is superseded
    row = conn.execute("SELECT next_action FROM applications").fetchone()
    assert "link" in row["next_action"]


def test_past_calendar_notice_is_not_a_new_interview_request(tmp_path):
    conn = db.connect(tmp_path / "c.db")
    db.add_application(conn, "TechHavenLabs", "Intern", "laron@techhaven.example", thread_id="t-c", applied_at=_ago(100))
    e = Email(gmail_id="c1", thread_id="t-cal", from_addr="laron@techhaven.example", from_name="Prince Jerald Laron",
              subject="Re: Accepted: Huzaifa and Prince Jerald Laron @ Wed Jun 24, 2026 10:30pm - 11pm (GMT+8)",
              received_at=_ago(94), body="See you tomorrow! Looking forward to the interview.")
    res = process_email(conn, e, use_llm=False)
    row = conn.execute("SELECT * FROM applications").fetchone()
    assert res["draft"] is None and row["status"] == "INTERVIEWING"
    assert "2026-06-24" in row["next_action"] and "set-status" in row["next_action"]


def test_unknown_employers_are_never_merged(tmp_path):
    from app.inbox_scan import find_by_company
    conn = db.connect(tmp_path / "u.db")
    db.add_application(conn, "Unknown", "Blockchain Developer", "x@gmail.com", thread_id="t-u")
    assert find_by_company(conn, "Unknown employer (via indeed.com)") is None


def test_deleted_gmail_thread_is_skipped_not_fatal():
    import httplib2
    from googleapiclient.errors import HttpError
    from app import gmail_client

    class Svc:
        def users(self):
            return self
        def threads(self):
            return self
        def get(self, **kw):
            class R:
                def execute(self):
                    raise HttpError(httplib2.Response({"status": 404}), b'{"error": {"message": "Not Found"}}')
            return R()

    assert gmail_client._thread(Svc(), "gone", format="minimal") is None


def test_if_not_selected_is_a_confirmation_not_a_rejection():
    e = Email(gmail_id="bn", thread_id="bn", from_addr="no-reply@hire.lever.co", from_name="Binance",
              subject="Thank you for your application to Binance", received_at=_ago(110),
              body="Hi Huzaifa, Thank you for your interest in Binance! We have received your application for Binance "
                   "Accelerator Program - Blockchain / Smart Contract Security. Our team will review your application "
                   "and will be in touch if your qualifications match the needs of the role. If you are not selected "
                   "for this position, keep an eye on our jobs page as we're growing and adding openings.")
    assert classify(e, use_llm=False).label == "AUTO_REPLY"


def test_portal_email_you_can_reply_to_within_window(tmp_path):
    conn = db.connect(tmp_path / "bh.db")
    db.add_application(conn, "LantroTech", "Software Development Intern", None, thread_id="t-l", applied_at=_ago(20))
    body = ("##- Yes! You can reply directly to this email within 30 days and Amala will get it. -## Hi Huzaifa, "
            "we regret to inform you that we won't be pursuing your candidacy for the position.")
    fresh = Email(gmail_id="l1", thread_id="t-l", from_addr="notifications@app.bamboohr.com", from_name="Amala Mumtaz",
                  subject="Thank you for your interest in LantroTech", received_at=_ago(5), body=body)
    r = process_email(conn, fresh, use_llm=False)
    assert r["label"] == "REJECTED" and r["draft"] is not None            # replyable: thank-you draft allowed
    conn2 = db.connect(tmp_path / "bh2.db")
    db.add_application(conn2, "LantroTech", "Software Development Intern", None, thread_id="t-l", applied_at=_ago(120))
    old = Email(**{**fresh.__dict__, "gmail_id": "l2", "received_at": _ago(110)})
    r2 = process_email(conn2, old, use_llm=False)
    row = conn2.execute("SELECT next_action FROM applications").fetchone()
    assert r2["draft"] is None and "window has passed" in row["next_action"]


def test_same_company_different_roles_get_separate_rows(tmp_path):
    from app.inbox_scan import ingest
    conn = db.connect(tmp_path / "bn.db")
    roles = ["Binance Accelerator Program - Blockchain / Smart Contract Security",
             "Binance Accelerator Programm - Software Engineer Web3",
             "Frontend Engineer - Web3 Wallet",
             "Product Manager - Web3/ DeFi",
             "Web3 Security Specialist"]
    for i, r in enumerate(roles):
        e = Email(gmail_id=f"b{i}", thread_id="one-gmail-thread", from_addr="no-reply@hire.lever.co", from_name="Binance",
                  subject="Thank you for your application to Binance", received_at=_ago(110),
                  body=f"Hi Huzaifa, Thank you for your interest in Binance! We have received your application for {r}, "
                       "and we are delighted that you would consider joining our team.")
        ingest(conn, e, use_llm=False)
    rows = conn.execute("SELECT company, role FROM applications ORDER BY id").fetchall()
    assert len(rows) == 5 and all(r["company"] == "Binance" for r in rows)
    assert rows[2]["role"] == "Frontend Engineer - Web3 Wallet"
    # a later email about one of those roles lands on the right row, not a new one
    again = Email(gmail_id="b9", thread_id="one-gmail-thread", from_addr="no-reply@hire.lever.co", from_name="Binance",
                  subject="Update on your application", received_at=_ago(60),
                  body="Thank you for applying to Binance. Unfortunately, we will not be moving forward with your "
                       "application for Frontend Engineer - Web3 Wallet.")
    r = ingest(conn, again, use_llm=False)
    assert not r["created"]
    st = {x["role"]: x["status"] for x in db.pipeline(conn)}
    assert st["Frontend Engineer - Web3 Wallet"] == "REJECTED" and st["Web3 Security Specialist"] == "APPLIED"


REAL_MAY_EMAILS = [
    # (gmail_id, days_ago, from_addr, from_name, subject, body)
    ("pda1", 134, "jobs-noreply@linkedin.com", "LinkedIn", "Huzaifa, your application was sent to Pakistan Digital Authority",
     "Your application was sent to Pakistan Digital Authority\nSummer Intern\nPakistan Digital Authority · Islamabad, "
     "Pakistan (On-site)\nApplied on May 15, 2026"),
    ("gate1", 131, "jobs-noreply@linkedin.com", "LinkedIn", "Your application to Web3 Product Intern at Gate",
     "Your update from Gate\nWeb3 Product Intern\nGate · APAC\nApplied on May 15\nThank you for your interest in the "
     "Web3 Product Intern position at Gate in APAC. Unfortunately, we will not be moving forward with your application, "
     "but we appreciate your time and interest in Gate."),
    ("bybit1", 131, "jobs-noreply@linkedin.com", "LinkedIn", "Your application to Blockchain Security Intern at Bybit",
     "Your update from Bybit\nBlockchain Security Intern\nBybit · APAC\nThank you for your interest in the Blockchain "
     "Security Intern position at Bybit in APAC. Unfortunately, we will not be moving forward with your application."),
    ("pda2", 131, "jobs-noreply@linkedin.com", "LinkedIn", "Your application to Summer Intern at Pakistan Digital Authority",
     "Your update from Pakistan Digital Authority\nSummer Intern\nThank you for your interest in the Summer Intern "
     "position at Pakistan Digital Authority in Islamabad, Pakistan. Unfortunately, we will not be moving forward with "
     "your application, but we appreciate your time and interest in Pakistan Digital Authority."),
    ("nm1", 133, "no-reply@ashbyhq.com", "Nethermind Talent Team", "Information about your Nethermind application",
     "Hi Huzaifa\nThanks for applying to the General Application opportunity at Nethermind. Your application has been "
     "received and a member of our team will review it and get back to you as soon as possible."),
    ("nm2", 119, "no-reply@ashbyhq.com", "Nethermind Talent Team", "Information about your application - General Application",
     "Hi Huzaifa,\nThank you for taking the time to apply for the General Application\nUnfortunately, we won't be "
     "moving forward with your application at this time."),
]


def test_real_may_linkedin_and_ashby_emails(tmp_path):
    from app.inbox_scan import ingest
    conn = db.connect(tmp_path / "may.db")
    for gid, days, frm, name, subj, body in REAL_MAY_EMAILS:
        ingest(conn, Email(gmail_id=gid, thread_id=gid, from_addr=frm, from_name=name, subject=subj,
                           body=body, received_at=_ago(days)), use_llm=False)
    rows = {(r["company"], r["role"]): r["status"] for r in db.pipeline(conn)}
    assert rows == {
        ("Pakistan Digital Authority", "Summer Intern"): "REJECTED",   # confirmation + later rejection -> one row
        ("Gate", "Web3 Product Intern"): "REJECTED",
        ("Bybit", "Blockchain Security Intern"): "REJECTED",
        ("Nethermind", "General Application"): "REJECTED",
    }, rows


def test_bitmart_company_domain_interview_you_answered(tmp_path):
    from app.inbox_scan import ingest
    conn = db.connect(tmp_path / "bm.db")
    e = Email(gmail_id="bm1", thread_id="bm", from_addr="intern@bitmart.com", from_name="Intern",
              subject="BitMart Interview Availability Request", received_at=_ago(137), you_replied_at=_ago(134),
              body="Hope you're having a fantastic day. Thank you for your interest in the BitMart Global Campus Ignite "
                   "Program. We would like to advance you to the next stage. This will be a 15-20 mins online interview. "
                   "To help us schedule your next interview, please share your availabilities this week.")
    r = ingest(conn, e, use_llm=False)
    row = db.pipeline(conn)[0]
    assert r["created"] and row["company"] == "BitMart" and "Campus Ignite" in row["role"]
    assert r["label"] == "INTERVIEW_REQUEST" and row["status"] == "WAITING_ON_THEM" and r["draft"] is None


@pytest.mark.parametrize("frm,subject,body,expected_related", [
    ("donotreply@match.indeed.com", "React JS Internship @ EZILINE", "New job match: React JS Internship. Apply now.", False),
    ("donotreply@match.indeed.com", "Pashto Language Expert - Freelance AI Trainer Project @ Meridial",
     "We think you'd be a great fit. Apply with your Indeed profile.", False),
    ("no-reply@indeed.com", "You're almost there! Complete your application",
     "You started an application on Indeed. Finish it to be considered.", False),
    ("indeedapply@indeed.com", "Indeed Application: Intern Lead Blockchain Developer & Web3 Infrastructure Architect",
     "Your application has been submitted. Good luck!", None),   # a real application (employer unknown)
    ("jobs-noreply@linkedin.com", "Your application to Web3 Product Intern at Gate",
     "Your update from Gate. Unfortunately, we will not be moving forward with your application.", True),
])
def test_job_suggestions_are_not_applications(frm, subject, body, expected_related):
    from app.inbox_scan import rules_extract
    r = rules_extract(Email(gmail_id="x", thread_id="x", from_addr=frm, from_name="Indeed", subject=subject,
                            body=body, received_at=_ago(60)))
    if expected_related is False:
        assert r["related"] is False
    elif expected_related is True:
        assert r["related"] is True and r["company"] == "Gate"
    else:
        assert r["related"] is not False


def test_indeed_submitted_without_employer_is_tracked(tmp_path):
    from app.inbox_scan import ingest
    conn = db.connect(tmp_path / "ind.db")
    e = Email(gmail_id="in1", thread_id="in1", from_addr="indeedapply@indeed.com", from_name="Indeed",
              subject="Indeed Application: Intern Lead Blockchain Developer & Web3 Infrastructure Architect",
              body="Your application has been submitted. Good luck!", received_at=_ago(110))
    r = ingest(conn, e, use_llm=False)
    assert r["created"] and db.pipeline(conn)[0]["company"] == "Unknown employer (via indeed.com)"


def test_wrapped_plain_text_rejection_is_still_a_rejection():
    e = Email(gmail_id="w", thread_id="w", from_addr="jobs-noreply@linkedin.com", from_name="LinkedIn",
              subject="Your application to Web3 Product Intern at Gate", received_at=_ago(131),
              body="Your update from Gate\n\nThank you for your interest in the Web3 Product Intern position at Gate\n"
                   "in APAC. Unfortunately, we will not be moving\nforward with your application, but we appreciate\n"
                   "your time and interest in Gate.")
    assert classify(e, use_llm=False).label == "REJECTED"


@pytest.mark.parametrize("frm,name,subject,body", [
    ("mongodb@team.mongodb.com", "MongoDB", "Simplify your application stack",
     "Build faster: simplify your application stack with MongoDB Atlas. Start your free trial."),
    ("api@jamendo.com", "Jamendo API", "Application key has been created",
     "Your application key has been created. Client ID: abc123. Keep your client secret safe."),
    ("hr@skybugtechnologies.com", "Skybug HR", "Complete your Skybug Internship Application",
     "You started your application but haven't finished it yet."),
    ("hr@skybugtechnologies.com", "Skybug HR", "Your Skybug Internship Application is incomplete ⏳",
     "Please finish your application to be considered."),
])
def test_software_and_unfinished_emails_are_not_applications(frm, name, subject, body):
    from app.inbox_scan import rules_extract
    r = rules_extract(Email(gmail_id="s", thread_id="s", from_addr=frm, from_name=name, subject=subject, body=body,
                            received_at=_ago(130)))
    assert r["related"] is not True


def test_company_from_real_domain_not_subdomain():
    from app.naming import domain_company, org_label
    assert org_label("mail.internshala.com") == "internshala"
    assert org_label("team.mongodb.com") == "mongodb"
    assert org_label("careers.abc.com.pk") == "abc"
    assert domain_company("hr@careers.acme.io") == "Acme"


def test_teamtailor_confirmation_with_if_shortlisted_is_not_interest(tmp_path):
    from app.inbox_scan import ingest, is_platform
    assert is_platform("amama.ilyas@telenortsspakistan.teamtailor-mail.com")
    conn = db.connect(tmp_path / "tt.db")
    e = Email(gmail_id="tt", thread_id="tt", from_addr="amama.ilyas@telenortsspakistan.teamtailor-mail.com",
              from_name="Amama Ilyas", subject="We have received your application!", received_at=_ago(141),
              body="Hi Huzaifa, thank you for applying to Telenor. We have received your application for the "
                   "Graduate Trainee position. If you are shortlisted, our team will contact you for the next steps.")
    r = ingest(conn, e, use_llm=False)
    row = db.pipeline(conn)[0]
    assert r["label"] == "AUTO_REPLY" and r["draft"] is None and row["company"] == "Telenor"


def test_body_uses_html_when_plain_text_is_a_stub():
    import base64
    from app.gmail_client import _body
    enc = lambda t: base64.urlsafe_b64encode(t.encode()).decode()
    payload = {"mimeType": "multipart/alternative", "parts": [
        {"mimeType": "text/plain", "body": {"data": enc("Your update from Gate\nView job: https://lnkd.in/x")}},
        {"mimeType": "text/html", "body": {"data": enc(
            "<html><head><style>.a{color:red}</style></head><body><h1>Your update from Gate</h1>"
            "<p>Thank you for your interest in the Web3 Product Intern position at Gate in APAC. Unfortunately, "
            "we will not be moving forward with your application, but we appreciate your time &amp; interest.</p>"
            "</body></html>")}},
    ]}
    body = _body(payload)
    assert "not be moving forward" in body and "color:red" not in body and "&amp;" not in body
    e = Email(gmail_id="g", thread_id="g", from_addr="jobs-noreply@linkedin.com", from_name="LinkedIn",
              subject="Your application to Web3 Product Intern at Gate", body=body, received_at=_ago(131))
    assert classify(e, use_llm=False).label == "REJECTED"


def test_api_key_email_with_your_application_in_body_is_ignored():
    from app.inbox_scan import rules_extract
    r = rules_extract(Email(gmail_id="j", thread_id="j", from_addr="api@jamendo.com", from_name="Jamendo API",
                            subject="Application key has been created", received_at=_ago(132),
                            body="Hello, your application MyMusicApp has been registered. Your client id is abc."))
    assert r["related"] is False


# --- LinkedIn / Lever rejections that came back UNCLEAR on the real inbox ---
def _rej(subject, body, frm="jobs-noreply@linkedin.com"):
    return classify(Email(gmail_id="r", thread_id="r", from_addr=frm, from_name="LinkedIn", subject=subject,
                          body=body), use_llm=False)


def test_linkedin_update_notice_is_rejection_even_without_the_sentence():
    c = _rej("Your application to Web3 Product Intern at Gate", "Your update from Gate\nView job\nHelp Unsubscribe")
    assert c.label == "REJECTED"


def test_linkedin_confirmation_is_not_treated_as_rejection():
    c = _rej("Huzaifa, your application was sent to Gate", "Your application was sent to Gate\nWeb3 Product Intern")
    assert c.label == "AUTO_REPLY"


@pytest.mark.parametrize("sentence", [
    "We have decided to move forward with another candidate for this role.",
    "After careful review we will be going with other applicants whose experience more closely matches.",
    "We are not able to move forward with your application at this time.",
    "Your application was unsuccessful on this occasion.",
    "You have not been shortlisted for the next round.",
    "We've decided not to proceed with your candidacy.",
])
def test_more_rejection_wordings(sentence):
    assert _rej("Wintermute - Update on your application", sentence, frm="no-reply@hire.lever.co").label == "REJECTED"


def test_gmail_preview_used_when_body_misses_the_message():
    from app.gmail_client import with_snippet
    body = with_snippet("View job: https://linkedin.com/jobs/123",
                        "Your update from Bybit Thank you for your interest in the Intern position at Bybit. "
                        "Unfortunately, we will not be moving forward with your application")
    assert "not be moving forward" in body
    assert with_snippet("Hello there, thanks", "Hello there, thanks") == "Hello there, thanks"  # no duplicate


def test_rejection_for_one_role_with_interview_for_another_still_goes_to_human():
    c = _rej("Update", "We're not moving forward with the frontend role, but we'd like to schedule an interview "
                      "for the backend role.", frm="hr@x.example")
    assert c.label == "UNCLEAR"


def test_lever_wintermute_decision_to_not_move_forward():
    body = ("Hi Huzaifa,\n\nThank you for your application to Wintermute. After reviewing your work and experience, "
            "we've made the decision to not move forward at this time. Feel free to follow us on LinkedIn and apply "
            "again if you see interesting opportunities in the future.\n\nWe appreciate your interest in Wintermute "
            "and wish you success in your job search.")
    c = _rej("Thanks for your interest in Wintermute, Huzaifa", body, frm="no-reply@hire.lever.co")
    assert c.label == "REJECTED"


def test_no_thank_you_draft_for_an_old_rejection(tmp_path):
    from datetime import datetime, timedelta, timezone
    from app.reply_classifier import draft_reply
    old = (datetime.now(timezone.utc) - timedelta(days=115)).isoformat()
    fresh = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    body = "Unfortunately, we will not be moving forward with your application."
    for when, expect_draft in ((old, False), (fresh, True)):
        e = Email(gmail_id="z", thread_id="z", from_addr="hr@menzies.example", from_name="Human Resources",
                  subject="Update on your application", body=body, received_at=when)
        cls = classify(e, use_llm=False)
        assert cls.label == "REJECTED"
        d = draft_reply(e, cls, "Huzaifa", "MENZIES-RAS", "IT & IS Intern", use_llm=False)
        assert (d is not None) == expect_draft


def test_nethermind_real_contact_and_reapply_window():
    from app.reply_classifier import contact_address, reapply_window
    body = ("We would like to stay in touch with you because we see the potential to be successful. Feel free to "
            "reapply again in 3-6 months after having gone through the materials provided in this email.\n\n"
            "*Please include talent@nethermind.io in cc in all your replies.\n"
            "Click here to view our Privacy Policy. privacy@nethermind.io")
    assert contact_address(body) == "talent@nethermind.io"
    assert contact_address("Sent from no-reply@hire.lever.co. Web | LinkedIn") is None
    assert reapply_window(body, 132)["state"] == "open now"
    assert reapply_window(body, 30)["state"].startswith("opens in")


def test_hr_email_with_loose_role_joins_the_one_existing_application(tmp_path):
    from app.inbox_scan import find_by_company
    conn = db.connect(tmp_path / "m.db")
    a = db.add_application(conn, "MENZIES-RAS Pvt Ltd", "IT & IS Intern")
    assert find_by_company(conn, "Menzies-Ras", "Summer Internship 2026", loose=True)["id"] == a
    assert find_by_company(conn, "Menzies-Ras", "Summer Internship 2026")["id"] == a  # one application there
    db.add_application(conn, "Menzies-Ras", "Data Intern")
    assert find_by_company(conn, "Menzies-Ras", "Summer Internship 2026", loose=True) is None  # 2 rows: ambiguous


def test_unclear_while_ai_rate_limited_is_deferred_not_recorded(tmp_path, monkeypatch):
    from app import reply_classifier as rc
    monkeypatch.setattr(rc, "chat_json", lambda *a, **k: None)
    monkeypatch.setattr(rc, "last_call_failed", lambda: True)
    conn = db.connect(tmp_path / "d.db")
    db.add_application(conn, "Acme", "Intern", contact_email="hr@acme.example", thread_id="t-a")
    e = Email(gmail_id="u1", thread_id="t-a", from_addr="hr@acme.example", from_name="HR", subject="Hello",
              body="Just checking in about something.")
    res = process_email(conn, e, use_llm=True)
    assert res["skipped"].startswith("deferred")
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_llm_switches_model_when_one_hits_its_daily_limit(monkeypatch):
    from types import SimpleNamespace
    from app import llm
    monkeypatch.setenv("GROQ_API_KEY", "x"); monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("JOBPILOT_RULES_ONLY", raising=False); monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(llm, "_state", {"model": None, "dead": set(), "disabled": False, "no_json_mode": set(),
                                        "cooldown_until": 0.0, "last_failed": False, "model_cooldown": {}})
    monkeypatch.setattr(llm, "_client", lambda *a: object())
    calls = []

    def fake_call(client, model, *a):
        calls.append(model)
        if model == "openai/gpt-oss-120b":
            raise Exception("429 rate_limit_exceeded: tokens per day. Please try again in 20m5s")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"label": "REJECTED"}'))])
    monkeypatch.setattr(llm, "_call", fake_call)
    assert llm.chat_json("s", "u") == {"label": "REJECTED"}
    assert calls[0] == "openai/gpt-oss-120b" and calls[1] != "openai/gpt-oss-120b"
    calls.clear()
    llm.chat_json("s", "u")
    assert "openai/gpt-oss-120b" not in calls  # stays on the working model until the limit resets


def test_gemini_key_becomes_automatic_backup_for_groq(monkeypatch):
    from types import SimpleNamespace
    from app import llm
    monkeypatch.setenv("LLM_PROVIDER", "groq"); monkeypatch.setenv("GROQ_API_KEY", "g"); monkeypatch.setenv("GEMINI_API_KEY", "m")
    monkeypatch.delenv("LLM_MODEL", raising=False); monkeypatch.delenv("JOBPILOT_RULES_ONLY", raising=False)
    monkeypatch.setattr(llm, "_state", {"model": None, "dead": set(), "disabled": False, "no_json_mode": set(),
                                        "cooldown_until": 0.0, "last_failed": False, "model_cooldown": {}})
    monkeypatch.setattr(llm, "_client", lambda *a: object())
    cands = llm._candidates()
    assert cands[0].startswith("openai/") and any(m.startswith("gemini") for m in cands)
    used = []

    def fake_call(client, model, *a):
        used.append(model)
        if not model.startswith("gemini"):
            raise Exception("429 rate limit: tokens per day. Please try again in 30m")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))])
    monkeypatch.setattr(llm, "_call", fake_call)
    assert llm.chat_json("s", "u") == {"ok": 1} and used[-1].startswith("gemini")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert not any(m.startswith("gemini") for m in llm._candidates())  # no key -> no backup


def test_scan_period_presets_and_custom_range():
    from datetime import date, datetime
    from app.date_range import describe, gmail_query, resolve
    now = datetime(2026, 9, 26, 11, 0).astimezone()
    assert describe(*resolve("3d", True, now=now)) == "24 Sep 2026 – 26 Sep 2026 (3 days)"
    assert describe(*resolve("3d", False, now=now)) == "23 Sep 2026 – 25 Sep 2026 (3 days)"
    assert describe(*resolve("today", now=now)) == "26 Sep 2026"
    assert "365 days" in describe(*resolve("1y", now=now))
    frm, to = resolve(start=date(2026, 6, 1), end=date(2026, 6, 30), now=now)
    assert describe(frm, to) == "1 Jun 2026 – 30 Jun 2026 (30 days)"
    assert gmail_query(frm, to).startswith("after:") and " before:" in gmail_query(frm, to)
    with pytest.raises(ValueError):
        resolve(start=date(2025, 1, 1), end=date(2026, 6, 1), now=now)   # longer than 1 year
    with pytest.raises(ValueError):
        resolve(start=date(2026, 9, 1), end=date(2026, 10, 1), now=now)  # future


def test_thread_they_started_is_never_cold(tmp_path):
    from app import gmail_client
    conn = db.connect(tmp_path / "k.db")
    a = db.add_application(conn, "BitMart", "BitMart Global Campus Ignite", contact_email="intern@bitmart.com",
                           thread_id="t-bm", kind="cold")

    def msg(frm, date, snippet):
        return {"snippet": snippet, "payload": {"headers": [{"name": "From", "value": frm},
                {"name": "Date", "value": date}, {"name": "Subject", "value": "BitMart Interview Availability Request"}]}}

    class Svc:
        def users(self):
            return self
        def getProfile(self, userId):
            return self._r({"emailAddress": "me@gmail.com"})
        def threads(self):
            return self
        def get(self, **kw):
            return self._r({"messages": [
                msg("Intern <intern@bitmart.com>", "Tue, 12 May 2026 16:41:00 +0500", "We would like to advance you"),
                msg("Huzaifa <me@gmail.com>", "Fri, 15 May 2026 01:28:00 +0500",
                    "Thank you for the opportunity and for advancing me to the next stage!")]})
        def _r(self, v):
            return type("R", (), {"execute": lambda self_: v})()
    gmail_client.sync_thread_dates(Svc(), conn)
    assert conn.execute("SELECT outreach_kind FROM applications WHERE id=?", (a,)).fetchone()[0] == "application"


def test_older_email_found_later_never_undoes_a_newer_decision(tmp_path):
    conn = db.connect(tmp_path / "o.db")
    a = db.add_application(conn, "Menzies-Ras", "Summer Intern", contact_email="hr@menzies.example", thread_id="t-r")
    rejection = Email("r1", "t-r", "hr@menzies.example", "HR", "Update on your application",
                      "Unfortunately, we will not be moving forward with your application.",
                      received_at="2026-07-24T10:00:00+00:00")
    invite = Email("i1", "t-i", "hr@menzies.example", "HR", "Assessment invitation",
                   "We'd like to schedule an interview with you. Please share your availability.",
                   received_at="2026-06-12T10:00:00+00:00")
    process_email(conn, rejection, use_llm=False)              # 3-month scan
    res = process_email(conn, invite, use_llm=False)           # 6-month scan finds the OLDER invite afterwards
    row = conn.execute("SELECT status FROM applications WHERE id=?", (a,)).fetchone()
    assert res.get("superseded") and row["status"] == "REJECTED"
    assert db.pipeline(conn)[0]["last_label"] == "REJECTED"   # board shows the newest email's label
    assert not db.pending_drafts(conn)                         # and no draft for the stale invite


def test_greeting_sentence_is_not_a_company_name():
    from app.naming import company_from_greeting
    assert company_from_greeting("Hello, pursuing a career with LantroTech has been my goal") is None
    assert company_from_greeting("Hi Dexterz Technologies Team, I'm Huzaifa") == "Dexterz Technologies"


def test_merge_two_rows_of_the_same_application(tmp_path):
    conn = db.connect(tmp_path / "mg.db")
    sent = db.add_application(conn, "MENZIES-RAS Pvt Ltd", "IT & IS Intern", contact_email="hr@menzies.example",
                              thread_id="t-sent", applied_at="2026-06-01T09:00:00+00:00")
    found = db.add_application(conn, "Menzies-Ras", "(role not stated)", thread_id="t-hr",
                               applied_at="2026-07-24T10:00:00+00:00")
    process_email(conn, Email("r", "t-hr", "noreply@portal.example", "HR", "Update",
                              "Unfortunately, we will not be moving forward.", received_at="2026-07-24T10:00:00+00:00"),
                  use_llm=False, force_app_id=found)
    db.merge_applications(conn, sent, found)
    rows = db.pipeline(conn)
    assert len(rows) == 1 and rows[0]["status"] == "REJECTED" and rows[0]["role"] == "IT & IS Intern"
    assert rows[0]["applied_at"].startswith("2026-06-01") and rows[0]["contact_email"] == "hr@menzies.example"


def test_repair_reopened_row_when_newest_email_is_a_rejection(tmp_path):
    conn = db.connect(tmp_path / "rp.db")
    a = db.add_application(conn, "Menzies-Ras", "Summer")
    for gid, label, when in (("r", "REJECTED", "2026-07-24T10:00:00+00:00"), ("i", "INTERVIEW_REQUEST", "2026-06-12T10:00:00+00:00")):
        conn.execute("INSERT INTO messages(gmail_id, application_id, label, received_at) VALUES (?,?,?,?)", (gid, a, label, when))
    conn.execute("UPDATE applications SET status='INTERVIEWING' WHERE id=?", (a,))   # the old bug's result
    b = db.add_application(conn, "Kept", "Role")
    conn.execute("INSERT INTO messages(gmail_id, application_id, label, received_at) VALUES ('k', ?, 'REJECTED', '2026-07-01')", (b,))
    conn.execute("UPDATE applications SET status='OFFER', next_action='Set by you: offer.' WHERE id=?", (b,))
    assert db.repair_statuses(conn) == 1
    st = {r["company"]: r["status"] for r in db.pipeline(conn)}
    assert st == {"Menzies-Ras": "REJECTED", "Kept": "OFFER"}   # your own setting is never overridden


def test_unanswered_message_older_than_window_is_closed(tmp_path):
    from datetime import datetime, timedelta, timezone
    from app.followups import refresh
    conn = db.connect(tmp_path / "ua.db")
    ago = lambda d: (datetime.now(timezone.utc) - timedelta(days=d)).isoformat()
    scam = db.add_application(conn, "eraviyasolutions", "internship program", applied_at=ago(120))
    fresh = db.add_application(conn, "Fresh Co", "Intern", applied_at=ago(20))
    mine = db.add_application(conn, "Mine Co", "Intern", applied_at=ago(200))
    for app_id, days, status, action in (
            (scam, 109, "NEEDS_REVIEW", "Possible scam in reply: Hiring runs over Telegram / WhatsApp. Don't pay or share ID."),
            (fresh, 10, "INFO_REQUESTED", "Send what they asked for."),
            (mine, 100, "INTERVIEWING", "Set by you: interviewing.")):
        conn.execute("INSERT INTO messages(gmail_id, application_id, label, received_at) VALUES (?,?,?,?)",
                     (f"m{app_id}", app_id, "NEEDS_INFO", ago(days)))
        conn.execute("UPDATE applications SET status=?, priority='high', next_action=? WHERE id=?", (status, action, app_id))
    refresh(conn)
    rows = {r["company"]: r for r in db.pipeline(conn)}
    s = rows["eraviyasolutions"]
    assert s["status"] == "CLOSED" and s["priority"] == "low"
    assert s["next_action"].startswith("They messaged you 109 days ago and you didn't reply. Treat as closed.")
    assert "Telegram" in s["next_action"]
    assert rows["Fresh Co"]["status"] == "INFO_REQUESTED"      # only 10 days: still waiting on you
    assert rows["Mine Co"]["status"] == "INTERVIEWING"         # your own setting is kept


def test_per_minute_limit_switches_model_instead_of_sleeping(monkeypatch):
    from types import SimpleNamespace
    from app import llm
    monkeypatch.setenv("GROQ_API_KEY", "g"); monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("JOBPILOT_RULES_ONLY", raising=False)
    monkeypatch.setattr(llm, "_state", {"model": None, "dead": set(), "disabled": False, "no_json_mode": set(),
                                        "cooldown_until": 0.0, "last_failed": False, "model_cooldown": {}})
    monkeypatch.setattr(llm, "_client", lambda *a: object())
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    used = []

    def fake_call(client, model, *a):
        used.append(model)
        if model == "openai/gpt-oss-120b":
            raise Exception("429 rate_limit_exceeded: tokens per minute. Please try again in 20s")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))])
    monkeypatch.setattr(llm, "_call", fake_call)
    assert llm.chat_json("s", "u") == {"ok": 1}
    assert slept == [] and used[0] == "openai/gpt-oss-120b" and used[1] != used[0]   # switched, didn't wait


def test_same_application_named_differently_by_hr_and_portal(tmp_path):
    from app.inbox_scan import find_by_company
    conn = db.connect(tmp_path / "mz.db")
    hr_row = db.add_application(conn, "Menzies-Ras", "Menzies-RAS Summer")
    conn.execute("UPDATE applications SET posting_url='via company email' WHERE id=?", (hr_row,))
    # RozeeGPT confirmation (portal): other spelling, specific role -> still the same application
    assert find_by_company(conn, "MENZIES-RAS Pvt Ltd", "IT & IS Intern")["id"] == hr_row
    # but two PORTAL confirmations with different roles stay two applications (Binance)
    b1 = db.add_application(conn, "Binance", "Frontend Engineer")
    conn.execute("UPDATE applications SET posting_url='via hire.lever.co' WHERE id=?", (b1,))
    assert find_by_company(conn, "Binance", "Web3 Security Specialist") is None