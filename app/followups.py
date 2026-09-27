"""Follow-ups for outreach that never got a reply.

Priorities mean something: HIGH = someone is waiting on YOU (fresh interview request, info request,
scam warning). Silence from them is never HIGH.

Cold outreach (you emailed unprompted):
  - wait FOLLOWUP_DAYS, then ONE short nudge if still within COLD_WINDOW_DAYS
  - after that: NO_RESPONSE, low priority, no draft. Chasing a 3-month-old cold email is spam.
Job applications (you applied to something they advertised):
  - follow up after FOLLOWUP_DAYS, up to MAX_FOLLOWUPS, while within APP_WINDOW_DAYS of applying
  - after that: NO_RESPONSE, low priority. Check the posting is still live before any check-in.
Waiting on them (they replied, you answered, they went quiet):
  - one nudge after FOLLOWUP_DAYS, closed (low) after 30 days.
Days count from YOUR last email in the thread, so a follow-up resets the clock.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from . import db
from .llm import chat_json
from .naming import unknown_company

FOLLOWUP_DAYS = int(os.getenv("FOLLOWUP_DAYS", "7"))
MAX_FOLLOWUPS = int(os.getenv("MAX_FOLLOWUPS", "2"))          # for job applications
COLD_WINDOW_DAYS = int(os.getenv("COLD_WINDOW_DAYS", "21"))   # one nudge for cold emails, only inside this window
APP_WINDOW_DAYS = int(os.getenv("APP_WINDOW_DAYS", "45"))     # applications older than this are probably closed
PORTFOLIO_URL = os.getenv("PORTFOLIO_URL", "")


def greeting_for(company: str | None) -> str:
    return "Hi there," if unknown_company(company) else f"Hi {company} team,"


def days_since(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        d = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - d).days)


def _no_reply_apps(conn):
    return conn.execute("SELECT * FROM applications "
                        "WHERE status IN ('APPLIED', 'NO_RESPONSE', 'WAITING_ON_THEM')").fetchall()


def decide(a, include_old: bool = False) -> tuple[str, str, str, bool]:
    """-> (status, priority, next_action, follow_up_due) for one application with no reply pending from you."""
    age = days_since(a["last_contact_at"] or a["applied_at"]) or 0
    first = days_since(a["applied_at"]) or age
    sent = a["followups_sent"] or 0
    kind = a["outreach_kind"] or "application"
    status = a["status"]

    if status == "WAITING_ON_THEM":
        if age < FOLLOWUP_DAYS:
            return status, "normal", f"You replied {age} days ago. Give them a week.", False
        if age <= 30 or include_old:
            return status, "normal", f"They've been quiet for {age} days since your reply. One short nudge (draft ready).", True
        return status, "low", f"You replied {age} days ago and they went quiet. Treat as closed unless they come back.", False

    if not a["contact_email"]:  # applied via LinkedIn / a portal: no person to email
        via = a["posting_url"] or "a job portal"
        if first <= APP_WINDOW_DAYS:
            return "APPLIED", "normal", (f"Applied {first} days ago ({via}). No direct contact to follow up with; "
                                         "wait, or message a recruiter there on LinkedIn."), False
        return "NO_RESPONSE", "low", f"Applied {first} days ago ({via}) with no update. Probably closed.", False

    if age < FOLLOWUP_DAYS:
        return "APPLIED", "normal", f"Sent {age} days ago. Wait; follow up after {FOLLOWUP_DAYS} days.", False

    if kind == "cold":
        if (sent == 0 and first <= COLD_WINDOW_DAYS) or (include_old and sent == 0):
            return "APPLIED", "normal", f"Cold email, no reply in {age} days. One short nudge is fine (draft ready).", True
        return ("NO_RESPONSE", "low",
                f"Cold email, no reply in {first} days. Treat as closed; if you still want them, try LinkedIn "
                "or a different person rather than another email.", False)

    if (sent < MAX_FOLLOWUPS and first <= APP_WINDOW_DAYS) or (include_old and sent < MAX_FOLLOWUPS):
        which = "a polite follow-up" if sent == 0 else "one final follow-up"
        return "APPLIED", "normal", f"No reply in {age} days. Send {which} (draft ready).", True
    extra = f" and {sent} follow-up(s)" if sent else ""
    return ("NO_RESPONSE", "low",
            f"No reply in {first} days{extra}. Probably closed; check the posting is still live before any check-in.",
            False)


# statuses where THEY are waiting on YOU (they asked something and you haven't answered)
WAITING_ON_YOU = ("INTERVIEWING", "IN_CONVERSATION", "INFO_REQUESTED", "NEEDS_REVIEW")


def close_unanswered(conn) -> int:
    """They messaged you, you never replied, and it's been longer than APP_WINDOW_DAYS: the opportunity has
    almost certainly moved on, so say so plainly instead of keeping it on your to-do list."""
    n = 0
    for a in conn.execute(f"SELECT * FROM applications WHERE status IN {WAITING_ON_YOU}").fetchall():
        action = a["next_action"] or ""
        if action.startswith(("Set by you", "Interview/call")):
            continue  # your own setting, or a scheduled interview (calendar), not an unanswered message
        row = conn.execute("SELECT MAX(received_at) AS at FROM messages WHERE application_id=? "
                           "AND label != 'IGNORED'", (a["id"],)).fetchone()
        theirs = days_since(row["at"] if row else None)
        if theirs is None or theirs <= APP_WINDOW_DAYS:
            continue
        mine_after = a["last_contact_at"] and row["at"] and a["last_contact_at"] > row["at"]
        if mine_after:
            continue  # you did reply after their message
        note = ""
        if action.startswith("Possible scam"):
            note = " It also looked like a scam (" + action.split(":", 1)[-1].split(". Don't")[0].strip() + ")."
        conn.execute("UPDATE applications SET status='CLOSED', priority='low', next_action=? WHERE id=?",
                     (f"They messaged you {theirs} days ago and you didn't reply. Treat as closed.{note}", a["id"]))
        conn.execute("UPDATE drafts SET status='SUPERSEDED' WHERE application_id=? AND status='PENDING_REVIEW'",
                     (a["id"],))  # a reply draft to a months-old message isn't useful any more
        db.log_event(conn, a["id"], "closed_unanswered", {"days": theirs})
        n += 1
    conn.commit()
    return n


def refresh(conn, include_old: bool = False) -> list[dict]:
    """Recompute status / priority / next action for every no-reply item. Returns the ones due a follow-up."""
    close_unanswered(conn)
    due = []
    for a in _no_reply_apps(conn):
        if not (a["last_contact_at"] or a["applied_at"]):
            continue
        status, prio, action, is_due = decide(a, include_old)
        conn.execute("UPDATE applications SET status=?, priority=?, next_action=? WHERE id=?",
                     (status, prio, action, a["id"]))
        if is_due:
            age = days_since(a["last_contact_at"] or a["applied_at"]) or 0
            due.append({**dict(a), "age": age, "applied_days": days_since(a["applied_at"]) or age})
    conn.commit()
    return due


FOLLOWUP_SYSTEM = """You write short follow-up emails for a student job-seeker who got no reply.
The situation is given: a job application, a cold email (they never asked to hear from him, so be brief
and easy to ignore, 50 words max), or a conversation that went quiet after he replied.
Polite, confident, not needy. Never invent facts. Greet with exactly the greeting you are given
(never "[Hiring Manager's Name]"). Only mention a portfolio/GitHub link if one is provided; never add
bracketed placeholders. If no company name is given, don't name the company.
If the application is old (over 30 days), acknowledge time has passed and ask whether the role or any
upcoming openings are still available. 70 words max. Greeting, blank line, 1-2 short paragraphs,
blank line, sign-off. Return JSON only: {"body": "..."}"""


def _template(a, age, sent, name: str = "") -> str:
    CANDIDATE_NAME = name or os.getenv("CANDIDATE_NAME", "Huzaifa")
    hello = greeting_for(a["company"])
    at = "" if unknown_company(a["company"]) else f" at {a['company']}"
    if a["status"] == "WAITING_ON_THEM":
        return (f"{hello}\n\nJust following up on my last message about the {a['role']} role{at}. "
                f"Happy to find a time that suits you whenever convenient.\n\nBest regards,\n{CANDIDATE_NAME}")
    if (a["outreach_kind"] or "application") == "cold":
        return (f"{hello}\n\nBumping my note from last week in case it got buried. If you're open to a "
                f"{a['role']}, I'd be glad to share more.\n\nBest regards,\n{CANDIDATE_NAME}")
    if sent >= 1:
        return (f"{hello}\n\nA final quick follow-up on my application for the {a['role']} role{at}. "
                f"If it's filled, no problem at all; I'd appreciate being considered for future openings.\n\n"
                f"Best regards,\n{CANDIDATE_NAME}")
    if age > 30:
        return (f"{hello}\n\nI applied for the {a['role']} role{at} about {age // 7} weeks ago and wanted to check "
                f"whether it, or any upcoming internship openings, are still available. I'm still very interested "
                f"and happy to share anything useful.\n\nBest regards,\n{CANDIDATE_NAME}")
    return (f"{hello}\n\nI'm following up on my application for the {a['role']} role{at} sent last week. "
            f"I'm still very interested and would be glad to share more about my work.\n\nBest regards,\n{CANDIDATE_NAME}")


def create_drafts(conn, use_llm: bool = True, redo: bool = False, include_old: bool = False) -> list[dict]:
    """Create follow-up drafts (PENDING_REVIEW) for due applications that don't already have one.
    redo=True first discards pending follow-up drafts (e.g. after fixing company names)."""
    if redo:
        conn.execute("DELETE FROM drafts WHERE kind='followup' AND status='PENDING_REVIEW'")
    made = []
    CANDIDATE_NAME = db.candidate_name(conn)
    for a in refresh(conn, include_old):
        if not a["contact_email"]:
            continue
        exists = conn.execute("SELECT 1 FROM drafts WHERE application_id=? AND kind='followup' "
                              "AND status='PENDING_REVIEW'", (a["id"],)).fetchone()
        if exists:
            continue
        sent = a["followups_sent"] or 0
        body = None
        if use_llm:
            out = chat_json(FOLLOWUP_SYSTEM,
                            f"Candidate: {CANDIDATE_NAME}\nGreeting to use: {greeting_for(a['company'])}\n"
                            f"Company: {'(unknown, do not name it)' if unknown_company(a['company']) else a['company']}\n"
                            f"Role: {a['role']}\nPortfolio link: {PORTFOLIO_URL or '(none, do not mention)'}\n"
                            f"Original subject: {a['subject']}\nDays since applying: {a['applied_days']}\n"
                            f"Follow-ups already sent: {sent}\nSituation: "
                            f"{'conversation went quiet after his reply' if a['status'] == 'WAITING_ON_THEM' else (a['outreach_kind'] or 'application')}",
                            temperature=0.4)
            body = (out or {}).get("body")
        body = (body or _template(a, a["applied_days"], sent, CANDIDATE_NAME)).strip()
        subj = a["subject"] or f"Application: {a['role']}"
        subject = subj if subj.lower().startswith("re:") else f"Re: {subj}"
        conn.execute("""INSERT INTO drafts(application_id, message_id, to_addr, subject, body, created_at, thread_id, kind)
                        VALUES (?,?,?,?,?,?,?,'followup')""",
                     (a["id"], None, a["contact_email"], subject, body, db.now(), a["thread_id"]))
        db.log_event(conn, a["id"], "followup_drafted", {"days": a["age"], "followups_sent": sent})
        made.append({"company": a["company"], "role": a["role"], "age": a["applied_days"], "body": body})
    conn.commit()
    return made