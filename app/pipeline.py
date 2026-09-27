"""The agent loop for replies: match -> classify -> safety-check -> update -> draft.

Every step is logged to `events`, so the UI can show *why* the agent did what it did.
Nothing is ever sent: drafts wait in PENDING_REVIEW until a human approves them.
"""
from __future__ import annotations

import os
import re

from . import db
from .reply_classifier import STALE_DAYS, contact_address, reapply_window, reply_allowed, respond_via_link, Classification, Email, classify, draft_reply, reply_age_days
from .scam_detector import run_rules

DRAFT_REJECTION_ACK = os.getenv("DRAFT_REJECTION_ACK", "1") == "1"

# label -> (new status or None to keep, priority, next action)
TRANSITIONS = {
    "REJECTED": ("REJECTED", "normal", "Closed. Optional: send the thank-you draft."),
    "INTERVIEW_REQUEST": ("INTERVIEWING", "high", "Review the interview reply draft, add your availability, send."),
    "INTERESTED": ("IN_CONVERSATION", "high", "Review the reply draft and propose a call."),
    "NEEDS_INFO": ("INFO_REQUESTED", "high", "Send what they asked for (see draft)."),
    "AUTO_REPLY": (None, None, "Auto-reply only (contact away?). Follow up once they're back."),  # status unchanged
    "UNCLEAR": ("NEEDS_REVIEW", "high", "Read this reply yourself; the agent wasn't confident."),
}


def match_application(conn, email: Email):
    """Thread id first (exact), then sender address, then sender domain."""
    if email.thread_id:
        row = conn.execute("SELECT * FROM applications WHERE thread_id=?", (email.thread_id,)).fetchone()
        if row:
            return row, "thread"
    addr = email.from_addr.lower()
    from .inbox_scan import is_noreply, is_platform
    if is_platform(addr) or is_noreply(addr):
        return None, "unmatched"  # LinkedIn & co. send every application's email from one address: never match on it
    row = conn.execute("SELECT * FROM applications WHERE contact_email=? ORDER BY id DESC", (addr,)).fetchone()
    if row:
        return row, "sender"
    domain = addr.split("@")[-1]
    rows = conn.execute("SELECT * FROM applications WHERE contact_domain=?", (domain,)).fetchall()
    if len(rows) == 1:  # only trust a domain match when it's unambiguous
        return rows[0], "domain"
    return None, "unmatched"


def reply_safety_flags(email: Email):
    """Run the scam rules over the *reply* too: fake 'offers' often arrive as replies."""
    flags, _ = run_rules(email.body, company="(reply)", website="-", contact_email=email.from_addr,
                         check_domain_age=False)
    return [f for f in flags if f.severity in ("critical", "high") and f.category in ("payment", "process")]


CONFIRMATION_RE = re.compile(r"your application (was|has been) (sent|submitted|received|viewed)|we (have )?received your "
                             r"application|thank you for (applying|your application)|application (received|submitted)", re.I)


def _utc(iso: str | None):
    from datetime import datetime, timezone
    try:
        d = datetime.fromisoformat(iso)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _newer_message_exists(conn, app_id: int, received_at: str | None) -> bool:
    mine = _utc(received_at)
    if mine is None:
        return False
    for r in conn.execute("SELECT received_at FROM messages WHERE application_id=? AND label != 'IGNORED'", (app_id,)):
        other = _utc(r["received_at"])
        if other and other > mine:
            return True
    return False


def process_email(conn, email: Email, use_llm: bool = True, force_app_id: int | None = None,
                  cls: Classification | None = None) -> dict:
    """cls: a classification already made for this email (e.g. by a batched AI call); otherwise classify here."""
    if conn.execute("SELECT 1 FROM messages WHERE gmail_id=?", (email.gmail_id,)).fetchone():
        return {"gmail_id": email.gmail_id, "skipped": "already processed"}

    if force_app_id:
        app, method = conn.execute("SELECT * FROM applications WHERE id=?", (force_app_id,)).fetchone(), "company"
    else:
        app, method = match_application(conn, email)
    cls = cls or classify(email, use_llm=use_llm)
    if cls.key_details.get("ai_missing"):
        # don't record a guess: leave the email unprocessed so the next run classifies it with the AI
        return {"gmail_id": email.gmail_id, "skipped": "deferred (AI limit reached)", "subject": email.subject}
    danger = reply_safety_flags(email)
    # Scans can reach emails out of order (a 6-month scan after a 3-month one finds OLDER mail). An older email
    # must never undo a newer decision: e.g. an assessment invite from July can't reopen a September rejection.
    superseded = bool(app) and _newer_message_exists(conn, app["id"], email.received_at)

    msg_id = conn.execute(
        """INSERT INTO messages(gmail_id, thread_id, application_id, from_addr, subject, snippet, received_at,
               label, confidence, reasoning, match_method, processed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (email.gmail_id, email.thread_id, app["id"] if app else None, email.from_addr, email.subject,
         email.body[:280], email.received_at, cls.label, cls.confidence, cls.reasoning, method, db.now())).lastrowid

    result = {"gmail_id": email.gmail_id, "from": email.from_addr, "subject": email.subject,
              "label": cls.label, "confidence": round(cls.confidence, 2), "reasoning": cls.reasoning,
              "source": cls.source, "match": method, "application_id": app["id"] if app else None,
              "company": app["company"] if app else None, "draft": None, "safety_flags": [f.title for f in danger]}

    if not app:
        db.log_event(conn, None, "unmatched_reply", result)
        conn.commit()
        return result
    if superseded:
        result["superseded"] = True
        result["reasoning"] += " (Older than the latest email for this application: kept for history only.)"
        db.log_event(conn, app["id"], "older_email_recorded", result)
        conn.commit()
        return result

    status, priority, next_action = TRANSITIONS[cls.label]
    if cls.key_details.get("calendar_event"):
        when = cls.key_details["calendar_event"]
        status = "INTERVIEWING"
        if cls.key_details.get("event_past"):
            priority = "normal"
            next_action = (f"Interview/call was on {when}. Record how it went: "
                           f"python demo.py set-status {app['id']} INTERVIEWED | OFFER | REJECTED | CLOSED")
        else:
            priority = "high"
            next_action = f"Interview/call scheduled for {when}. Prepare, and check the invite for the link."
    from .inbox_scan import is_noreply, is_platform
    no_reply_sender = is_noreply(email.from_addr) or is_platform(email.from_addr)
    replyable, window_expired = reply_allowed(email.body[:1500], reply_age_days(email))
    if no_reply_sender and replyable and not window_expired:
        no_reply_sender = False  # e.g. BambooHR: "You can reply directly to this email within 30 days"
    if cls.label == "AUTO_REPLY" and CONFIRMATION_RE.search(f"{email.subject} {email.body[:800]}"):
        next_action = "Application confirmed. Nothing to do yet."
    via_link = cls.label in ("INTERVIEW_REQUEST", "NEEDS_INFO", "INTERESTED") and (
        str(cls.key_details.get("respond_via", "")).lower() == "link"
        or respond_via_link(f"{email.subject}\n{email.body[:3000]}"))
    if via_link:
        next_action = "Do it through the link in their email (assessment / AI interview / form); no email reply needed."
    if no_reply_sender and cls.label in ("INTERVIEW_REQUEST", "NEEDS_INFO", "INTERESTED"):
        next_action = "Sent from a no-reply address: open the email and use its link (scheduling / portal) to respond."
    if no_reply_sender and cls.label == "REJECTED":
        next_action = ("Closed. (Their email allowed replies for a limited time, and that window has passed.)"
                       if window_expired else "Closed (automated rejection, no one to reply to).")
    if cls.label == "REJECTED" and not no_reply_sender and (reply_age_days(email) or 0) > STALE_DAYS:
        next_action = "Closed. (Rejection is too old for a thank-you note; nothing to do.)"
    alt_contact = contact_address(email.body[:6000]) if no_reply_sender else None
    if alt_contact:
        # the no-reply email names a real address ("include talent@nethermind.io in cc"): that's who to write to
        conn.execute("UPDATE applications SET contact_email=COALESCE(contact_email, ?) WHERE id=?", (alt_contact, app["id"]))
        if cls.label == "REJECTED":
            next_action = f"Closed. The sender is no-reply, but they gave a real contact: {alt_contact}."
    reapply = reapply_window(email.body[:6000], reply_age_days(email)) if cls.label == "REJECTED" else None
    if reapply:
        next_action = (f"Rejected, but they invited you to reapply in {reapply['months']} months "
                       f"({reapply['state']})." + (f" Contact: {alt_contact}." if alt_contact else ""))
    if cls.key_details.get("bounce"):
        status, priority = "BOUNCED", "normal"  # worth fixing, but nobody is waiting on you
        next_action = "Your email bounced. Find a working contact address and resend."
    if danger:
        # a reply asking for fees / ID / a move to WhatsApp overrides whatever the classifier said
        status, priority = "NEEDS_REVIEW", "high"
        next_action = "Possible scam in reply: " + "; ".join(f.title for f in danger) + ". Don't pay or share ID."
    age = reply_age_days(email)
    result["age_days"] = age
    actionable = cls.label in ("INTERVIEW_REQUEST", "INTERESTED", "NEEDS_INFO", "UNCLEAR")
    answered = bool(email.you_replied_at) and actionable and not danger
    result["answered"] = answered
    if answered:
        # you already replied after this message: the ball is in their court, nothing to draft
        from .followups import days_since
        yd = days_since(email.you_replied_at) or 0
        status = "WAITING_ON_THEM"
        priority = "low" if yd > 30 else "normal"
        next_action = (f"You replied {yd} days ago and they went quiet. "
                       + ("Treat as closed unless they come back." if yd > 30 else "One short nudge after a week is fine."))
        conn.execute("UPDATE applications SET last_contact_at=MAX(COALESCE(last_contact_at,''), ?) WHERE id=?",
                     (email.you_replied_at, app["id"]))
    elif age is not None and age > STALE_DAYS and actionable and not danger:
        priority = "normal"  # nobody has been waiting on you for weeks; still worth a reply
        if cls.label == "UNCLEAR":
            next_action = ((f"The AI read this {age}-day-old email but it doesn't fit a clear outcome "
                            f"({cls.reasoning.rstrip('.')}). Read it yourself and set the status.")
                           if cls.source == "llm" else
                           (f"Couldn't classify this {age}-day-old email automatically. Read it yourself, or run the "
                            "scan again when the AI is available."))
        elif via_link:
            next_action = (f"Their request is {age} days old; the link or deadline has probably expired. "
                           "Open the email to check, and only follow up if it still matters to you.")
        else:
            next_action = (f"Their reply is {age} days old and unanswered; any dates in it have passed. "
                           "Send the draft asking if the role is still open.")
    if status:
        conn.execute("UPDATE applications SET status=?, priority=?, next_action=?, last_update=? WHERE id=?",
                     (status, priority, next_action, db.now(), app["id"]))
    elif app["status"] == "APPLIED":  # auto-reply: note it, but never overwrite a real decision
        conn.execute("UPDATE applications SET next_action=?, last_update=? WHERE id=?",
                     (next_action, db.now(), app["id"]))
    db.log_event(conn, app["id"], "reply_classified", result)

    if cls.label != "AUTO_REPLY" or cls.key_details.get("calendar_event"):
        # a newer message makes earlier unsent drafts for this application obsolete
        conn.execute("UPDATE drafts SET status='SUPERSEDED' WHERE application_id=? AND status='PENDING_REVIEW' "
                     "AND kind='reply'", (app["id"],))
    wants_draft = (not danger and not answered and not no_reply_sender and not via_link
                   and (cls.label != "REJECTED" or DRAFT_REJECTION_ACK))
    if wants_draft:
        d = draft_reply(email, cls, db.candidate_name(conn), app["company"], app["role"], use_llm=use_llm)
        if d:
            conn.execute("""INSERT INTO drafts(application_id, message_id, to_addr, subject, body, created_at, thread_id, kind)
                            VALUES (?,?,?,?,?,?,?,'reply')""",
                         (app["id"], msg_id, email.from_addr, d["subject"], d["body"], db.now(), email.thread_id))
            result["draft"] = d
            db.log_event(conn, app["id"], "draft_created", {"subject": d["subject"]})
    conn.commit()
    return result


def set_draft_status(conn, draft_id: int, status: str, body: str | None = None, gmail_draft_id: str | None = None):
    if body is not None:
        conn.execute("UPDATE drafts SET body=? WHERE id=?", (body, draft_id))
    conn.execute("UPDATE drafts SET status=?, gmail_draft_id=COALESCE(?, gmail_draft_id) WHERE id=?",
                 (status, gmail_draft_id, draft_id))
    row = conn.execute("SELECT application_id FROM drafts WHERE id=?", (draft_id,)).fetchone()
    db.log_event(conn, row["application_id"] if row else None, f"draft_{status.lower()}", {"draft_id": draft_id})
    conn.commit()