"""Gmail API access. Free, official, no scraping.

Setup (one time, ~5 min, no card):
  1. console.cloud.google.com -> new project -> enable "Gmail API".
  2. OAuth consent screen -> External -> add your own Gmail as a Test user.
  3. Credentials -> Create OAuth client ID -> Desktop app -> download as credentials.json
     into the project root.
  4. First run opens a browser to consent; token.json is cached afterwards.

Scopes: gmail.readonly to read replies, gmail.compose to save drafts.
The code never calls messages.send: a human sends from Gmail after reviewing.
"""
from __future__ import annotations

import base64
import re
from email.mime.text import MIMEText
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

from .naming import company_from_greeting, domain_company, guess_kind, looks_auto_named
from .reply_classifier import Email, clean_role

ROOT = Path(__file__).resolve().parent.parent
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.compose"]


def get_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    token, creds_file = ROOT / "token.json", ROOT / "credentials.json"
    creds = Credentials.from_authorized_user_file(token, SCOPES) if token.exists() else None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_file.exists():
                raise FileNotFoundError("credentials.json missing: see setup steps at the top of gmail_client.py")
            creds = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES).run_local_server(port=0)
        token.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _x(request):
    """Execute a Gmail API request, waiting and retrying when Gmail says 'slow down' (rate limit / quota)."""
    import time
    from googleapiclient.errors import HttpError
    for wait in (0, 10, 20, 40, 60):
        if wait:
            print(f"[gmail] rate limit reached; waiting {wait}s and retrying...")
            time.sleep(wait)
        try:
            return request.execute()
        except HttpError as exc:
            text = str(exc)
            if exc.resp.status == 429 or (exc.resp.status == 403 and ("ateLimitExceeded" in text or "Quota exceeded" in text)):
                continue
            raise
    return request.execute()


def _thread(service, tid, **kw):
    """threads.get that returns None if you deleted the thread in Gmail (instead of crashing)."""
    from googleapiclient.errors import HttpError
    try:
        return _x(service.users().threads().get(userId="me", id=tid, **kw))
    except HttpError as exc:
        if exc.resp.status in (404, 400):
            return None
        raise


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def html_to_text(raw: str) -> str:
    """Readable text from an HTML email: drop <style>/<script>, keep line breaks, decode &amp; etc."""
    import html as _html
    t = re.sub(r"(?is)<(style|script|head)[^>]*>.*?</\1>", " ", raw)
    t = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h[1-6]|table)>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = _html.unescape(t)
    t = re.sub(r"[ \t\u00a0\u200b\u200c\u034f]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def _body(payload) -> str:
    """Both versions of the email, and use the richer one. Some senders (LinkedIn, Lever) put the actual
    message ("Unfortunately, we will not be moving forward...") only in the HTML part, while the
    plain-text part is a short stub with links."""
    stack, plain, html_text = [payload], "", ""
    while stack:
        part = stack.pop()
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if mime == "text/plain" and data and not plain:
            plain = _decode(data)
        elif mime == "text/html" and data and not html_text:
            html_text = html_to_text(_decode(data))
        stack.extend(part.get("parts", []) or [])
    if html_text and len(" ".join(html_text.split())) > 1.3 * len(" ".join(plain.split())):
        return html_text
    return plain or html_text


def strip_quoted(text: str) -> str:
    """Drop the quoted original ("On Mon, X wrote:" / "> ..."), which would confuse the classifier."""
    cut = re.split(r"\n\s*On .{5,200}wrote:\s*\n|\n-{2,}\s*Original Message\s*-{2,}|\nFrom: .+\nSent: ", text, maxsplit=1)[0]
    return "\n".join(l for l in cut.splitlines() if not l.startswith(">")).strip()


def to_email(msg) -> Email:
    headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
    name, addr = parseaddr(headers.get("From", ""))
    try:
        from datetime import timezone
        received = parsedate_to_datetime(headers.get("Date", "")).astimezone(timezone.utc).isoformat()  # one clock
    except Exception:
        received = ""
    return Email(gmail_id=msg["id"], thread_id=msg["threadId"], from_addr=addr.lower(), from_name=name,
                 subject=headers.get("Subject", ""), body=with_snippet(strip_quoted(_body(msg["payload"])),
                                                                       msg.get("snippet")),
                 received_at=received, headers=headers)


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", t or "").lower().split())


def with_snippet(body: str, snippet: str | None) -> str:
    """Gmail's own preview line is built from the text you actually SEE in the email. Some senders (LinkedIn,
    Lever) hide the real message where our parser can't reach it (a big HTML part Gmail stores separately, or
    a plain-text stub full of links). If the preview isn't in what we extracted, put it first so the rules
    and the AI still see "Unfortunately, we will not be moving forward..."."""
    import html as _html
    snip = _html.unescape(snippet or "").strip()
    if not snip:
        return body
    probe = _norm(snip)[:60]
    if probe and probe not in _norm(body):
        return f"{snip}\n\n{body}".strip()
    return body


def fetch_replies(service, conn, days: int = 30, max_results: int = 50) -> list[Email]:
    """Inbox messages that belong to a tracked thread or come from a tracked contact."""
    me = _x(service.users().getProfile(userId="me"))["emailAddress"].lower()
    apps = conn.execute("SELECT thread_id, contact_email FROM applications").fetchall()
    threads = {a["thread_id"] for a in apps if a["thread_id"]}
    senders = [a["contact_email"] for a in apps if a["contact_email"]]

    ids = set()
    for tid in threads:  # replies inside threads we started
        t = _thread(service, tid, format="minimal")
        if t is None:
            continue  # deleted in Gmail
        ids.update(m["id"] for m in t.get("messages", []))
    if senders:  # replies that came as new threads (common with ATS systems)
        q = f"in:inbox newer_than:{days}d (" + " OR ".join(f"from:{s}" for s in senders[:40]) + ")"
        resp = _x(service.users().messages().list(userId="me", q=q, maxResults=max_results))
        ids.update(m["id"] for m in resp.get("messages", []))

    my_last = {}  # thread_id -> your latest message date in that thread
    _my_last = lambda tid: _thread_my_last(service, tid, me, my_last)
    emails = []
    for mid in ids:
        msg = _x(service.users().messages().get(userId="me", id=mid, format="full"))
        e = to_email(msg)
        if e.from_addr == me:  # skip our own sent messages in the thread
            continue
        mine = _my_last(e.thread_id)
        if mine and e.received_at and _as_dt(mine) > _as_dt(e.received_at):
            e.you_replied_at = mine  # you already answered this one
        emails.append(e)
    return sorted(emails, key=lambda e: e.received_at)


def _thread_my_last(service, tid, me, cache):
    if tid not in cache:
        t = _thread(service, tid, format="metadata", metadataHeaders=["From", "Date"]) or {}
        dates = []
        for m in t.get("messages", []):
            h = {x["name"]: x["value"] for x in m["payload"].get("headers", [])}
            if parseaddr(h.get("From", ""))[1].lower() == me:
                dates.append(_iso_date(h.get("Date")))
        cache[tid] = max([d for d in dates if d], default=None)
    return cache[tid]


def scan_inbox(service, conn, days: int = 365, max_results: int = 800, use_llm: bool = True,
               recheck_ignored: bool = False, only: str | None = None, window: str | None = None) -> list[dict]:
    """Applications made via LinkedIn / job portals, found from the emails those platforms send you."""
    if recheck_ignored:  # give previously ignored emails another look (e.g. after an AI rate limit)
        conn.execute("DELETE FROM messages WHERE label='IGNORED'")
        conn.commit()
    from .inbox_scan import DEFAULT_QUERY, ingest
    me = _x(service.users().getProfile(userId="me"))["emailAddress"].lower()
    ids, page = [], None
    while len(ids) < max_results:
        resp = _x(service.users().messages().list(userId="me", q=(f"({only}) -in:sent -in:spam -in:trash" if only else DEFAULT_QUERY)
                                               + " " + (window or f"newer_than:{days}d"),
                                               maxResults=min(100, max_results - len(ids)), pageToken=page))
        ids += [m["id"] for m in resp.get("messages", [])]
        page = resp.get("nextPageToken")
        if not page:
            break
    seen = {r["gmail_id"] for r in conn.execute("SELECT gmail_id FROM messages")}
    todo = [mid for mid in reversed(ids) if mid not in seen]  # Gmail lists newest first; process oldest first
    from .inbox_scan import is_noreply, is_platform
    import time
    results, cache = [], {}
    for i, mid in enumerate(todo, 1):
        e = to_email(_x(service.users().messages().get(userId="me", id=mid, format="full")))
        if e.from_addr == me:
            continue
        # only look up the whole thread when a real person wrote (you never reply to no-reply platforms)
        if not (is_noreply(e.from_addr) or is_platform(e.from_addr)):
            mine = _thread_my_last(service, e.thread_id, me, cache)
            if mine and e.received_at and _as_dt(mine) > _as_dt(e.received_at):
                e.you_replied_at = mine
        results.append(ingest(conn, e, use_llm=use_llm))  # saved immediately: a re-run resumes from here
        if i % 25 == 0:
            print(f"[scan] {i}/{len(todo)} emails checked...")
        time.sleep(0.15)  # stay well under Gmail's per-minute quota
    return results


def _as_dt(iso):
    from datetime import datetime, timezone
    d = datetime.fromisoformat(iso)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def import_sent_applications(service, conn, query: str, add_application) -> int:
    """Seed the pipeline from outreach you already sent, e.g. query='in:sent subject:(application OR internship)'."""
    found, page = [], None
    while len(found) < 1000:  # every page, not just the first 100 sent emails
        resp = _x(service.users().messages().list(userId="me", q=query, maxResults=100, pageToken=page))
        found += resp.get("messages", [])
        page = resp.get("nextPageToken")
        if not page:
            break
    known = {r["thread_id"] for r in conn.execute("SELECT thread_id FROM applications")}
    me = _x(service.users().getProfile(userId="me"))["emailAddress"].lower()
    n = 0
    for m in reversed(found):  # oldest first, so a first application isn't mistaken for a later one
        msg = _x(service.users().messages().get(userId="me", id=m["id"], format="metadata",
                                             metadataHeaders=["To", "Subject", "Date"]))
        if msg["threadId"] in known:
            continue
        if not _you_started(service, msg, me):
            # a REPLY inside someone else's thread (e.g. answering a newsletter or a recruiter), not an application
            # you started; recruiter threads are picked up from their emails by the inbox scan instead
            continue
        h = {x["name"]: x["value"] for x in msg["payload"]["headers"]}
        _, to = parseaddr(h.get("To", ""))
        from .inbox_scan import is_noreply
        if is_noreply(to):
            continue  # nobody applies to a no-reply address
        company = company_from_greeting(msg.get("snippet")) or domain_company(to)
        same = _existing_for_sent(conn, company, to)
        if same:
            # already on the board (e.g. found earlier from THEIR reply): add what this sent email tells us
            # instead of creating a second row for the same company
            sent_at = _iso_date(h.get("Date"))
            conn.execute("""UPDATE applications SET contact_email=COALESCE(contact_email, ?),
                                   thread_id=COALESCE(thread_id, ?), subject=COALESCE(subject, ?),
                                   applied_at=CASE WHEN ? < applied_at THEN ? ELSE applied_at END WHERE id=?""",
                         (to.lower() or None, msg["threadId"], h.get("Subject"), sent_at, sent_at, same["id"]))
            conn.commit()
            known.add(msg["threadId"])
            continue
        add_application(conn, company=company, role=clean_role(h.get("Subject", "(role?)"))[:80], contact_email=to,
                        thread_id=msg["threadId"], applied_at=_iso_date(h.get("Date")), subject=h.get("Subject"),
                        kind=guess_kind(h.get("Subject"), msg.get("snippet")))
        known.add(msg["threadId"])
        n += 1
    return n


def _you_started(service, msg, me: str) -> bool:
    """Did YOU write the first message of this email's thread? (Not the same as threadId == message id: Gmail
    gives the thread the id of the first DRAFT you saved, so for most emails you composed they differ.)"""
    t = _thread(service, msg["threadId"], format="metadata", metadataHeaders=["From"])
    if not t or not t.get("messages"):
        return True
    first = t["messages"][0]
    h = {x["name"]: x["value"] for x in first.get("payload", {}).get("headers", [])}
    return parseaddr(h.get("From", ""))[1].lower() == me


def _existing_for_sent(conn, company: str, to: str):
    """The board row this sent email belongs to, if the company is already there (same email domain, or the
    same company name when only one row has it)."""
    from .inbox_scan import is_platform, norm
    from .naming import FREE_MAIL, org_label
    domain = to.split("@")[-1].lower() if "@" in (to or "") else ""
    if domain and org_label(domain) not in FREE_MAIL and not is_platform(to):
        row = conn.execute("SELECT * FROM applications WHERE contact_domain=? ORDER BY id LIMIT 1", (domain,)).fetchone()
        if row:
            return row
    key = norm(company or "")
    if len(key) < 4 or (company or "").startswith("Unknown"):
        return None
    rows = [r for r in conn.execute("SELECT * FROM applications") if (k := norm(r["company"] or "")) and
            (k == key or (min(len(k), len(key)) >= 5 and (k in key or key in k)))]
    return rows[0] if len(rows) == 1 else None


def _iso_date(raw):
    try:
        from datetime import timezone
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def sync_thread_dates(service, conn) -> int:
    """For every tracked thread: real first-sent date, original subject, your latest message date,
    and how many follow-ups you've already sent. Fixes rows imported before dates were tracked."""
    me = _x(service.users().getProfile(userId="me"))["emailAddress"].lower()
    n = 0
    for app in conn.execute("SELECT id, thread_id, company, contact_email, outreach_kind, kind_locked FROM applications "
                            "WHERE thread_id IS NOT NULL").fetchall():
        if app["thread_id"].startswith("demo-"):
            continue
        t = _thread(service, app["thread_id"], format="metadata", metadataHeaders=["From", "Date", "Subject"])
        if t is None:
            continue  # thread deleted in Gmail; keep the row as it is
        mine = []
        subject = None
        first_snippet = None
        they_started = None
        for m in t.get("messages", []):
            h = {x["name"]: x["value"] for x in m["payload"].get("headers", [])}
            subject = subject or h.get("Subject")
            if they_started is None:
                they_started = parseaddr(h.get("From", ""))[1].lower() != me
            if parseaddr(h.get("From", ""))[1].lower() == me:
                mine.append(_iso_date(h.get("Date")))
                first_snippet = first_snippet or m.get("snippet")
        mine = [d for d in mine if d]
        if not mine:
            continue
        conn.execute("UPDATE applications SET applied_at=?, last_contact_at=?, followups_sent=?, subject=? WHERE id=?",
                     (min(mine), max(mine), len(mine) - 1, subject, app["id"]))
        if not app["kind_locked"]:  # re-guess with the current rules unless you set it yourself
            # if THEY wrote first (e.g. BitMart inviting you to interview), you're answering them, so it can't be a
            # cold email, whatever your reply says; only threads you started are guessed from your own words
            kind = "application" if they_started else guess_kind(subject, first_snippet)
            conn.execute("UPDATE applications SET outreach_kind=? WHERE id=?", (kind, app["id"]))
        # replace names we guessed from the domain with the name you used in your own greeting
        greeted = company_from_greeting(first_snippet)
        if looks_auto_named(app["company"], app["contact_email"]):
            better = greeted or domain_company(app["contact_email"])
            if better != app["company"]:
                conn.execute("UPDATE applications SET company=? WHERE id=?", (better, app["id"]))
        n += 1
    conn.commit()
    return n


def create_gmail_draft(service, to: str, subject: str, body: str, thread_id: str | None = None) -> str:
    """Save to Gmail Drafts (threaded). The human presses Send."""
    mime = MIMEText(body)
    mime["to"], mime["subject"] = to, subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    message = {"raw": raw}
    if thread_id:
        message["threadId"] = thread_id
    draft = _x(service.users().drafts().create(userId="me", body={"message": message}))
    return draft["id"]