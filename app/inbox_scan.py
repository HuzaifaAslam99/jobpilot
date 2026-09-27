"""Find applications you made OUTSIDE email (LinkedIn Easy Apply, Greenhouse, Lever, Workday, Rozee...)
from the emails those platforms send you, and add them to the pipeline.

Pure logic lives here (testable without Gmail); gmail_client.scan_inbox() feeds it real messages.

  1. Cheap rules decide: job-alert spam? -> ignore.  Clear application email? -> extract company/role.
  2. Unsure -> ask the LLM (is this about an application I made? which company? which role?).
  3. Match to an existing row by company name (never by sender: every LinkedIn email comes from the
     same no-reply address), otherwise create a new row, then run the normal reply pipeline on it.
"""
from __future__ import annotations

import re

from . import db
from .llm import chat_json, last_call_failed
from .reply_classifier import Email, clean_role

PLATFORM_DOMAINS = (
    "linkedin.com", "greenhouse.io", "greenhouse-mail.io", "lever.co", "hire.lever.co", "myworkday.com",
    "workday.com", "myworkdayjobs.com", "ashbyhq.com", "smartrecruiters.com", "bamboohr.com", "workable.com",
    "recruitee.com", "jobvite.com", "icims.com", "breezy.hr", "teamtailor.com", "zohorecruit.com",
    "rozee.pk", "mustakbil.com", "indeed.com", "indeedemail.com", "glassdoor.com", "wellfound.com",
    "angel.co", "dice.com", "bayt.com", "naukri.com", "hirist.com", "join.com", "personio.de",
    "internshala.com", "teamtailor-mail.com", "freshteam.com", "jazzhr.com", "applytojob.com", "manatal.com",
    "keka.com", "darwinbox.in", "successfactors.com", "taleo.net", "rippling-ats.com", "pinpointhq.com",
)

PLATFORM_BRANDS = ("internshala", "teamtailor", "linkedin", "indeed", "glassdoor", "greenhouse", "lever", "workday", "ashby", "smartrecruiters",
                   "bamboohr", "workable", "recruitee", "jobvite", "icims", "breezy", "teamtailor", "zohorecruit",
                   "rozee", "mustakbil", "wellfound", "angellist", "cryptojobs", "bayt", "naukri", "dice")

# Broad on purpose; the rules/LLM below throw out what isn't about you.
# All mail except Sent/Spam/Trash: LinkedIn & portal emails often sit in the Promotions/Social tabs or get archived.
# The rules below discard newsletters and job alerts without calling the AI, so the wider net is cheap.
DEFAULT_QUERY = ('-in:sent -in:spam -in:trash ('
                 'subject:(application OR applying OR applied OR "thank you for applying" OR "your application" '
                 'OR interview OR assessment OR "next steps" OR candidacy OR shortlisted OR "update from" '
                 'OR availability) OR from:(' + " OR ".join(PLATFORM_DOMAINS) + '))')

ALERT_RE = re.compile(r"\b(jobs? (you may|you might|for you|alert|recommendation)|is hiring|new jobs?|jobs? matching|"
                      r"recommended (jobs?|for you)|apply now|top job picks|people also viewed|"
                      r"who'?s viewed your profile|weekly digest|newsletter|webinar|course|api keys?|application keys?|"
                      r"client (?:id|secret)|developer (?:account|portal)|access tokens?|application stack|deploy\w*|"
                      r"free trial|pricing|invoice|receipt)\b", re.I)

# Some patterns also capture the role (?P<r>...). Most specific first.
COMPANY_PATTERNS = [
    r"your application to (?P<r>[^\n|]{2,70}?) at (?P<c>[^.!\n|]+?)\s*$",                        # LinkedIn subject
    r"your update from (?P<c>[^.!\n|]+)",                                                         # LinkedIn body
    r"your application was (?:sent|submitted) to (?P<c>[^.!\n|]+)",
    r"your application (?:was|has been) viewed by (?P<c>[^.!\n|]+)",
    r"interest in the (?P<r>[^.\n]{2,70}?) (?:position|role|internship|opening) at (?P<c>[^.!\n|,]+?)"
    r"(?: in [A-Z][\w ]{1,30})?[.,!]",
    r"(?:thanks|thank you) for applying to (?:the )?(?P<r>[^.\n]{2,70}?) (?:opportunity|role|position|job|opening) "
    r"at (?P<c>[^.!\n|,]+)",
    r"(?:thanks|thank you) for (?:applying|your application|your interest) (?:to|at|with|in(?: joining)?) "
    r"(?P<c>[^.!\n|,]+?)(?:\s[-–—|]\s|[.!\n,]|$)",
    r"(?:application|candidacy) (?:to|at|with) (?P<c>[^.!\n|]+?) (?:has been|was|is) (?:received|submitted|reviewed)",
    r"^(?P<c>[^|\n]{2,40}?)\s*[|:–—-]\s*(?:your )?(?:application|thank you|next steps|interview)",
    r"update (?:on|about|regarding) your application (?:to|at|with) (?P<c>[^.!\n|]+)",
    r"about your (?P<c>[A-Z][\w&.\-]{1,30}(?: [A-Z][\w&.\-]{1,30}){0,2}) application\b",          # "…your Nethermind application"
    r"(?:we|the team) at (?P<c>[A-Z][\w&.\- ]{1,40}?) (?:have|has|would|are|received)",
]
ROLE_PATTERNS = [
    r"your application to (?P<r>[^\n|]{2,70}?) at ",
    r"(?:interest in|applying to|apply for|application for) the (?P<r>[^.\n,]{3,70}?) (?:position|role|internship|"
    r"opening|opportunity|program|programme)\b",
    r"(?:for|to) the (?P<r>[^.\n,]{3,70}?) (?:role|position|opening|internship)\b",
    r"(?:application|applying) for (?:the )?(?P<r>[^.\n,]{3,70}?)(?: at | with |[.\n,]| role| position)",
    r"(?:role|position):\s*(?P<r>[^\n]{3,70})",
]
TEAM_WORDS_RE = re.compile(r"\s*\b(talent( acquisition)?( team)?|recruit\w*( team)?|careers?|hiring( team)?|people( team)?|"
                           r"hr( team)?|jobs|team)\s*$", re.I)
APPLICATION_RE = re.compile(r"\b(your (?:job |internship )?application(?!\s+(?:stack|keys?|id|secret|tokens?|credentials|"
                            r"settings|dashboard|performance|monitoring|logs?|servers?|code|deployment|has been created))|(?:thank you|thanks) for applying|application (?:was|has been) (?:sent|"
                            r"received|submitted|viewed)|we (?:have )?received your application|your candidacy|"
                            r"regret to inform|move forward with other|(?:not|n't) be moving forward|"
                            r"next steps in (?:the|our) (?:hiring|interview)|advance you to the next (?:stage|round)|"
                            r"interview availability|time to apply for|your update from)\b", re.I)


def is_platform(addr: str) -> bool:
    dom = (addr or "").split("@")[-1].lower()
    return any(dom == p or dom.endswith("." + p) for p in PLATFORM_DOMAINS)


def is_noreply(addr: str) -> bool:
    return bool(re.search(r"no-?reply|do-?not-?reply|notifications?@|mailer@|jobs-noreply", addr or "", re.I))


def _clean_company(c: str | None) -> str | None:
    if not c:
        return None
    c = re.sub(r"\s+", " ", c).strip(" .,:;!-–—|\"'")
    c = re.sub(r"^(the)\s+", "", c, flags=re.I)
    # "your interest in pursuing a career with LantroTech" -> LantroTech
    c = re.sub(r"^(?:pursuing |building |starting )?(?:a |your )?(?:career|future|internship|job|role|position)s?"
               r"(?: opportunit(?:y|ies))? (?:with|at|in)\s+|^(?:joining|working (?:at|with|for))\s+", "", c, flags=re.I)
    c = re.sub(r"\s+(team|careers|recruiting|talent( acquisition)?|hiring team|hr)$", "", c, flags=re.I)
    if not c or len(c) > 50 or len(c.split()) > 6 or c.lower() in {"us", "our company", "you", "linkedin"}:
        return None
    if set(re.findall(r"[a-z]+", c.lower())) & set(PLATFORM_BRANDS):  # "Indeed Ireland Operations Limited" is the platform, not the employer
        return None
    return c


def norm(company: str | None) -> str:
    c = (company or "").lower()
    c = re.sub(r"\b(inc|ltd|llc|pvt|private|limited|corp|corporation|co|technologies|technology|tech|"
               r"solutions|software|labs?|studios?|group|pk)\b\.?", "", c)
    return re.sub(r"[^a-z0-9]", "", c)


def _tidy_role(r: str) -> str:
    """A role taken from email text: just trim it (clean_role() is for subjects and would turn
    'General Application' into 'General')."""
    return re.sub(r"\s+", " ", r).strip(" .,:;-–—|\"'")[:80]


def _company_from_sender(email: Email) -> str | None:
    """Fallbacks when the text doesn't name the employer clearly."""
    addr = (email.from_addr or "").lower()
    from .naming import org_label
    dom_root = org_label(addr.split("@")[-1])
    text = f"{email.subject}\n{email.body[:1500]}"
    if not (is_platform(addr) or is_noreply(addr)) and dom_root and dom_root not in {"gmail", "yahoo", "outlook", "hotmail"}:
        # the company's own domain (intern@bitmart.com) is the most reliable signal; keep their spelling ("BitMart")
        m = re.search(rf"\b({re.escape(dom_root)})\b", text, re.I)
        return m.group(1) if m else dom_root.title()
    name = (email.from_name or "").strip()
    if name:
        # "Nethermind Talent Team" -> Nethermind; "Binance" (also named in the subject) -> Binance;
        # a person's name ("Amala Mumtaz") is never taken as the company
        stripped = TEAM_WORDS_RE.sub("", name).strip()
        if stripped and (stripped != name or re.search(rf"\b{re.escape(stripped)}\b", email.subject or "", re.I)):
            return _clean_company(stripped)
    return None


# Job SUGGESTIONS and unfinished applications look a lot like applications, but you never applied.
SUGGESTION_SENDER_RE = re.compile(r"@(?:match\.indeed\.com|jobalerts?[\w.-]*|jobs-listings@linkedin\.com|"
                                  r"alerts?\.[\w.-]+|jobmatch[\w.-]*)$|jobalerts-noreply@|jobs-listings@", re.I)
SUGGESTION_SUBJECT_RE = re.compile(r"(?:^|\s)@\s*[A-Z][\w&.\- ]{1,40}$|you'?re almost there|"
                                   r"(?:complete|finish) your\b[^\n]{0,40}\bapplication|application is incomplete|"
                                   r"incomplete application|still interested in applying|jobs? (?:picked|selected) for you|"
                                   r"based on your (?:profile|search)|similar jobs|apply (?:now|today|with)", re.I)
SUBMITTED_RE = re.compile(r"\b(your application (?:was|has been) (?:sent|submitted|received|viewed)|thank(?:s| you) for "
                          r"(?:applying|your application)|we (?:have )?received your application|your update from|"
                          r"regret to inform|(?:not|n't) be moving forward|move forward with other)\b", re.I)


def is_job_suggestion(email: Email) -> bool:
    """Indeed 'React JS Internship @ EZILINE', LinkedIn 'jobs for you', 'Complete your application'... -> not yours."""
    text = f"{email.subject}\n{email.body[:1500]}"
    if SUBMITTED_RE.search(text):
        return False  # a real confirmation / decision about something you submitted
    return bool(SUGGESTION_SENDER_RE.search(email.from_addr or "") or SUGGESTION_SUBJECT_RE.search(email.subject or ""))


def rules_extract(email: Email) -> dict:
    """-> {'related': True/False/None(unsure), 'company', 'role'}"""
    email = Email(**{**email.__dict__, "body": "\n".join(" ".join(l.split()) for l in email.body[:3000].splitlines())})
    text = f"{email.subject}\n{' '.join(email.body[:1500].split())}"
    # an alert-looking subject ("Application key has been created") wins unless the text clearly confirms
    # something you SUBMITTED; a bare "your application ..." in the body isn't enough
    if is_job_suggestion(email) or (ALERT_RE.search(email.subject or "") and not SUBMITTED_RE.search(text)):
        return {"related": False, "company": None, "role": None}
    company = role = None
    sender_company = _company_from_sender(email)
    if sender_company and not (is_platform(email.from_addr) or is_noreply(email.from_addr)):
        company = sender_company  # a company writing from its own domain
    for pat in COMPANY_PATTERNS:
        if company:
            break
        m = re.search(pat, email.subject or "", re.I | re.M) or re.search(pat, email.body[:1500], re.I | re.M)
        if m and (c := _clean_company(m.group("c"))):
            company = c
            if "r" in m.groupdict() and m.group("r"):
                role = _tidy_role(m.group("r"))
    company = company or sender_company
    if not role:
        for pat in ROLE_PATTERNS:
            m = re.search(pat, text, re.I)
            if m:
                role = _tidy_role(m.group("r"))
                break
    related = True if (APPLICATION_RE.search(text) and company) else None
    if related is None and SUBMITTED_RE.search(text) and (is_platform(email.from_addr) or is_noreply(email.from_addr)):
        related = True  # clearly about something you submitted; employer may be unnamed ("Unknown employer (via …)")
    return {"related": related, "company": company, "role": role}


LLM_SYSTEM = """You triage a student's inbox. Decide if this email is about a job/internship application
HE made (confirmation, viewed, rejection, interview, assessment, request for info, offer).
Job alerts, recommendations, newsletters and marketing are NOT about his application. Neither are job
SUGGESTIONS or invitations to apply ("React JS Internship @ Company", "jobs picked for you", "apply now"), nor
reminders to finish an application he never submitted ("You're almost there! Complete your application").
Only say true if he actually SUBMITTED an application (or the email is a decision/next step about one).
Return JSON only: {"about_my_application": true/false, "company": "employer name or null",
"role": "job title or null"}. The company is the EMPLOYER, never the platform (LinkedIn, Greenhouse, etc.)."""


def extract(email: Email, use_llm: bool = True) -> dict:
    r = rules_extract(email)
    if r["related"] is False or (r["related"] and r["company"]):
        return r
    if use_llm:
        out = chat_json(LLM_SYSTEM, f"From: {email.from_name} <{email.from_addr}>\nSubject: {email.subject}\n\n"
                                    f"{email.body[:1200]}")
        if out is None and last_call_failed():
            return {"related": None, "company": r["company"], "role": r["role"], "deferred": True}
        if out is not None:
            return {"related": bool(out.get("about_my_application")),
                    "company": _clean_company(out.get("company")) or r["company"],
                    "role": (clean_role(out["role"])[:80] if out.get("role") else r["role"])}
    return {"related": bool(r["related"]) or (bool(r["company"]) and
                                              bool(APPLICATION_RE.search(f"{email.subject} {email.body[:1500]}"))),
            "company": r["company"], "role": r["role"], "unsure": True}  # rules' best guess; worth an AI check


def _role_key(role: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (role or "").lower().replace("programme", "program").replace("programm", "program"))


def find_by_company(conn, company: str, role: str | None = None, loose: bool = False):
    """Same company AND same role -> that row. Same company, role not stated yet -> that row (role gets filled).
    Same company but a DIFFERENT role -> None, so a separate application gets its own row."""
    key = norm(company)
    if not key or company.lower().startswith("unknown"):
        return None  # "Unknown employer (via indeed.com)" is not the same company as any other unknown
    same_company = []
    for row in conn.execute("SELECT * FROM applications ORDER BY id").fetchall():
        if (row["company"] or "").lower().startswith("unknown"):
            continue
        k = norm(row["company"])
        if k and (k == key or (len(key) >= 5 and (key in k or k in key))):
            same_company.append(row)
    if not same_company:
        return None
    if not role:
        return same_company[0]
    rk = _role_key(role)
    for row in same_company:
        k = _role_key(row["role"])
        if k and (k == rk or (min(len(k), len(rk)) >= 10 and (k in rk or rk in k))):
            return row
    for row in same_company:
        if (row["role"] or "").startswith("(role"):
            return row  # role not known yet: this email tells us
    if len(same_company) == 1:
        only = same_company[0]
        from_portal = (only["posting_url"] or "").startswith("via ") and only["posting_url"] != "via company email"
        if loose or not from_portal:
            # One application at this company, seen from two angles: HR calls it "Summer Internship Programme",
            # the portal (RozeeGPT) calls it "IT & IS Intern", and the company name is written differently in
            # each ("Menzies-Ras" / "MENZIES-RAS Pvt Ltd"). Same application. Only two PORTAL confirmations with
            # different roles mean two applications (Binance: 6 roles, 6 confirmation emails).
            return only
    return None


def ingest(conn, email: Email, use_llm: bool = True, cls=None, info: dict | None = None) -> dict:
    """One inbox email -> ignored, matched to an application, or a new application. Then the normal pipeline.
    cls / info: a classification / extraction already made for this email (the web worker batches them)."""
    from .pipeline import process_email

    if conn.execute("SELECT 1 FROM messages WHERE gmail_id=?", (email.gmail_id,)).fetchone():
        return {"gmail_id": email.gmail_id, "skipped": "already processed"}
    platform_mail = is_platform(email.from_addr) or is_noreply(email.from_addr)
    if not platform_mail and conn.execute("SELECT 1 FROM applications WHERE thread_id=?", (email.thread_id,)).fetchone():
        return process_email(conn, email, use_llm=use_llm, cls=cls)  # a person replying in a thread we already track
    # portal emails: Gmail groups e.g. 6 "Thank you for your application to Binance" emails into ONE thread even
    # though each is a different role, so match them on company + role instead of on the thread

    info = info or extract(email, use_llm)
    if info.get("deferred"):
        # the AI was rate-limited and the rules weren't sure: record nothing, so the next scan retries it
        return {"gmail_id": email.gmail_id, "skipped": "deferred (AI limit reached)", "subject": email.subject}
    if info["related"] and not info["company"] and is_platform(email.from_addr):
        # clearly about your application, but the portal didn't name the employer: still track it
        info["company"] = f"Unknown employer (via {email.from_addr.split('@')[-1]})"
    if not info["related"] or not info["company"]:
        conn.execute("INSERT INTO messages(gmail_id, thread_id, from_addr, subject, received_at, label, reasoning, "
                     "match_method, processed_at) VALUES (?,?,?,?,?,'IGNORED',?, 'scan', ?)",
                     (email.gmail_id, email.thread_id, email.from_addr, email.subject, email.received_at,
                      "Not about an application you made" if not info["related"] else "Couldn't tell which company",
                      db.now()))
        conn.commit()
        return {"gmail_id": email.gmail_id, "skipped": "not about your application",
                "subject": email.subject, "company": info["company"]}

    app = find_by_company(conn, info["company"], info["role"], loose=not platform_mail)
    created = False
    if not app:
        platform = is_platform(email.from_addr)
        app_id = db.add_application(
            conn, info["company"], info["role"] or "(role not stated)",
            contact_email=None if platform or is_noreply(email.from_addr) else email.from_addr,
            thread_id=email.thread_id, applied_at=email.received_at or None, subject=email.subject, kind="application")
        src = email.from_addr.split("@")[-1] if platform else "company email"
        conn.execute("UPDATE applications SET posting_url=COALESCE(posting_url, ?) WHERE id=?", (f"via {src}", app_id))
        conn.commit()
        created = True
    else:
        app_id = app["id"]
        generic = (app["role"] or "").startswith("(role") or app["posting_url"] == "via company email"
        if info["role"] and generic and (is_platform(email.from_addr) or (app["role"] or "").startswith("(role")):
            # the portal's confirmation names the exact role ("IT & IS Intern"); keep that over HR's programme name
            conn.execute("UPDATE applications SET role=? WHERE id=?", (info["role"], app_id))
    res = process_email(conn, email, use_llm=use_llm, force_app_id=app_id, cls=cls)
    res["created"] = created
    return res