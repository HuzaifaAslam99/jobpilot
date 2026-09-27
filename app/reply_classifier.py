"""Classify replies to job outreach and draft responses.

Order of operations:
  1. Header check for auto-replies (deterministic, no LLM call wasted).
  2. LLM classification with a confidence score, if available.
  3. Keyword rules as fallback / tie-breaker.
  4. Anything low-confidence becomes UNCLEAR -> human review.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .llm import chat_json, last_call_failed

LABELS = ("REJECTED", "INTERESTED", "INTERVIEW_REQUEST", "NEEDS_INFO", "AUTO_REPLY", "UNCLEAR")
CONFIDENCE_THRESHOLD = 0.70


@dataclass
class Email:
    gmail_id: str
    thread_id: str
    from_addr: str
    from_name: str
    subject: str
    body: str
    received_at: str = ""
    headers: dict | None = None
    you_replied_at: str | None = None   # your latest message in the same thread, if after this one


@dataclass
class Classification:
    label: str
    confidence: float
    reasoning: str
    key_details: dict      # e.g. {"proposed_times": [...], "requested_items": [...], "deadline": ...}
    source: str            # header | llm | rules


AUTO_HEADERS = {"auto-submitted": re.compile(r"auto-(replied|generated)", re.I),
                "x-autoreply": re.compile(r".+"), "x-autorespond": re.compile(r".+"),
                "precedence": re.compile(r"auto_reply|bulk|junk", re.I)}

RULES = [
    ("AUTO_REPLY", 0.90, r"\b(out of (the )?office|automatic reply|auto(matic)?[- ]?reply|on (annual )?leave until|currently away|we (have )?received your application|thank you for applying|your application (was|has been) (sent|submitted|received|viewed)|application (received|submitted)|(?:your )?application has been received)\b"),
    ("REJECTED", 0.85, r"\b(decided (not )?to (move|proceed) forward with other|not (be )?moving forward|won'?t be moving forward|(?:will|shall) not be (?:moving|proceeding) forward|will not be moving forward|unable to offer|position has (been|now been) filled|not selected|pursue other candidates|regret to inform|unfortunately[^.]{0,80}(not|other candidates|another candidate)|"
     r"(?:move|moving|proceed|proceeding|go|going) (?:forward |ahead )?with (?:an?other|other|a different) (?:candidates?|applicants?|profiles?)|"
     r"(?:not|n't) (?:be )?(?:able to )?(?:move|moving|proceed|proceeding|progress|progressing) (?:forward |further )?with your (?:application|candidacy|profile)|"
     r"(?:not to|to not) (?:move|moving|proceed|progress|continue|pursue)(?: forward| further)?|(?:have|has|were|was) not been (?:selected|shortlisted|successful)|"
     r"application (?:was|has been|is) (?:unsuccessful|not successful)|no longer (?:being )?consider(?:ed|ing)|"
     r"(?:role|position|vacancy|job) (?:is|has been) no longer available)\b"),
    ("INTERVIEW_REQUEST", 0.80, r"\b(schedule (a|an) (call|interview|chat)|your availabilit(?:y|ies)|schedule (?:your|the|a|an)(?: next)? (?:call|interview|chat)|(?:online|video|virtual) interview|available (for|on)|book a (time|slot)|calendly\.com|interview (on|at|with)|technical (interview|round)|next round|phone screen|complete (the|your) (ai |video |online |technical )?(interview|assessment|test|challenge)|online assessment|can (i|we) (call|speak|talk|chat)|(call|speak with) you (tomorrow|today|this week|on)|hop on a (quick )?call|quick call)\b"),
    ("NEEDS_INFO", 0.75, r"\b(could you (please )?(send|share|provide)|please (send|share|provide|attach)|can you (send|share)|we('d| would) need|missing (your )?(cv|resume|transcript|portfolio)|expected salary|notice period)\b"),
    ("INTERESTED", 0.75, r"\b((?<!not )(?<!not been )(?<!n't been )selected for (the )?(next|phase|round|stage|second)|selection for (phase|round|stage)|(?<!not )(?<!not been )(?<!n't been )shortlisted|(phase|round|stage) (2|two|ii)\b|next (phase|stage|round)|moved? you (forward|to the next))"),
    ("INTERESTED", 0.70, r"\b(impressed|interested in (your|learning)|would love to (chat|talk|learn)|like to (move|take) (you|this) forward|great fit|strong fit)\b"),
]

LLM_SYSTEM = f"""You classify email replies a student job-seeker received after applying or doing outreach.
Labels:
- REJECTED: clearly declines the candidate.
- INTERVIEW_REQUEST: asks to schedule a call/interview/assessment or proposes times.
- INTERESTED: positive, wants to move forward, but no concrete scheduling ask yet.
- NEEDS_INFO: asks for documents or answers (CV, portfolio, salary expectation, availability of start date).
- AUTO_REPLY: out-of-office or automated acknowledgement; no human decision in it.
- UNCLEAR: mixed, ambiguous, or you are not confident.
If an email both asks for info AND to schedule, prefer INTERVIEW_REQUEST and list the info in key_details.
respond_via: "link" if the candidate should act through a link/portal (take an assessment, AI interview,
fill a form, book on a calendar page) rather than by replying to the email; otherwise "email".
Return JSON only:
{{"label": one of {list(LABELS)},
  "confidence": 0.0-1.0,
  "reasoning": "one sentence citing the phrase that decided it",
  "key_details": {{"proposed_times": [], "requested_items": [], "deadline": null, "contact_name": null,
                   "respond_via": "email|link"}}}}"""


def _is_bounce(email: Email) -> bool:
    if re.search(r"\(delay\)|delivery (has been )?delayed|will (keep|continue) trying", f"{email.subject} {email.body[:300]}", re.I):
        return False  # a delay is a warning, not a failure
    return bool(re.match(r"(mailer-daemon|postmaster)@", email.from_addr or "", re.I)
                or re.search(r"mail delivery (subsystem|system)", email.from_name or "", re.I)
                or re.search(r"delivery status notification|undeliver(able|ed)|address not found", email.subject or "", re.I))


def _header_auto(email: Email) -> bool:
    for k, v in (email.headers or {}).items():
        pat = AUTO_HEADERS.get(k.lower())
        if pat and pat.search(str(v)):
            return True
    return False


NEGATED_PROGRESS_RE = re.compile(r"[^.!?,;]*\b(?:not|n't) (?:been |be )?(?:selected|shortlisted|chosen|invited|"
                                 r"progress\w*|moving|moved|proceed\w*)\b[^.!?,;]*", re.I)
LINKEDIN_UPDATE_RE = re.compile(r"^\s*your application to .+ at .+", re.I)

# Hypotheticals aren't decisions: "If you are not selected for this position, keep an eye on our jobs page"
CONDITIONAL_RE = re.compile(r"\b(?:if|should|in case|in the event that|once)\b[^.!?\n]{0,60}\b(?:not|n't|unsuccessful|"
                            r"successful|shortlisted|selected|chosen|a match|qualif\w*)\b[^.!?\n]*[.!?]?", re.I)


def _rules(email: Email) -> Classification:
    # plain-text emails wrap lines ("will not be moving\nforward"): collapse whitespace before matching
    text = " ".join(CONDITIONAL_RE.sub(" ", f"{email.subject}.\n{' '.join(email.body.split())}").split())
    # "You have not been shortlisted for the next round" is a rejection; its "next round" isn't good news
    positive_text = NEGATED_PROGRESS_RE.sub(" ", text)
    hits = []
    for label, conf, pat in RULES:
        m = re.search(pat, positive_text if label in ("INTERESTED", "INTERVIEW_REQUEST") else text, re.I)
        if m:
            hits.append((label, conf, m.group(0)))
    if not hits and LINKEDIN_UPDATE_RE.search(email.subject or "") and re.search(r"linkedin\.com$", email.from_addr or "") \
            and re.search(r"your update from", text, re.I):
        # LinkedIn only sends "Your application to <role> at <company>" + "Your update from <company>" when the
        # employer closes your application; confirmations and "viewed" notices use different subjects
        return Classification("REJECTED", 0.75, 'LinkedIn "Your update from ..." notice (employer closed the '
                              "application).", {}, "rules")
    if not hits:
        return Classification("UNCLEAR", 0.3, "No clear signal found; needs a human read.", {}, "rules")
    labels = {h[0] for h in hits}
    if "AUTO_REPLY" in labels and len(labels) > 1:
        # "Thank you for applying ... unfortunately" is a rejection, not an acknowledgement
        hits = [h for h in hits if h[0] != "AUTO_REPLY"]
        labels.discard("AUTO_REPLY")
    # rejection + scheduling language together is contradictory -> human
    if "REJECTED" in labels and labels & {"INTERVIEW_REQUEST", "INTERESTED"}:
        return Classification("UNCLEAR", 0.4, "Contains both rejection and positive/scheduling language.", {}, "rules")
    label, conf, phrase = hits[0]
    if label == "NEEDS_INFO" and "INTERVIEW_REQUEST" in labels:
        label, conf, phrase = next(h for h in hits if h[0] == "INTERVIEW_REQUEST")
    return Classification(label, conf, f'Matched "{phrase}".', {"requested_items": _requested_items(text)}, "rules")


ITEM_PATTERNS = {
    "portfolio / project links": r"portfolio|projects? you'?re proud of|github",
    "CV": r"\b(cv|resume|résumé)\b",
    "transcript": r"transcript|grades|cgpa",
    "expected stipend/salary": r"expected (monthly )?(stipend|salary|compensation)|salary expectation",
    "earliest start date": r"start date|notice period|when (can|could) you (start|join)",
    "references": r"\breferences?\b",
}


LINK_ACTION_RE = re.compile(r"\b(complete|take|start|finish|attempt|submit)\b[^.\n]{0,40}\b(assessment|test|challenge|"
                            r"ai interview|interview|form|questionnaire|task)\b[^.\n]{0,80}(link|below|portal|here|https?://)"
                            r"|\b(click|use) (the|this) link\b|calendly\.com|book (a|your) (slot|time) (here|using|via)", re.I)


EMAIL_ASK_RE = re.compile(r"\b(share|send|let (me|us) know|reply (with|to)|your availability|when (are|would) you|"
                          r"what times?|could you (tell|confirm))\b", re.I)


REPLY_WINDOW_RE = re.compile(r"(?:you can |feel free to |simply )?reply (?:directly )?to this (?:email|message)"
                             r"(?:[^.\n]{0,40}?within (\d+) days)?", re.I)


def reply_allowed(text: str, age_days: int | None) -> tuple[bool, bool]:
    """Portal emails that say 'you can reply to this email (within N days)'. -> (replyable, window_expired)"""
    m = REPLY_WINDOW_RE.search(text or "")
    if not m:
        return False, False
    if m.group(1) and age_days is not None and age_days > int(m.group(1)):
        return True, True
    return True, False


EMAIL_ADDR_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
CONTACT_HINT_RE = re.compile(r"\b(?:cc|copy|contact|reach|write to|email|e-mail|reply to|send (?:it|them|us) to)\b", re.I)
REAPPLY_RE = re.compile(r"\bre-?apply(?: again)?(?: to us| with us)? (?:in|after) (\d+)\s*(?:-|–|to|or)?\s*(\d+)?\s*months?", re.I)


def contact_address(text: str) -> str | None:
    """A real address a no-reply email points you to: "Please include talent@nethermind.io in cc in all your replies"."""
    for m in EMAIL_ADDR_RE.finditer(text or ""):
        addr = m.group(0).rstrip(".").lower()
        if re.search(r"no-?reply|do-?not-?reply|notifications?@|privacy|unsubscribe|support@|help@", addr):
            continue
        window = text[max(0, m.start() - 80): m.end() + 40]
        if CONTACT_HINT_RE.search(window):
            return addr
    return None


def reapply_window(text: str, age_days: int | None) -> dict | None:
    """ "Feel free to reapply in 3-6 months" -> when that window opens/closes, counted from their email."""
    m = REAPPLY_RE.search(text or "")
    if not m or age_days is None:
        return None
    lo = int(m.group(1))
    hi = int(m.group(2) or lo)
    opens_in, closes_in = lo * 30 - age_days, hi * 30 - age_days
    state = "open now" if opens_in <= 0 <= closes_in or (opens_in <= 0 and lo == hi) else (
        f"opens in ~{opens_in} days" if opens_in > 0 else "suggested window has passed, but you can still reapply")
    return {"months": f"{lo}-{hi}" if hi != lo else str(lo), "state": state}


def respond_via_link(text: str) -> bool:
    """True only when the link IS the way to respond (not when they also ask you to reply by email)."""
    return bool(LINK_ACTION_RE.search(text or "")) and not EMAIL_ASK_RE.search(text or "")


def _requested_items(text: str) -> list[str]:
    return [name for name, pat in ITEM_PATTERNS.items() if re.search(pat, text, re.I)]


CALENDAR_RE = re.compile(r"^\s*(re:\s*)?(invitation|updated invitation|accepted|declined|tentatively accepted|"
                         r"new event|event canceled|cancelled event|invitation from an unknown sender)\b.*@", re.I)
CAL_DATE_RE = re.compile(r"@\s*(?:\w{3},?\s+)?(\w{3,9}\s+\d{1,2},?\s+\d{4})")


def calendar_event(email: Email):
    """Google/Outlook calendar notices: 'Accepted: Interview @ Wed Jun 24, 2026 10:30pm'. -> (date or None, is_past)"""
    if not CALENDAR_RE.search(email.subject or ""):
        return None
    m = CAL_DATE_RE.search(email.subject or "")
    when = None
    if m:
        for fmt in ("%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%B %d %Y"):
            try:
                when = datetime.strptime(m.group(1).replace("  ", " "), fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    past = bool(when and when.date() < datetime.now(timezone.utc).date())
    return {"calendar_event": when.date().isoformat() if when else "unknown date", "event_past": past}


def _pre(email: Email) -> Classification | None:
    """Deterministic answers that need neither rules nor AI: calendar notices, bounces, auto-reply headers."""
    cal = calendar_event(email)
    if cal:
        return Classification("AUTO_REPLY", 0.97, f"Calendar notice for a meeting on {cal['calendar_event']}.",
                              cal, "header")
    if _is_bounce(email):
        return Classification("AUTO_REPLY", 0.99, "Delivery failure: your email never reached them.",
                              {"bounce": True}, "header")
    if re.match(r"(mailer-daemon|postmaster)@", email.from_addr or "", re.I):
        return Classification("AUTO_REPLY", 0.99, "Delivery delay notice: Gmail is still retrying.", {}, "header")
    if _header_auto(email):
        return Classification("AUTO_REPLY", 0.99, "Email headers mark it as automatically generated.", {}, "header")
    return None


def _combine(rule_result: Classification, llm: dict | None, use_llm: bool) -> Classification:
    """Merge the rules' answer with the AI's (or fall back to the rules when the AI gave none)."""
    if not llm or llm.get("label") not in LABELS:
        result = rule_result
        if use_llm and last_call_failed() and result.label == "UNCLEAR":
            # the rules weren't sure and the AI was rate-limited: say so, so the caller can retry later
            result.key_details = {**result.key_details, "ai_missing": True}
    else:
        try:
            conf = float(llm.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        result = Classification(llm["label"], conf, str(llm.get("reasoning", "")), llm.get("key_details") or {}, "llm")
        # disagreement between two confident-ish systems -> let a human decide
        if rule_result.label not in ("UNCLEAR", result.label) and rule_result.confidence >= 0.8 and conf < 0.9:
            result = Classification("UNCLEAR", min(conf, rule_result.confidence),
                                    f"LLM said {result.label} but rules said {rule_result.label}; needs review.",
                                    result.key_details, "llm+rules")
    if result.confidence < CONFIDENCE_THRESHOLD and result.label != "UNCLEAR":
        result = Classification("UNCLEAR", result.confidence,
                                f"Low confidence ({result.confidence:.2f}) for {result.label}: {result.reasoning}",
                                result.key_details, result.source)
    return result


def classify(email: Email, use_llm: bool = True) -> Classification:
    pre = _pre(email)
    if pre:
        return pre
    rule_result = _rules(email)
    llm = chat_json(LLM_SYSTEM, f"From: {email.from_name} <{email.from_addr}>\nSubject: {email.subject}\n\n{email.body[:2500]}") if use_llm else None
    return _combine(rule_result, llm, use_llm)


# ---------------------------------------------------------------- many emails at once (web app)

RULES_FIRST = float(os.getenv("RULES_FIRST_CONFIDENCE", "0.85"))  # rule hits this sure skip the AI
BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "6"))

BATCH_SYSTEM = LLM_SYSTEM.split("Return JSON only:")[0] + """You get several emails, each starting with "### EMAIL <n>".
Classify each one independently. Return JSON only:
{"results": [{"i": <n>, "label": one of """ + str(list(LABELS)) + """, "confidence": 0.0-1.0,
  "reasoning": "one sentence citing the deciding phrase",
  "key_details": {"proposed_times": [], "requested_items": [], "deadline": null, "respond_via": "email|link"}}]}"""


def skeleton(email: Email) -> str:
    """A fingerprint of an email's TEMPLATE, identical for every copy of e.g. LinkedIn's "Your update from X" notice
    whatever the company, role, name, dates or links. Hashed, so the cache holds no personal text."""
    import hashlib
    domain = (email.from_addr or "").split("@")[-1].lower()
    text = f"{email.subject}\n{email.body[:1500]}"
    text = re.sub(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", text)
    text = re.sub(r"\b[A-Z][\w&'.-]*", "X", text)          # names, companies, roles, months
    text = re.sub(r"\d+", "0", text)
    text = " ".join(re.sub(r"[^\w\s]", " ", text).lower().split())
    return hashlib.sha256(f"{domain}|{text}".encode()).hexdigest()


def _is_template_sender(email: Email) -> bool:
    from .inbox_scan import is_noreply, is_platform  # (import here: inbox_scan imports this module)
    return is_platform(email.from_addr) or is_noreply(email.from_addr)


def classify_many(emails: list[Email], use_llm: bool = True, cache=None,
                  batch_size: int = BATCH_SIZE) -> list[Classification]:
    """classify() for a list, built to save AI calls when many users share free limits:
      1. deterministic checks (calendar, bounce, auto-reply headers)
      2. rules first: a confident rule hit needs no AI
      3. template cache: a portal template the AI already read (for anyone) is reused
      4. the rest go to the AI several emails per request
    cache: object with get(key) -> dict | None and put(key, dict); only portal/no-reply templates are cached."""
    out: list[Classification | None] = [None] * len(emails)
    need = []  # (index, email, rule_result, cache_key)
    for i, e in enumerate(emails):
        pre = _pre(e)
        if pre:
            out[i] = pre
            continue
        r = _rules(e)
        if r.label != "UNCLEAR" and r.confidence >= RULES_FIRST:
            out[i] = r
            continue
        key = skeleton(e) if cache is not None and _is_template_sender(e) else None
        hit = cache.get(key) if key else None
        if hit and hit.get("label") in LABELS:
            out[i] = Classification(hit["label"], float(hit.get("confidence", 0.8)),
                                    str(hit.get("reasoning", "")) + " (same template seen before)", {}, "cache")
            continue
        need.append((i, e, r, key))

    for start in range(0, len(need), max(1, batch_size)):
        chunk = need[start:start + batch_size]
        answers = {}
        if use_llm:
            prompt = "\n\n".join(f"### EMAIL {n}\nFrom: {e.from_name} <{e.from_addr}>\nSubject: {e.subject}\n\n"
                                  f"{e.body[:1200]}" for n, (_, e, _, _) in enumerate(chunk, 1))
            got = chat_json(BATCH_SYSTEM, prompt, max_tokens=300 + 220 * len(chunk))
            for item in (got or {}).get("results", []) if isinstance(got, dict) else []:
                try:
                    answers[int(item.get("i"))] = item
                except (TypeError, ValueError):
                    continue
        for n, (i, e, r, key) in enumerate(chunk, 1):
            res = _combine(r, answers.get(n), use_llm)
            out[i] = res
            if key and res.source == "llm" and res.label != "UNCLEAR" and res.confidence >= 0.85:
                cache.put(key, {"label": res.label, "confidence": res.confidence, "reasoning": res.reasoning})
    return out


# ---------------------------------------------------------------- drafting

DRAFT_SYSTEM = """You draft short, warm, professional email replies for a student job-seeker.
Never invent facts about the candidate. Greet with the greeting you are given, never a name placeholder.
Use a bracketed placeholder only for facts only the candidate knows, like [your availability]. Never claim something is attached unless told it is; write [attach X] instead.
Never agree to pay money or share ID/bank details. Respect dates: never accept a date or deadline that
is already in the past. Format as greeting, blank line, 1-2 short paragraphs, blank line, sign-off. 90 words max.
Return JSON only: {"subject": "...", "body": "..."}"""


def clean_role(subject: str) -> str:
    """Turn an application email subject into a role name for drafts and the board."""
    r = re.sub(r"^\s*((re|fwd?|fw)\s*:\s*)+", "", subject or "", flags=re.I)
    r = re.sub(r"\bapplication\s+(for|to)\b|\bapplication\b", "", r, flags=re.I)
    r = re.sub(r"[\s,–—-]*(local candidate|open to opportunities)\b", "", r, flags=re.I)
    r = re.sub(r"\s*[–—-]\s*huzaifa\b.*$", "", r, flags=re.I)
    r = re.sub(r"\s{2,}", " ", r).strip(" –—-:,")
    r = re.sub(r"^[–—-]\s*|\s+[–—-]\s*$", "", r).strip()
    return r or (subject or "").strip()


STALE_DAYS = int(os.getenv("STALE_DAYS", "14"))


def reply_age_days(email: Email) -> int | None:
    if not email.received_at:
        return None
    try:
        received = datetime.fromisoformat(email.received_at)
        if received.tzinfo is None:
            received = received.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - received).days)
    except ValueError:
        return None


def draft_reply(email: Email, cls: Classification, candidate_name: str, company: str, role: str,
                use_llm: bool = True) -> dict | None:
    if cls.label in ("AUTO_REPLY", "UNCLEAR"):
        return None
    age = reply_age_days(email)
    if cls.label == "REJECTED" and age is not None and age > STALE_DAYS:
        return None  # a thank-you note months after the rejection looks odd; only draft it while it's fresh
    stale = age is not None and age > STALE_DAYS and cls.label != "REJECTED"
    role = clean_role(role)
    subject = email.subject if email.subject.lower().startswith("re:") else f"Re: {email.subject}"
    name = email.from_name or ""
    is_team = not name or re.search(r"\b(team|hiring|hr|human|resources?|talent|recruit\w*|careers|jobs|no-?reply|"
                                    r"support|info|admin|people|office|contact|notifications?|help|service)\b", name, re.I) \
        or company.lower().split()[0] in name.lower()
    first = f"{company} team" if is_team else name.split(" ")[0]

    if use_llm and not stale:  # an old unanswered message gets the simple template: no AI call spent on it
        goal = {
            "REJECTED": "Thank them graciously, ask to be kept in mind for future roles. No pushback.",
            "INTERVIEW_REQUEST": "Confirm enthusiasm and propose availability. If times were proposed, accept the first one "
                                 "tentatively and offer [alternative times] as a placeholder.",
            "INTERESTED": "Thank them, confirm strong interest, propose a short call with [your availability].",
            "NEEDS_INFO": "Acknowledge and say the requested items are attached / to follow, listing them.",
        }[cls.label]
        if stale:
            goal = (f"Their email is {age} days old and was never answered. Apologise briefly for the late reply, "
                    "do NOT accept any date, call time or deadline mentioned in it (they have passed), and ask whether "
                    "the opportunity is still open, offering [your availability].")
        today = datetime.now(timezone.utc).date().isoformat()
        out = chat_json(DRAFT_SYSTEM, f"Today: {today}\nTheir email was received: {email.received_at or 'unknown'}\n"
                                      f"Greeting to use: Hi {first},\n"
                                      f"Candidate: {candidate_name}\nCompany: {company}\nRole: {role}\nGoal: {goal}\n"
                                      f"Details extracted: {cls.key_details}\n\nTheir email:\n{email.body[:3000]}", temperature=0.4)
        if out and out.get("body"):
            body = out["body"].strip()
            if not re.match(r"(hi|hello|dear|hey|good (morning|afternoon|evening))\b", body, re.I):
                body = f"Hi {first},\n\n{body}"  # the AI sometimes drops the greeting
            return {"subject": subject, "body": body}

    if stale:
        return {"subject": subject, "body":
                f"Hi {first},\n\nApologies for my late reply to your message about the {role} role. "
                f"I'm still very interested in joining {company}. Is the opportunity still open? "
                f"If so, I'm available [your availability] and happy to share anything you need.\n\n"
                f"Best regards,\n{candidate_name}"}
    templates = {
        "REJECTED": f"Hi {first},\n\nThank you for letting me know, and for considering me for the {role} role. "
                    f"I enjoyed learning about {company} and would be glad to be kept in mind for future openings.\n\n"
                    f"Best regards,\n{candidate_name}",
        "INTERVIEW_REQUEST": f"Hi {first},\n\nThank you, I'd be glad to interview for the {role} role. "
                             f"I'm available [your availability — e.g. Tue/Wed 10:00–16:00 PKT]. "
                             f"Please let me know what works best, or feel free to send a calendar invite.\n\n"
                             f"Best regards,\n{candidate_name}",
        "INTERESTED": f"Hi {first},\n\nThanks for getting back to me. I'm very interested in the {role} role at {company} "
                      f"and would welcome a short call. I'm free [your availability].\n\nBest regards,\n{candidate_name}",
        "NEEDS_INFO": f"Hi {first},\n\nThank you for the follow-up. Please find the requested "
                      f"{', '.join(cls.key_details.get('requested_items') or ['information'])} attached "
                      f"[attach before sending]. Happy to share anything else you need.\n\nBest regards,\n{candidate_name}",
    }
    return {"subject": subject, "body": templates[cls.label]}