"""Company names and draft hygiene.

Company names guessed from email domains look wrong ("Vizzwebsolutions", or "Gmail" for a
@gmail.com recruiter). The best source is the greeting in YOUR original email
("Hi Vizz Web Solutions Team, ...") because that's the name you already used with them.
"""
from __future__ import annotations

import html
import re

FREE_MAIL = {"gmail", "googlemail", "yahoo", "outlook", "hotmail", "live", "aol", "icloud", "proton",
             "protonmail", "mail", "gmx", "yandex", "zoho"}
GENERIC = {"hiring", "manager", "hiring manager", "hr", "sir", "madam", "sir/madam", "team", "there", "all",
           "recruiter", "recruitment", "recruiting", "talent", "concerned", "whom it may concern", "hiring team",
           "hr team", "everyone", "folks"}


SECOND_LEVEL = {"co", "com", "org", "net", "edu", "gov", "ac", "gob"}


def org_label(domain: str) -> str:
    """The organisation part of a domain: mail.internshala.com -> internshala, team.mongodb.com -> mongodb,
    abc.com.pk -> abc (not the first label, which is often 'mail', 'team', 'hr' or 'careers')."""
    parts = [p for p in (domain or "").lower().strip(".").split(".") if p]
    if len(parts) >= 3 and parts[-2] in SECOND_LEVEL and len(parts[-1]) == 2:
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")


def domain_company(email: str | None) -> str:
    if not email or "@" not in email:
        return "Unknown"
    name = org_label(email.split("@")[-1])
    return "Unknown" if name.lower() in FREE_MAIL else name.title()


def company_from_greeting(text: str | None) -> str | None:
    """'Hi Dexterz Technologies Team, I'm Huzaifa...' -> 'Dexterz Technologies'."""
    if not text:
        return None
    text = html.unescape(text)  # Gmail previews encode apostrophes as &#39;
    m = re.match(r"\s*(?:hi|hello|dear|hey|greetings)\s*,?\s+(.{2,60}?)\s*(?:[,!:\n.]|\s+i[' ]|$)", text, re.I)
    if not m:
        return None
    name = re.split(r"\s+(?:i'?m|i am|my name|this is|hope)\b", m.group(1), flags=re.I)[0]  # no comma after greeting
    name = re.sub(r"\s+(team|hiring team|hr team|recruitment team|family)$", "", name.strip(), flags=re.I).strip()
    name = re.sub(r"^the\s+", "", name, flags=re.I)
    if not name or name.lower() in GENERIC or len(name.split()) > 6:
        return None
    if not re.match(r"[A-Z0-9]", name):
        return None  # "Hello, pursuing a career with LantroTech..." is a sentence, not a company name
    return name


def looks_auto_named(company: str | None, contact_email: str | None) -> bool:
    """True if the stored name was guessed by us (so it's safe to overwrite), not typed by the user."""
    if not company or company.startswith("Unknown"):
        return True
    return company.lower() in FREE_MAIL or company == domain_company(contact_email)


def unknown_company(company: str | None) -> bool:
    return not company or company.startswith("Unknown") or company.lower() in FREE_MAIL


PLACEHOLDER_RE = re.compile(r"\[[^\]\n]{2,60}\]")


def unfilled_placeholders(body: str) -> list[str]:
    """Placeholders like [your availability] that must be filled before an email can go out."""
    return PLACEHOLDER_RE.findall(body or "")


# Applying for a SPECIFIC thing they advertised.
STRONG_APPLICATION_RE = re.compile(
    r"\b(application for\b|apply(?:ing)? (?:for|to) (?:the|your) [^.\n]{0,60}?\b(?:position|role|internship|program|"
    r"programme|track|opening|vacancy|job)\b|(?:position|role|internship|opening|vacancy|job|program) (?:that )?you "
    r"(?:posted|advertised|listed|shared)|saw (?:your|the) (?:job )?(?:post|posting|ad|advert|opening|listing|vacancy)|"
    r"job (?:id|ref)|req(?:uisition)?\s*#?\s*\d|"
    # "I found / saw / came across an open role on your site / careers page / LinkedIn"
    r"(?:found|find|saw|see|noticed|came across|spotted) (?:a|an|the|your) (?:[\w/+-]+ ){0,4}?(?:role|position|"
    r"opening|vacancy|job|internship)[^.\n]{0,40}?\b(?:on|at|in) (?:your|the) (?:site|website|careers? page|page|"
    r"job board|linkedin|portal)|late to apply|missed the (?:deadline|application))", re.I)
NEGATED_FIND_RE = re.compile(r"\b(?:couldn'?t|could not|didn'?t|did not|can'?t|cannot|haven'?t|have not|never|wasn'?t able to)"
                             r" (?:find|see|found|seen|notice|noticed|come across|came across|spot|spotted)[^.!?\n]*", re.I)
# Reaching out without an advertised role.
COLD_RE = re.compile(
    r"\b((?:couldn'?t|could not|didn'?t|did not|can'?t|cannot) (?:find|see) (?:any )?(?:open |current |listed )?"
    r"(?:roles|positions|openings|vacancies|jobs)|no (?:open|current|listed) (?:roles|positions|openings|vacancies)|"
    r"if (?:anything|something|one|a role|a position|an opening) (?:comes up|opens)|future (?:openings|opportunities|roles)|"
    r"open to (?:new )?opportunities|reaching out|would you be open to|in case you(?:'re| are) hiring|"
    r"not sure (?:if|whether) you(?:'re| are) hiring|any (?:openings|opportunities|roles) (?:you|that|at))", re.I)
# Weaker hints that usually mean an application.
APPLICATION_RE = re.compile(r"\b(appl(y|ying|ied|ication)|submission|interest in the .{0,40}(position|role|internship)"
                            r"|(position|role|internship) (you|posted|advertised))", re.I)


def guess_kind(subject: str | None, snippet: str | None) -> str:
    """'application' if you applied to something they advertised; 'cold' if you reached out unprompted.

    Order matters: a specific application beats cold phrasing ("Application for the X internship"), cold
    phrasing beats a bare 'apply' ("no open roles, but I'd love to apply if anything comes up")."""
    text = html.unescape(f"{subject or ''} {snippet or ''}")
    # "I didn't see a role on your site" must not count as having found one: drop negated sentences first
    positive = NEGATED_FIND_RE.sub(" ", text)
    if STRONG_APPLICATION_RE.search(positive):
        return "application"
    if COLD_RE.search(text) or NEGATED_FIND_RE.search(text):  # "didn't see a role" = no advertised role
        return "cold"
    return "application" if APPLICATION_RE.search(text) else "cold"