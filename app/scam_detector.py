"""Scam / exploitative job-posting detector.

Two layers, deliberately:
  1. Deterministic rules  -> cheap, explainable, never hallucinate, work offline.
  2. LLM review           -> catches subtle patterns rules miss (tone, inconsistency).

The LLM is *grounded*: every red flag it returns must quote evidence from the
posting, and we drop any flag whose quote isn't actually in the text. Critical
rule hits (upfront fees, ID/bank requests) set a score floor the LLM can't
talk its way below.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from .domain_check import domain_age_days
from .llm import chat_json

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com", "aol.com",
    "proton.me", "protonmail.com", "icloud.com", "mail.com", "gmx.com", "yandex.com",
}

CATEGORIES = ("payment", "company", "content", "process")


@dataclass
class Flag:
    category: str      # payment | company | content | process
    severity: str      # critical | high | medium | low
    title: str
    evidence: str      # exact quote from the posting (or a fact we looked up)
    why: str           # plain-English explanation
    source: str = "rule"  # rule | llm | lookup


@dataclass
class ScamReport:
    verdict: str                 # likely_scam | suspicious | looks_legit
    risk_score: int              # 0-100
    summary: str
    flags: list[Flag] = field(default_factory=list)
    green_flags: list[str] = field(default_factory=list)
    mode: str = "rules"          # rules | rules+llm

    def to_dict(self):
        d = asdict(self)
        d["flags"] = [asdict(f) for f in self.flags]
        return d


SEVERITY_POINTS = {"critical": 40, "high": 25, "medium": 12, "low": 5}

# (category, severity, title, regex, why)
RULES = [
    ("payment", "critical", "Asks you to pay to get the job",
     r"\b(registration|training|onboarding|processing|application|certification|starter[- ]kit|visa processing|security)\s+(fee|deposit|charge)s?\b"
     r"|\bpay\s+(a|the|for|your)\s+[^.]{0,40}(fee|kit|training|equipment|deposit)"
     r"|\brefundable\s+(deposit|fee)",
     "Real employers pay you. Any fee to apply, train, or 'secure your slot' is the most common job-scam pattern."),
    ("payment", "critical", "Fake-check / equipment purchase scheme",
     r"(send|mail|deposit)\s+(you\s+)?a\s+(check|cheque)|purchase\s+[^.]{0,40}from\s+our\s+(approved\s+|preferred\s+)?vendor",
     "Scammers send a check, ask you to buy 'equipment' from their vendor, then the check bounces and you're out the money."),
    ("payment", "high", "Paid in crypto or gift cards",
     r"\b(paid|payment|salary)\b[^.]{0,40}\b(crypto|bitcoin|btc|usdt|gift\s*cards?)\b",
     "Legitimate payroll doesn't run on crypto or gift cards; these are untraceable and irreversible."),
    ("payment", "high", "Unpaid 'trial' work",
     r"\bunpaid\s+(trial|test|probation|assignment)|\btrial\s+(task|period|project|week)[^.]{0,60}\b(unpaid|without\s+pay|no\s+pay|not\s+paid)"
     r"|\bfree\s+(sample|trial)\s+(work|task|article|design)",
     "Substantial unpaid 'trial' work is how some outfits get free labour and never hire. Short take-homes are normal; multi-day unpaid work is not."),
    ("process", "critical", "Requests ID or banking details early",
     r"\b(send|provide|share|submit|upload)\b[^.]{0,60}\b(cnic|ssn|social\s+security|passport|national\s+id|id\s+card|bank\s+(account|details|login)|iban|credit\s+card|debit\s+card)\b",
     "Identity documents and bank details belong in formal onboarding after a signed offer, never in the application stage."),
    ("process", "high", "Hiring runs over Telegram / WhatsApp",
     r"\b(telegram|whatsapp|signal|wickr|kik)\b",
     "Scammers move you to encrypted chat apps to avoid company email trails and platform moderation."),
    ("process", "high", "Offer without a real interview",
     r"\bno\s+interview|hired\s+(immediately|instantly|on\s+the\s+spot)|instant\s+(hiring|offer|approval)|you\s+have\s+been\s+(selected|shortlisted)\s+without",
     "Real hiring involves speaking with someone. Instant offers are designed to rush you past scrutiny."),
    ("process", "critical", "Asks for an OTP / verification code",
     r"\b(share|send|provide|forward|tell)\b[^.]{0,40}\b(otp|one[- ]time (password|code)|verification code|pin code|6[- ]digit code)\b",
     "No employer ever needs your OTP. Sharing it hands over your WhatsApp, bank, or JazzCash/Easypaisa account."),
    ("payment", "critical", "Task scam: pay to withdraw 'earnings'",
     r"\b(withdraw(al)?|unlock|release)\b[^.]{0,50}\b(fee|tax|deposit|recharge|upgrade)\b|\bupgrade\s+(your\s+)?(level|vip|account)\b"
     r"|\b(like|subscribe to|rate|review)\s+[^.]{0,30}\b(posts?|videos?|channels?|products?|hotels?|apps?)\b[^.]{0,60}\b(earn|paid|commission|per task)\b",
     "The 'like posts and earn' pattern: small real payouts build trust, then a fee is demanded before you can withdraw. The money never comes back."),
    ("process", "high", "Overseas job with guaranteed visa / free travel",
     r"\b(guaranteed|easy|free|fast)\s+(visa|work permit)\b|\bfree\s+(air\s+)?tickets?\b|\b(submit|hand over|give)\s+(your\s+)?original\s+passport",
     "Pakistan's Bureau of Emigration has warned about fake overseas jobs with 'easy visas' that trap people in scam compounds. Verify the agent is licensed by the Bureau of Emigration & Overseas Employment."),
    ("content", "medium", "Urgency pressure",
     r"\burgent(ly)?\b|\bact\s+now\b|\blimited\s+(slots|seats|positions)\b|\bwithin\s+24\s+hours\b|\bimmediate\s+start\b|\bapply\s+today\s+only\b",
     "Artificial deadlines stop you from checking the company out."),
    ("content", "high", "Too-good-to-be-true pay",
     r"\bearn\s+(up\s+to\s+)?[$₨]?\s?\d[\d,]{2,}\s*(\+\s*)?(per|a|/)\s*(day|week)|\b(no\s+experience)[^.]{0,80}\b(\$|usd|rs\.?|pkr)\s?\d[\d,]{3,}"
     r"|\b(\$|usd)\s?\d{3,}\s*(per|/)\s*(hour|hr)\b",
     "Very high pay for low effort or no experience is the classic lure."),
    ("content", "low", "Vague 'easy money' language",
     r"\b(work\s+from\s+home\s+and\s+earn|be\s+your\s+own\s+boss|financial\s+freedom|unlimited\s+earning|simple\s+copy[- ]paste|data\s+entry\s+from\s+home)\b",
     "Common phrasing in mass-posted scam listings."),
]


def _find(pattern: str, text: str):
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    # widen to the surrounding sentence so the evidence reads naturally
    start = max(text.rfind(".", 0, m.start()) + 1, text.rfind("\n", 0, m.start()) + 1, 0)
    end_candidates = [i for i in (text.find(".", m.end()), text.find("\n", m.end())) if i != -1]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return text[start:end].strip()[:220]


EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+\.)+[\w-]{2,}")
URL_RE = re.compile(r"https?://(?:www\.)?([\w-]+(?:\.[\w-]+)+)", re.IGNORECASE)


def _domain_of_email(addr: str) -> str:
    return addr.split("@")[-1].lower()


def run_rules(text: str, company: str = "", website: str = "", contact_email: str = "",
              check_domain_age: bool = True) -> tuple[list[Flag], list[str]]:
    flags: list[Flag] = []
    greens: list[str] = []

    for category, severity, title, pattern, why in RULES:
        ev = _find(pattern, text)
        if ev:
            flags.append(Flag(category, severity, title, ev, why))

    # --- salary transparency ---
    has_number_pay = re.search(r"(\$|usd|pkr|rs\.?|₨|€|£)\s?\d|\d[\d,]*\s?(k|,000)\s*(-|–|to)\s*\d|\d[\d,]{3,}\s*(pkr|usd|per\s+month|/\s?month|a\s+year|per\s+year)", text, re.I)
    vague_pay = re.search(r"\b(competitive|attractive|handsome|negotiable|market[- ]leading)\s+(salary|pay|package|compensation)\b", text, re.I)
    if has_number_pay:
        greens.append("States concrete pay.")
    elif vague_pay:
        flags.append(Flag("payment", "low", "Salary is vague", vague_pay.group(0),
                          "Not a scam signal by itself (many real jobs do this), but it removes one way to check the offer is realistic."))

    # --- contact email vs company domain ---
    emails = [contact_email] if contact_email else [m.group(0) for m in EMAIL_RE.finditer(text)]
    urls = [website] if website else [m.group(1) for m in URL_RE.finditer(text)]
    site_domain = ""
    if urls:
        site_domain = re.sub(r"^https?://(www\.)?", "", urls[0].lower()).split("/")[0]
    for addr in emails:
        dom = _domain_of_email(addr)
        if dom in FREE_EMAIL_DOMAINS:
            flags.append(Flag("company", "medium", "Recruiter uses a free email address", addr,
                              "Established companies recruit from their own domain. A Gmail/Outlook contact for a 'company' role is a common impersonation tactic."))
        elif site_domain and not (dom == site_domain or dom.endswith("." + site_domain) or site_domain.endswith("." + dom)):
            flags.append(Flag("company", "medium", "Email domain doesn't match company website",
                              f"{addr} vs {site_domain}",
                              "Look-alike domains are used to impersonate real companies."))
        else:
            greens.append(f"Contact uses a company domain ({dom}).")
        break

    if not urls:
        flags.append(Flag("company", "low", "No company website given", "(no URL in posting)",
                          "Not proof of anything, but you'll need to verify the company exists before sharing details."))
    elif check_domain_age and site_domain:
        age = domain_age_days(site_domain)
        if age is not None:
            if age < 180:
                flags.append(Flag("company", "high", "Company domain registered very recently",
                                  f"{site_domain} registered {age} days ago",
                                  "Scam 'companies' often spin up fresh domains. A months-old domain for an 'established' firm doesn't add up.", "lookup"))
            else:
                greens.append(f"Website domain is {age // 365}+ years old." if age >= 365 else f"Website domain is {age} days old.")

    if not company and not re.search(r"\b(about us|our company|founded|headquartered)\b", text, re.I):
        flags.append(Flag("company", "low", "Company barely identified", "(no company name or background)",
                          "Anonymous postings make it impossible to verify who you'd work for."))

    # --- shouting / exclamation density ---
    exclam = text.count("!")
    caps_words = len(re.findall(r"\b[A-Z]{4,}\b", text))
    if exclam >= 4 or caps_words >= 5:
        flags.append(Flag("content", "low", "Hype formatting", f"{exclam} exclamation marks, {caps_words} ALL-CAPS words",
                          "Professional postings rarely shout; spam-style formatting correlates with low-quality or fake listings."))

    return flags, greens


def _score(flags: list[Flag]) -> int:
    return min(100, sum(SEVERITY_POINTS[f.severity] for f in flags))


def _verdict(score: int) -> str:
    return "likely_scam" if score >= 50 else "suspicious" if score >= 25 else "looks_legit"


LLM_SYSTEM = """You are a careful job-fraud analyst helping a student job-seeker.
Assess whether a job posting is fake or exploitative. Consider: payment red flags
(fees, unpaid trial work, vague pay), company legitimacy, content patterns (urgency,
too-good pay, inconsistencies, copy-paste text), and process red flags (chat-app
hiring, instant offers, early ID/bank requests).
Be fair: vague salary alone or a short take-home test is NOT a scam.
Return JSON only:
{"risk_score": 0-100,
 "red_flags": [{"category": "payment|company|content|process",
                "severity": "critical|high|medium|low",
                "title": "short label",
                "evidence": "EXACT quote copied from the posting",
                "why": "one plain-English sentence"}],
 "green_flags": ["short strings"],
 "summary": "2 sentences max, plain English, addressed to the job-seeker"}"""


def _grounded(evidence: str, text: str) -> bool:
    """Is the LLM's quoted evidence actually in the posting? (fuzzy, case-insensitive)"""
    ev = " ".join(evidence.lower().split())
    body = " ".join(text.lower().split())
    if not ev or len(ev) < 4:
        return False
    if ev in body:
        return True
    # allow small paraphrase: best matching window
    window = len(ev)
    best = 0.0
    step = max(1, window // 4)
    for i in range(0, max(1, len(body) - window + 1), step):
        best = max(best, SequenceMatcher(None, ev, body[i:i + window]).ratio())
        if best > 0.85:
            return True
    return False


def analyze_posting(text: str, company: str = "", website: str = "", contact_email: str = "",
                    use_llm: bool = True, check_domain_age: bool = True) -> ScamReport:
    rule_flags, greens = run_rules(text, company, website, contact_email, check_domain_age)
    rule_score = _score(rule_flags)
    has_critical = any(f.severity == "critical" for f in rule_flags)

    llm = None
    if use_llm:
        context = f"Company: {company or '(not given)'}\nWebsite: {website or '(not given)'}\nContact: {contact_email or '(not given)'}\n"
        rule_hint = "\n".join(f"- {f.title}: {f.evidence}" for f in rule_flags) or "(none)"
        llm = chat_json(LLM_SYSTEM, f"{context}\nRule engine already found:\n{rule_hint}\n\nPOSTING:\n\"\"\"\n{text}\n\"\"\"")

    if not llm:
        summary = _rules_summary(rule_flags, rule_score)
        return ScamReport(_verdict(rule_score), rule_score, summary, rule_flags, greens, "rules")

    flags = list(rule_flags)
    seen = {f.title.lower() for f in flags}
    for rf in llm.get("red_flags", []):
        ev = str(rf.get("evidence", ""))
        if not _grounded(ev, text):
            continue  # drop hallucinated evidence
        title = str(rf.get("title", "Red flag"))[:80]
        if title.lower() in seen:
            continue
        sev = rf.get("severity") if rf.get("severity") in SEVERITY_POINTS else "medium"
        cat = rf.get("category") if rf.get("category") in CATEGORIES else "content"
        flags.append(Flag(cat, sev, title, ev[:220], str(rf.get("why", ""))[:300], "llm"))
        seen.add(title.lower())

    llm_score = int(max(0, min(100, llm.get("risk_score", rule_score))))
    score = round(0.5 * llm_score + 0.5 * _score(flags))
    if has_critical:
        score = max(score, 60)  # hard floor: fees / ID requests are never "fine"
    greens += [g for g in llm.get("green_flags", []) if isinstance(g, str)][:4]
    summary = str(llm.get("summary") or _rules_summary(flags, score))
    return ScamReport(_verdict(score), score, summary, flags, greens, "rules+llm")


def _rules_summary(flags: list[Flag], score: int) -> str:
    if not flags:
        return "No red flags found. Still verify the company independently before sharing personal details."
    top = sorted(flags, key=lambda f: -SEVERITY_POINTS[f.severity])[:2]
    reasons = "; ".join(f.title.lower() for f in top)
    if score >= 50:
        if any(f.severity == "critical" for f in flags):
            return f"This looks like a scam: {reasons}. Don't pay anything or share ID/bank details."
        return (f"High risk / likely exploitative: {reasons}. Ask for a paid trial and a company email "
                f"address before investing time.")
    if score >= 25:
        return f"Proceed carefully: {reasons}. Verify the company before going further."
    return f"Mostly fine, with minor concerns: {reasons}."