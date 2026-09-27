"""Domain age lookup via RDAP (the free, keyless successor to WHOIS).

rdap.org redirects to the right registry. Results are cached in-process.
Returns None on any failure so the detector degrades gracefully offline.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache

import requests


@lru_cache(maxsize=256)
def domain_age_days(domain: str) -> int | None:
    if os.getenv("JOBPILOT_OFFLINE") == "1":
        return None
    # strip subdomains for common cases: careers.acme.com -> acme.com
    parts = domain.lower().strip(".").split(".")
    root = ".".join(parts[-3:]) if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "edu", "gov"} else ".".join(parts[-2:])
    try:
        r = requests.get(f"https://rdap.org/domain/{root}", timeout=6)
        if r.status_code != 200:
            return None
        for ev in r.json().get("events", []):
            if ev.get("eventAction") == "registration":
                created = datetime.fromisoformat(ev["eventDate"].replace("Z", "+00:00"))
                return (datetime.now(timezone.utc) - created).days
    except Exception:
        return None
    return None
