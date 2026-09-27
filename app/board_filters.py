"""Filters for the web board: which period and which kind of outreach to show.

Nothing is deleted: filters only hide rows. A row's date is either its LAST ACTIVITY (your latest email or
their latest reply, whichever is newer) or when you APPLIED / first wrote.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone


def _dt(iso):
    try:
        d = datetime.fromisoformat(str(iso))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def row_date(row: dict, basis: str = "activity"):
    """basis 'activity' -> newest of applied / your last email / their last reply; 'applied' -> applied date."""
    if basis == "applied":
        return _dt(row.get("applied_at"))
    dates = [d for d in (_dt(row.get("applied_at")), _dt(row.get("last_contact_at")), _dt(row.get("last_reply_at"))) if d]
    return max(dates) if dates else None


def custom_window(start: date, end: date) -> tuple[datetime, datetime]:
    """Any range (both days inclusive) in the user's timezone. Unlike scans, the board has no 1-year limit."""
    if start > end:
        start, end = end, start
    frm = datetime.combine(start, time.min).astimezone()
    to = datetime.combine(end + timedelta(days=1), time.min).astimezone()
    return frm, to


def in_window(row: dict, frm: datetime | None, to: datetime | None, basis: str = "activity") -> bool:
    if frm is None and to is None:
        return True  # "All time"
    d = row_date(row, basis)
    return d is not None and (frm is None or d >= frm) and (to is None or d < to)


def kind_matches(row: dict, kind: str) -> bool:
    """kind: 'all' | 'application' | 'cold'."""
    return kind == "all" or (row.get("outreach_kind") or "application") == kind