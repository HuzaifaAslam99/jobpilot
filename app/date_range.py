"""Which period of the inbox to scan: presets (today, last 3/7 days, last 1-12 months) or a custom range.

The first scan of a big inbox is slow on free AI tiers, so the user picks how far back to go.
Everything becomes a Gmail search like "after:1719792000 before:1727308800" (Unix seconds, so the
boundaries are exact in the user's own timezone instead of Gmail's default Pacific time).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

MAX_DAYS = 366  # "max 1 year" (a leap year has 366 days)

PRESETS = {  # key: (label, unit, amount)
    "today": ("Today", "days", 0),
    "3d": ("Last 3 days", "days", 3),
    "7d": ("Last 7 days", "days", 7),
    "1m": ("Last month", "months", 1),
    "3m": ("Last 3 months", "months", 3),
    "6m": ("Last 6 months", "months", 6),
    "9m": ("Last 9 months", "months", 9),
    "1y": ("Last 1 year (max)", "months", 12),
}


def _months_back(d: date, n: int) -> date:
    y, m = divmod(d.year * 12 + (d.month - 1) - n, 12)
    m += 1
    for day in (d.day, 30, 29, 28):  # 31 March minus 1 month -> 28/29 Feb
        try:
            return date(y, m, day)
        except ValueError:
            continue


def _local(d: date) -> datetime:
    return datetime.combine(d, time.min).astimezone()  # midnight in the user's timezone


def resolve(preset: str = "1y", include_today: bool = True, start: date | None = None, end: date | None = None,
            now: datetime | None = None) -> tuple[datetime, datetime]:
    """-> (from, to) as timezone-aware datetimes; 'to' is exclusive.

    include_today=False ends the window at midnight this morning, so "last 3 days" means the 3 full days before today.
    A custom range (start/end, both inclusive) may be at most one year long and can't be in the future."""
    now = (now or datetime.now()).astimezone()
    today = now.date()
    if start or end:
        start, end = start or end, end or start
        if start > end:
            raise ValueError("The start date is after the end date.")
        if end > today:
            raise ValueError("The range can't include future dates.")
        if (end - start).days + 1 > MAX_DAYS:
            raise ValueError("A custom range can be at most 1 year long.")
        return _local(start), min(_local(end + timedelta(days=1)), now)
    if preset not in PRESETS:
        raise ValueError(f"Unknown range '{preset}'. Choose one of: {', '.join(PRESETS)}")
    _, unit, amount = PRESETS[preset]
    if preset == "today":
        return _local(today), now
    last_day = today if include_today else today - timedelta(days=1)
    if unit == "days":
        first_day = last_day - timedelta(days=amount - 1)
    else:
        first_day = _months_back(last_day, amount) + timedelta(days=1)
    return _local(first_day), (now if include_today else _local(today))


def gmail_query(frm: datetime, to: datetime) -> str:
    return f"after:{int(frm.timestamp())} before:{int(to.timestamp())}"


def describe(frm: datetime, to: datetime) -> str:
    last = (to - timedelta(seconds=1)).date()
    first = frm.date()
    fmt = lambda d: f"{d.day} {d.strftime('%b %Y')}"
    return fmt(first) if first == last else f"{fmt(first)} – {fmt(last)} ({(last - first).days + 1} days)"