"""Due-date parsing for commitments and deadlines ("tomorrow", "by Friday", "Oct 15", "2026-10-15")."""

import re
from datetime import datetime, timedelta

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MD = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
_DM = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)


def _eod(d: datetime) -> datetime:
    return d.replace(hour=23, minute=59, second=0, microsecond=0)


def parse_due(text: str, now: datetime) -> datetime | None:
    t = text.lower()
    if m := _ISO.search(t):
        try:
            return _eod(now.replace(year=int(m[1]), month=int(m[2]), day=int(m[3])))
        except ValueError:
            return None
    for rx, mi, di in ((_MD, 1, 2), (_DM, 2, 1)):
        if m := rx.search(t):
            try:
                d = now.replace(month=_MONTHS[m[mi][:3].lower()], day=int(m[di]))
            except ValueError:
                return None
            if d < now - timedelta(days=1):
                d = d.replace(year=d.year + 1)
            return _eod(d)
    if "tomorrow" in t:
        return _eod(now + timedelta(days=1))
    if re.search(r"\btoday\b|\btonight\b|\beod\b|end of (the )?day", t):
        return _eod(now)
    if "next week" in t:
        return _eod(now + timedelta(days=7 - now.weekday() + 4))  # next week's Friday
    if re.search(r"end of (the )?week", t):
        return _eod(now + timedelta(days=max(0, 4 - now.weekday())))
    for i, wd in enumerate(_WEEKDAYS):
        if re.search(rf"\b{wd}\b", t):
            delta = (i - now.weekday()) % 7 or 7
            return _eod(now + timedelta(days=delta))
    return None
