"""Temporal cue parsing for queries: "yesterday", "last week", "3 days ago", "before we switched"."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class TemporalCue:
    start: datetime | None = None
    end: datetime | None = None
    historical: bool = False   # include superseded (no longer valid) memories
    last_chat: bool = False    # "where we left off", "last time", "previous chat"

    @property
    def has_window(self) -> bool:
        return self.start is not None


_HISTORICAL = re.compile(
    r"\b(used to|previously|originally|at first|back then|in the past|before we|before i|"
    r"what did we (?:use|have) before|earlier version|old(?:er)? (?:setup|approach|version))\b",
    re.I,
)
_LAST_CHAT = re.compile(
    r"\b(left off|last time|last chat|previous chat|previous conversation|last conversation|"
    r"continue (?:from|where)|pick up where|we discussed|we talked about|you said earlier)\b",
    re.I,
)
_AGO = re.compile(r"\b(\d+|a|an|one|two|three|four|five|six|seven)\s+(day|week|month)s?\s+ago\b", re.I)
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def _day_start(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def parse(query: str, now: datetime) -> TemporalCue:
    q = query.lower()
    historical = bool(_HISTORICAL.search(q))
    last_chat = bool(_LAST_CHAT.search(q))
    start = end = None
    today = _day_start(now)
    if "yesterday" in q:
        start, end = today - timedelta(days=1), today
    elif re.search(r"\btoday\b|\bthis morning\b", q):
        start, end = today, now
    elif "last week" in q:
        start, end = today - timedelta(days=today.weekday() + 7), today - timedelta(days=today.weekday())
    elif "this week" in q:
        start, end = today - timedelta(days=today.weekday()), now
    elif "last month" in q:
        first = today.replace(day=1)
        start, end = (first - timedelta(days=1)).replace(day=1), first
    elif "this month" in q:
        start, end = today.replace(day=1), now
    elif m := _AGO.search(q):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1).lower()]
        days = {"day": 1, "week": 7, "month": 30}[m.group(2).lower()] * n
        center = now - timedelta(days=days)
        pad = timedelta(days=max(1, days // 4))
        start, end = center - pad, center + pad
    # A window entirely in the past asks about what was true *then*: include superseded memories.
    if end is not None and end <= today:
        historical = True
    return TemporalCue(start=start, end=end, historical=historical, last_chat=last_chat)
