"""Preference reversals: the same topic with the opposite attitude.

"The user enjoys playing chess" (+chess) vs "The user hates chess" / "prefers cricket instead of chess" (-chess).
The newer attitude wins; the older memory is retired (kept as history). Momentary states ("not playing chess
today") never retire lasting preferences.
"""

import re

from app.textutil import STOPWORDS, stem, words

_NEG = re.compile(
    r"^(not|no|never|don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|wasn'?t|won'?t|cannot|can'?t|hate[sd]?|hating|"
    r"dislike[sd]?|stop|stopped|quit|avoid|avoids|without|longer)$",
    re.I,
)
_INSTEAD = re.compile(r"\b(instead of|rather than|over|than)\s+(?:\w+\s+){0,2}$", re.I)
TRANSIENT = re.compile(
    r"\b(today|tonight|right now|at the moment|this (?:morning|evening|week|weekend)|currently feeling|"
    r"for now|feeling)\b",
    re.I,
)
# words that carry attitude or grammar, not the topic itself
_GENERIC = {
    "user", "want", "like", "love", "enjoy", "prefer", "hate", "dislike", "play", "interest", "improv",
    "discuss", "feel", "go", "try", "really", "project", "instead", "rather", "anymore", "longer", "skill",
    "talk", "doe", "not", "keep", "much", "more", "less", "thing", "stuff", "time", "day", "today",
}
WINDOW = 6


def topic_stems(text: str) -> set[str]:
    return {stem(w) for w in words(text) if len(w) >= 3 and w not in STOPWORDS and stem(w) not in _GENERIC}


def polarity(text: str, topic: str) -> int:
    """+1 / -1 attitude toward ``topic`` (a stem) in ``text``; 0 if the topic isn't mentioned."""
    toks = words(text)
    for i, w in enumerate(toks):
        if stem(w) != topic:
            continue
        before = toks[max(0, i - WINDOW):i]
        if any(_NEG.match(b) for b in before):
            return -1
        if _INSTEAD.search(" ".join(toks[max(0, i - 4):i]) + " "):
            return -1
        return 1
    return 0


def contradicts(new: str, old: str) -> str | None:
    """The shared topic on which ``new`` reverses ``old``, or None."""
    if TRANSIENT.search(new):
        return None
    for t in topic_stems(new) & topic_stems(old):
        p_new, p_old = polarity(new, t), polarity(old, t)
        if p_new and p_old and p_new != p_old:
            return t
    return None
