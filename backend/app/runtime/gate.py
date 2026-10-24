"""Metamemory gate: does this prompt need long-term memory at all?

Deliberately conservative: it only skips prompts that clearly carry no information need
(greetings, acknowledgements, empty). Everything else goes to retrieval, where the strict
relevance floor decides what (if anything) is injected.
"""

import re

from app.textutil import content_tokens

_SKIP = re.compile(
    r"^\s*(hi|hello|hey|yo|hiya|good (morning|afternoon|evening)|thanks|thank you|thx|ty|ok|okay|k|cool|"
    r"great|nice|awesome|perfect|got it|sounds good|yes|no|yep|nope|sure|bye|goodbye|lol|haha)"
    r"[\s!.,?]*(there|so much|a lot|again)?[\s!.,?]*$",
    re.I,
)


_PERSONAL = re.compile(
    r"\b(i|i'm|i've|i'll|me|my|mine|we|we're|we've|we'll|our|ours|us|you|your|yours|let's|project|todo|todos|"
    r"pending|remind|reminder|again|previous|previously|earlier|last time|left off|remember|recall)\b",
    re.I,
)


def is_personal(query: str) -> bool:
    """Does the prompt refer to the user / the project / shared history (vs. a general-knowledge ask)?"""
    return bool(_PERSONAL.search(query))


def should_retrieve(query: str) -> tuple[bool, str]:
    q = query.strip()
    if not q:
        return False, "empty"
    if _SKIP.match(q):
        return False, "phatic"
    if not content_tokens(q):
        return False, "no-content"
    return True, "retrieve"
