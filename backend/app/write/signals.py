"""Tier-1 signal detection (regex, ~0.1 ms): what kind of memory-worthy content is in a turn?"""

import re
from dataclasses import dataclass, field

from app.textutil import detect_tech

_USER = {
    "explicit_remember": r"\b(remember|don'?t forget|note that|keep in mind|for future reference|fyi)\b",
    "preference": r"\bi (?:really )?(?:prefer|like|love|hate|dislike|don'?t like|do not like)\b|"
                  r"\b(?:please )?(?:always|never) (?:use|do|give|write|answer|respond|include)\b",
    "self_fact": r"\b(?:my (?:name|role|job|title|team|company|email|timezone) is|i am an?|i'?m an?|i live|i work|"
                 r"i moved|i relocated|i'?m based)\b",
    "decision": r"\b(?:let'?s (?:go with|use|stick with|switch to|move to)|we(?:'ll| will)? (?:go with|use|chose|picked|"
                r"decided)|decided (?:to|on)|we(?:'re| are) (?:using|going with|building|switching)|we switched|"
                r"we moved to|we migrated|i chose|i picked|we use|we are on)\b",
    "commitment": r"\b(?:i(?:'ll| will) (?!be\b)\w+|i need to|i have to|remind me|to-?do)\b",
    "goal": r"\b(?:(?:our|the|my) goal|the aim|objective is|we want to|we need to|i want to build|trying to build|"
            r"the plan is|deadline|launch(?:ing)? (?:on|by|in))\b",
    "correction": r"\b(?:actually|correction|not anymore|no longer|instead of|i meant|that'?s (?:wrong|not right)|"
                  r"scratch that|changed to|switched (?:to|from)|moved to|migrated to|renamed|increased|decreased|"
                  r"raised|reduced|upgraded|downgraded|update:|big news|now (?:i|we|it|he|she|my|our)?\s?"
                  r"(?:am|is|are|use|live|work|run|have|struggl\w*))\b",
    "open_question": r"\b(?:still need to (?:figure out|decide)|open question|not sure (?:yet|whether|if)|tbd|"
                     r"to be decided|haven'?t decided)\b",
    "feedback_pos": r"\b(?:that worked|it worked|it works|works now|working now|worked perfectly|fixed it|that fixed|"
                    r"problem solved|solved it|all good now)\b",
    "feedback_neg": r"\b(?:didn'?t work|doesn'?t work|still (?:failing|broken|getting|not working|fails)|same error|"
                    r"not working|that broke|made it worse|still see)\b",
    "adoption": r"\b(?:let'?s do (?:that|it|this|option)|sounds good|go ahead|do it|go with (?:that|your|option)|"
                r"agreed|approved|ship it|i'?ll take (?:that|option))\b",
    "temporal": r"\b(?:today|tomorrow|yesterday|next (?:week|month)|this (?:week|month)|by (?:mon|tues|wednes|thurs|"
                r"fri|satur|sun)day|deadline|due)\b",
    # dated life events: exams, interviews, birthdays, trips, races...
    "event": r"\b(?:exam|test|interview|birthday|anniversary|appointment|meeting|flight|trip|race|marathon|wedding|"
             r"launch|recital|match|surgery|visit|holiday|vacation)s?\b",
    # safety / health / hard constraints
    "constraint": r"\b(?:allerg\w*|intoleran\w*|injur\w*|diabet\w*|pregnan\w*|vegan|vegetarian|halal|kosher|"
                  r"can'?t (?:eat|have|drink)|avoid|never (?:use|eat|suggest))\b",
    # skills and progress
    "skill": r"\b(?:strong (?:in|at)|weak (?:in|at|spot)|good at|bad at|struggl\w*|confus\w*|my weakness|"
             r"level|scored|progress)\b",
    # people & pets in the user's life
    "relation": r"\b(?:my|our) (?:wife|husband|partner|son|daughter|kid|child|sister|brother|mom|mother|dad|"
                r"father|friend|boss|manager|teacher|tutor|colleague|dog|cat|pet|landlord|doctor|coach)\b",
    # quantities that matter (money, distance, time, counts)
    "quantity": r"\b\d[\d,.]*\s?(?:k|km|kg|mi|miles|inr|usd|eur|rs|lakh|crore|%|minutes?|mins?|hours?|hrs?|"
                r"per (?:day|week|month|set)|a (?:day|week|month))\b|[$₹€£]\s?\d",
}
_ASSISTANT = {
    "recommendation": r"\b(?:i recommend|i'?d recommend|i suggest|i'?d suggest|you should|consider using|"
                      r"the best (?:option|approach|choice) (?:is|would be)|i would (?:use|go with)|recommended approach)\b",
    "plan": r"\b(?:here'?s (?:the|a) plan|step 1\b|next steps)\b",
    "promise": r"\b(?:i'?ll remember|i will remember|noted[.,!])",
}
_USER_RX = {k: re.compile(v, re.I) for k, v in _USER.items()}
# Social memory: facts about named people (case-sensitive: needs a capitalised name).
_SOCIAL = re.compile(
    r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)? (?:owns|leads|manages|runs|wants|needs|prefers|said|asked|approved|handles|"
    r"is responsible for|works on|reports to|is (?:our|the|my) )"
)
_REQUIREMENT = re.compile(r"\b(must|has to|have to|needs to|required|requirement|should always|should never)\b", re.I)
_ASSISTANT_RX = {k: re.compile(v, re.I) for k, v in _ASSISTANT.items()}
_FIRST_PERSON = re.compile(r"\b(i|we|our|my|us)\b", re.I)


@dataclass
class Signals:
    user: set[str] = field(default_factory=set)
    assistant: set[str] = field(default_factory=set)
    tech: list[tuple[str, str]] = field(default_factory=list)

    @property
    def all(self) -> set[str]:
        return self.user | {f"a:{s}" for s in self.assistant}


def detect(user_msg: str, assistant_msg: str) -> Signals:
    s = Signals()
    for name, rx in _USER_RX.items():
        if rx.search(user_msg):
            s.user.add(name)
    if _SOCIAL.search(user_msg):
        s.user.add("social")
    if _REQUIREMENT.search(user_msg):
        s.user.add("requirement")
    for name, rx in _ASSISTANT_RX.items():
        if rx.search(assistant_msg):
            s.assistant.add(name)
    if _FIRST_PERSON.search(user_msg):
        s.tech = detect_tech(user_msg)
        if s.tech:
            s.user.add("tech")
    return s


# Requests *to* the assistant (questions, commands) rarely carry facts about the user's world.
_ASK = re.compile(
    r"^\s*(?:what|how|why|when|where|who|which|can|could|should|would|is|are|do|does|did|will|explain|tell me|"
    r"give me|write|draft|make|show|list|summari[sz]e|translate|define)\b",
    re.I,
)
_SELF = re.compile(r"\b(?:i|i'm|im|i've|i'll|my|me|we|we're|our|us)\b", re.I)


def informative(user_msg: str) -> bool:
    """A declarative statement with substance (worth showing the extractor even without a keyword signal).

    "The protagonist is Elena, a lighthouse keeper." -> True; "what's the capital of Peru?" -> False.
    """
    from app.textutil import content_tokens

    t = user_msg.strip()
    if len(content_tokens(t)) < 3:
        return False
    if _SELF.search(t) and len(content_tokens(t)) >= 4:
        return True
    # judge sentence by sentence: "She loves pottery. Maybe a pottery class?" has a declarative fact
    for sent in re.split(r"(?<=[.!?;])\s+", t):
        s = sent.strip()
        if not s or s.endswith("?") or _ASK.match(s):
            continue
        if len(content_tokens(s)) >= 2:
            return True
    return False
