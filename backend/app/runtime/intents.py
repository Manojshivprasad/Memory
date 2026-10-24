"""Cheap intent lexicons for two structured recall channels (no model calls).

* Attribute channel: questions about a known attribute ("which company do I work for?") map to the
  canonical predicates the write path stores (works_at, lives_in, ...). Small embedding models rate
  "company" vs "I work at Swiggy" no higher than unrelated facts; the predicate is exact.
* Standing constraints: allergies, intolerances, injuries and "never/avoid" rules are tagged with the
  domains they govern when stored; any request in that domain recalls them ("dinner tonight" -> peanut
  allergy), the way a person always remembers a friend's allergy when cooking for them.
"""

import re

# canonical predicate -> question patterns
ATTRIBUTE_TRIGGERS: dict[str, re.Pattern] = {k: re.compile(v, re.I) for k, v in {
    "works_at": r"\b(company|employer|work (?:for|at)|where do i work|office|my job)\b",
    "job_title": r"\b(my (?:job|role|title|position)|what do i do for (?:a )?living|job title)\b",
    "lives_in": r"\b(where do i live|where i live|my (?:city|home|address)|live these days|am i based|living now)\b",
    "name": r"\b(my name|what(?:'s| is) my name|call me|who am i)\b",
    "exam_date": r"\b(exam|test date|when is my (?:exam|test))\b",
    "deadline": r"\b(deadline|due date|when is .* due)\b",
    "budget": r"\b(budget|how much can i spend)\b",
    "monthly_investment": r"\b(sip|invest(?:ing|ment)? (?:each|every|per|a) month|monthly investment)\b",
    "weekly_distance": r"\b(running (?:per|a|each) week|weekly (?:mileage|distance|km)|how much am i running)\b",
    "goal_time": r"\b(goal time|target time)\b",
    "protagonist_name": r"\b(main character|protagonist|hero(?:ine)?)\b",
    "current_focus": r"\b(practi[cs]e (?:today|now|next)|work on (?:today|now|next)|focus (?:on|today)|weak spot)\b",
    "uses_database": r"\b(database|db)\b",
    "uses_framework": r"\b(framework|backend (?:stack|framework))\b",
}.items()}

# domains a request can touch
DOMAINS: dict[str, re.Pattern] = {k: re.compile(v, re.I) for k, v in {
    "food": r"\b(eat|eating|food|snack|meal|dinner|lunch|breakfast|brunch|recipe|cook\w*|bake|restaurant|dish|"
            r"menu|drink|dessert|grocer\w*|nutrition|diet|protein|smoothie|order (?:food|in))\b",
    "exercise": r"\b(run|running|jog\w*|workout|exercise|training|train|gym|race|marathon|hike|hiking|stretch\w*|"
                r"squat|lift\w*|cardio|sport|cycling|swim\w*|long run|recovery)\b",
    "travel": r"\b(trip|travel|flight|hotel|itinerary|train|bus|visit|tour)\b",
    "money": r"\b(buy|spend|invest\w*|budget|save|saving|crypto|stock|loan|price|cost|afford)\b",
}.items()}

# what makes a memory a standing constraint, and which domains it implies even if not spelled out
CONSTRAINT = re.compile(
    r"\b(allerg\w*|intoleran\w*|celiac|coeliac|gluten[- ]free|vegan|vegetarian|halal|kosher|diabet\w*|"
    r"injur\w*|sprain\w*|pain|surgery|pregnan\w*|avoid\w*|never|must not|can'?t (?:eat|have|drink|run))\b",
    re.I,
)
_IMPLIED = [
    (re.compile(r"\b(allerg\w*|intoleran\w*|celiac|coeliac|gluten|vegan|vegetarian|halal|kosher|diabet\w*)\b", re.I),
     "food"),
    (re.compile(r"\b(injur\w*|knee|ankle|back pain|sprain\w*|surgery|downhill)\b", re.I), "exercise"),
]


# Preferences about HOW the assistant should respond apply to every request (a standing instruction).
RESPONSE_STYLE = re.compile(
    r"\b(answers?|respon\w*|repl\w*|format\w*|bullet\w*|concise|brief|short|verbose|tone|explain\w*|"
    r"language|emoji\w*|markdown|examples?)\b",
    re.I,
)
ALWAYS = "always"


_RELATION_Q = re.compile(r"\bwho(?:'s| is| are)\s+(?:my|our|his|her|their|\w+'s)\s+([a-z]+)", re.I)


def query_attributes(query: str) -> list[str]:
    preds = [pred for pred, rx in ATTRIBUTE_TRIGGERS.items() if rx.search(query)]
    # "who is Aarav's teacher?" -> the relation itself is the attribute (teacher / teacher_name)
    for m in _RELATION_Q.finditer(query):
        rel = m.group(1).lower()
        preds += [rel, f"{rel}_name", f"has_{rel}"]
    return preds


def _topic_domains(text: str) -> list[str]:
    return [d for d, rx in DOMAINS.items() if rx.search(text)]


def query_domains(query: str) -> list[str]:
    """Domains a request touches. Every request touches ALWAYS (standing response instructions)."""
    return _topic_domains(query) + [ALWAYS]


def constraint_domains(content: str, kind: str = "fact") -> list[str] | None:
    """Domains a memory constrains, or None if it is not a standing constraint."""
    if kind == "preference" and RESPONSE_STYLE.search(content):
        return [ALWAYS]  # "keep answers short, use bullets" applies to every request
    if not CONSTRAINT.search(content):
        return None
    doms = {d for rx, d in _IMPLIED if rx.search(content)} | set(_topic_domains(content))
    return sorted(doms) or ["general"]
