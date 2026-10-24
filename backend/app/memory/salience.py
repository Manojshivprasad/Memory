"""Selective encoding: how memory-worthy is a turn?

Noisy-OR over detected signals, damped when the content is not novel (already known).
Explicit "remember this" and corrections are never damped: they matter precisely because
they touch something already known.
"""

from app.write.signals import Signals

WEIGHTS = {
    "explicit_remember": 0.9,
    "correction": 0.6,
    "decision": 0.55,
    "self_fact": 0.5,
    "preference": 0.5,
    "goal": 0.45,
    "social": 0.45,
    "requirement": 0.35,
    "event": 0.5,
    "constraint": 0.65,
    "skill": 0.45,
    "relation": 0.5,
    "quantity": 0.3,
    "commitment": 0.45,
    "feedback_neg": 0.45,
    "adoption": 0.4,
    "feedback_pos": 0.4,
    "open_question": 0.35,
    "tech": 0.3,
    "temporal": 0.15,
    "a:recommendation": 0.25,
    "a:plan": 0.15,
    "a:promise": 0.2,
}
# Never damped by "already known": corrections and changes matter precisely because they touch known things,
# and safety constraints must always be (re)checked.
UNDAMPED = {"explicit_remember", "correction", "feedback_pos", "feedback_neg", "adoption", "constraint"}


def salience(signals: Signals, novelty: float) -> float:
    names = signals.all
    miss = 1.0
    for n in names:
        miss *= 1.0 - WEIGHTS.get(n, 0.0)
    base = 1.0 - miss
    if names & UNDAMPED:
        return round(base, 3)
    return round(base * (0.55 + 0.45 * max(0.0, min(1.0, novelty))), 3)
