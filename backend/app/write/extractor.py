"""Tier-2 extraction: turn -> memory operations.

LLM mode: a small model gets the turn, recent context, the project brief and the memories that
were retrieved for this turn (so it can reconsolidate them) and returns ops as JSON.
Heuristic mode (no LLM configured or LLM failure): deterministic patterns that cover the common
cases (tech stack decisions, deadlines, preferences, identity, commitments, recommendations).
"""

import re
from datetime import datetime

import orjson

from app.llm import json_complete, llm_enabled
from app.textutil import detect_tech, split_clauses
from app.write.dates import parse_due
from app.write.ops import Op
from app.write.signals import Signals

SYSTEM = """You maintain the long-term memory of an AI assistant. From the LATEST turn, output memory
operations as JSON: {"ops": [...]}.

CAPTURE, thoroughly, everything durable the USER says about themselves and their world:
- identity & life: name, where they live, job/employer/title, people in their life (name + relationship/role),
  pets, things they own that matter;
- dated events & deadlines: exams, interviews, birthdays, trips, races, launches (keep the date; resolve
  relative dates like "Friday" to YYYY-MM-DD using `today`);
- numbers that matter: budgets, amounts, distances, levels, scores, counts ("max 3 problems per set");
- constraints: allergies, intolerances, injuries, things to avoid or never do;
- preferences, including how the assistant should answer them;
- skills & progress: strengths, weaknesses, what they are working on now;
- project facts: stack, decisions, who owns what, open questions, commitments ("I'll send X by Friday");
- details of the work itself: a novel's title, characters, setting and plot choices; a course's syllabus;
  a trip's itinerary; a codebase's components. These are facts too (subj = the thing described, e.g. "novel").
DO NOT capture: greetings, small talk, jokes, weather, general-knowledge Q&A (capitals, definitions),
momentary happenings that won't matter tomorrow ("my cat just knocked over a glass", "I spilled coffee"),
or anything the assistant merely explained, agreed with, or offered ("I can help you plan...").

Op shapes:
  {"op":"ADD","kind":"fact|preference|decision|goal|commitment|open_question|outcome|entity_relation",
   "content":"...","subj":"user|project|<entity>","pred":"<snake_case>","obj":"<value>",
   "importance":0..1,"confidence":0..1,"scope":"project|user","speaker":"user|assistant",
   "due_at":"YYYY-MM-DD|null","entities":["<named people/things/places, <=5>"],"evidence_span":"<short quote>"}
  {"op":"UPDATE","memory_id":"...","content":"<full new content>","reason":"..."}
  {"op":"REINFORCE","memory_id":"..."}
  {"op":"INVALIDATE","memory_id":"...","reason":"..."}
  {"op":"CLOSE","memory_id":"...","result":"..."}
  {"op":"OUTCOME","memory_id":"...|null","result":"success|failure","lesson":"..."}

RULES
1. ONE atomic fact per ADD. Split "I live in X and work at Y" into two ADDs.
2. content = present-tense statement with an explicit subject, <= 20 words, names/numbers/dates verbatim,
   e.g. "The user lives in Bangalore." / "Aarav's teacher is Mrs. Rao." Use subj "user" ONLY for the person
   talking; other people and things are their own subject ("Alice owns payments" -> subj "Alice").
3. CHANGES: if the turn changes something in existing_memories (moved, switched, renamed, increased, now,
   no longer...), output INVALIDATE for EVERY existing memory it makes outdated, plus ONE ADD stating the new
   current value. Never add a separate "switched from X to Y" memory. A reversed attitude ("I love chess" ->
   "I actually hate chess") retires EVERY existing memory expressing the old attitude toward that topic
   (loves it, is interested in it, wants to discuss it...). Moods about today don't reverse lasting preferences.
4. Refinement of an existing memory -> UPDATE (full new content). Plain restatement -> REINFORCE.
5. pred: snake_case attribute name. For single-valued attributes reuse the same pred every time, e.g. lives_in,
   works_at, job_title, name, uses_database, uses_framework, deadline, exam_date, event_date, budget,
   monthly_investment, weekly_distance, goal_time, protagonist_name, current_focus, weakness.
6. speaker="assistant" ONLY for a concrete proposal the assistant made for this project (a tool, approach,
   setting): kind "decision", content "Assistant suggestion: ...". Never store assistant facts or offers.
7. scope="user" only for personal facts true in every project (name, home, job, family, pets, allergies, health,
   diet) and preferences the user says apply everywhere ("always", "in general", "for every project").
   Tasks, commitments, decisions, goals and anything about this project -> scope "project".
8. importance: safety/health constraints 0.9; dates, deadlines, decisions, goals 0.75; preferences 0.6; other 0.5.
9. Every ADD MUST have a non-empty "content".
10. Resolve pronouns (she, he, it, they, his, her) to the person/thing from recent_conversation, e.g.
    "She loves pottery" after talking about Priya -> "Priya loves pottery."
Refer to existing memories only by the ids given. Return {"ops": []} when nothing qualifies.

EXAMPLE 1 (several atomic facts from one sentence)
today: 2026-09-28 (Monday)
user: "My daughter Riya starts violin lessons on Oct 3; her teacher is Mr. Das. Keep practice tips short."
-> {"ops":[
 {"op":"ADD","kind":"fact","content":"The user's daughter is Riya.","subj":"user","pred":"daughter","obj":"Riya","scope":"user","speaker":"user","importance":0.6,"confidence":0.95,"entities":["Riya"]},
 {"op":"ADD","kind":"fact","content":"Riya starts violin lessons on 2026-10-03.","subj":"Riya","pred":"event_date","obj":"2026-10-03","due_at":"2026-10-03","scope":"project","speaker":"user","importance":0.75,"confidence":0.95,"entities":["Riya"]},
 {"op":"ADD","kind":"entity_relation","content":"Riya's violin teacher is Mr. Das.","subj":"Riya","pred":"teacher","obj":"Mr. Das","scope":"project","speaker":"user","importance":0.5,"confidence":0.95,"entities":["Riya","Mr. Das"]},
 {"op":"ADD","kind":"preference","content":"The user wants practice tips kept short.","subj":"user","pred":"tip_length","obj":"short","scope":"project","speaker":"user","importance":0.6,"confidence":0.9,"entities":[]}]}

EXAMPLE 2 (a change: retire the old value, state the new one)
existing_memories: [{"id":"m7","content":"The user's monthly rent is 25,000 INR.","pred":"monthly_rent"}]
user: "Rent went up, it's 28,000 now."
-> {"ops":[
 {"op":"INVALIDATE","memory_id":"m7","reason":"rent increased"},
 {"op":"ADD","kind":"fact","content":"The user's monthly rent is 28,000 INR.","subj":"user","pred":"monthly_rent","obj":"28,000 INR","scope":"project","speaker":"user","importance":0.6,"confidence":0.95,"entities":[]}]}

EXAMPLE 3 (nothing to keep)
user: "haha nice, what's the capital of Peru?"  assistant: "Lima!"
-> {"ops":[]}"""


def _mem_view(m: dict) -> dict:
    return {k: m.get(k) for k in ("id", "kind", "content", "subj", "pred", "obj", "scope", "is_belief",
                                  "thread_status")}


async def extract_llm(user_msg: str, assistant_msg: str, existing: list[dict], brief: str, recent: str,
                      now: datetime) -> list[Op] | None:
    if not llm_enabled():
        return None
    payload = {
        "today": now.strftime("%Y-%m-%d (%A)"),
        "recent_conversation": recent,
        "existing_memories": [_mem_view(m) for m in existing][:24],
        "turn": {"user": user_msg[:6000], "assistant": assistant_msg[:6000]},
    }
    data = await json_complete(SYSTEM, orjson.dumps(payload, default=str).decode())
    if data is None or not isinstance(data.get("ops"), list):
        return None
    ops = [o for o in (Op.from_dict(d) for d in data["ops"] if isinstance(d, dict)) if o is not None]
    for o in ops:  # LLMs are unreliable at calendar arithmetic: prefer the deterministic parser when it can read it
        if o.op == "ADD" and (o.due_at or o.kind in ("commitment", "goal") or o.pred in ("deadline", "launch_date")):
            parsed = parse_due(o.evidence or "", now) or parse_due(o.content, now)
            if parsed:
                o.due_at = parsed
    return ops


# ------------------------------------------------------------------ heuristic fallback

_PREF = re.compile(r"\bi (?:really )?(prefer|like|love|hate|dislike|don'?t like|do not like|want)\s+(.+)", re.I)
_ALWAYS = re.compile(r"\b(?:please )?(always|never)\s+(.+)", re.I)
_NAME = re.compile(r"\bmy name is\s+([A-Z][\w-]+(?:\s+[A-Z][\w-]+)?)", re.I)
_LIVE = re.compile(r"\bi (?:live|am based|'m based|moved|relocated) (?:in|to)\s+([A-Z][\w\s-]{1,40}?)(?:[.,;!]|$| last| this| now)", re.I)
_WORK = re.compile(r"\bi work (?:at|for)\s+([A-Z][\w&.\s-]{1,40}?)(?:[.,;!]|$| as| on)", re.I)
_ROLE = re.compile(r"\bi(?:'m| am) (?:a|an|the)\s+([\w\s-]{2,40}?(?:engineer|developer|designer|manager|founder|"
                   r"lead|scientist|analyst|student|architect|cto|ceo))\b", re.I)
_DEADLINE = re.compile(r"\b(?:deadline|due date|launch(?:ing)?(?: date)?)\s+(?:is|:|on|by)?\s*(.+)", re.I)
_REMEMBER = re.compile(r"\b(?:remember|note|keep in mind)(?: that)?[:,]?\s+(.+)", re.I)
_COMMIT = re.compile(r"\bi(?:'ll| will| need to| have to)\s+(.+)", re.I)
_OPENQ = re.compile(r"\b(?:still need to (?:figure out|decide)|not sure (?:yet )?(?:whether|if)|haven'?t decided)\s+(.+)", re.I)
_GOAL = re.compile(r"\b(?:(?:our|the|my) goal is|we want to|we need to|i want to build|the plan is)\s+(?:to\s+)?(.+)", re.I)
_SOCIAL = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?) (owns|leads|manages|runs|wants|needs|prefers|approved|handles|"
                     r"is responsible for|works on|reports to)\b(.+)")
_RECO = re.compile(r"([^.!?\n]*\b(?:i recommend|i'?d recommend|i suggest|i'?d suggest|you should|the best (?:option|"
                   r"approach|choice) (?:is|would be)|i would go with)\b[^.!?\n]*[.!?]?)", re.I)


_VERB3 = {"prefer": "prefers", "like": "likes", "love": "loves", "hate": "hates", "dislike": "dislikes",
          "dont like": "doesn't like", "do not like": "doesn't like", "want": "wants"}


def _clean(s: str, n: int = 160) -> str:
    return s.strip().rstrip(".!?,;: ")[:n]


def extract_heuristic(user_msg: str, assistant_msg: str, signals: Signals, now: datetime) -> list[Op]:
    ops: list[Op] = []
    for clause in split_clauses(user_msg):
        c = clause.strip()
        low = c.lower()
        techs = detect_tech(c) if re.search(r"\b(we|our|i|my|us|let'?s|project)\b", low) else []
        # "switched from Postgres to MySQL": only the target is current
        techs = [t for t in techs if not re.search(
            rf"\b(from|instead of|replac\w*|no longer( use| using)?|not|dropp\w*|away from)\s+{re.escape(t[1])}\b", low)]
        if techs and not re.search(r"\b(should we|could we|what about|vs\.?|versus|or)\b", low):
            for cat, name in techs:
                ops.append(Op(op="ADD", kind="decision", content=f"The project uses {name} ({cat}).",
                              subj="project", pred=f"uses_{cat}", obj=name, importance=0.75, confidence=0.85,
                              evidence=c, entities=[name]))
        if m := _DEADLINE.search(c):
            due = parse_due(m.group(1), now)
            ops.append(Op(op="ADD", kind="fact", content=f"Project deadline: {_clean(m.group(1), 60)}.",
                          subj="project", pred="deadline", obj=_clean(m.group(1), 60), importance=0.8,
                          confidence=0.85, evidence=c, due_at=due))
        if m := _NAME.search(c):
            ops.append(Op(op="ADD", kind="fact", content=f"The user's name is {m.group(1)}.", subj="user",
                          pred="name", obj=m.group(1), importance=0.7, scope="user", evidence=c))
        if m := _LIVE.search(c):
            place = _clean(m.group(1), 60)
            ops.append(Op(op="ADD", kind="fact", content=f"The user lives in {place}.", subj="user",
                          pred="lives_in", obj=place, importance=0.6, scope="user", evidence=c, entities=[place]))
        if m := _WORK.search(c):
            org = _clean(m.group(1), 60)
            ops.append(Op(op="ADD", kind="fact", content=f"The user works at {org}.", subj="user", pred="works_at",
                          obj=org, importance=0.6, scope="user", evidence=c, entities=[org]))
        if m := _ROLE.search(c):
            ops.append(Op(op="ADD", kind="fact", content=f"The user is a {_clean(m.group(1), 60)}.", subj="user",
                          pred="role", obj=_clean(m.group(1), 60), importance=0.6, scope="user", evidence=c))
        if m := _PREF.search(c):
            verb, what = _VERB3.get(m.group(1).lower().replace("'", ""), "prefers"), _clean(m.group(2))
            if what and not techs:
                ops.append(Op(op="ADD", kind="preference", content=f"The user {verb} {what}.",
                              subj="user", pred="prefers", obj=what, importance=0.6, evidence=c))
        elif m := _ALWAYS.search(c):
            ops.append(Op(op="ADD", kind="preference", content=f"The user wants the assistant to {m.group(1).lower()} "
                          f"{_clean(m.group(2))}.", subj="user", pred="instruction", obj=_clean(m.group(2)),
                          importance=0.65, evidence=c))
        if (m := _SOCIAL.search(c)) and m.group(1) not in ("I", "We", "The", "This", "That", "It"):
            ops.append(Op(op="ADD", kind="fact", content=_clean(c, 240) + ".", subj=m.group(1),
                          pred=m.group(2).replace(" ", "_"), obj=_clean(m.group(3), 80), importance=0.6,
                          evidence=c, entities=[m.group(1)]))
        if m := _REMEMBER.search(c):
            ops.append(Op(op="ADD", kind="fact", content=_clean(m.group(1), 240) + ".", importance=0.75,
                          confidence=0.9, evidence=c))
        if m := _GOAL.search(c):
            if not techs:
                ops.append(Op(op="ADD", kind="goal", content=f"Goal: {_clean(m.group(1), 200)}.", subj="project",
                              pred="goal", importance=0.7, evidence=c))
        if m := _OPENQ.search(c):
            ops.append(Op(op="ADD", kind="open_question", content=f"Open question: {_clean(m.group(1), 200)}.",
                          importance=0.55, evidence=c))
        elif (m := _COMMIT.search(c)) and "commitment" in signals.user:
            ops.append(Op(op="ADD", kind="commitment", content=f"The user will {_clean(m.group(1), 200)}.",
                          importance=0.6, evidence=c, due_at=parse_due(c, now)))
    if "recommendation" in signals.assistant:
        if m := _RECO.search(assistant_msg):
            sentence = _clean(m.group(1), 240)
            ops.append(Op(op="ADD", kind="decision", content=f"Assistant suggestion: {sentence}.", speaker="assistant",
                          importance=0.4, confidence=0.55, evidence=sentence,
                          entities=[n for _, n in detect_tech(sentence)]))
    # de-duplicate identical contents within the turn
    seen: set[str] = set()
    uniq = []
    for o in ops:
        if o.content not in seen:
            seen.add(o.content)
            uniq.append(o)
    return uniq


def outcome_ops(signals: Signals, prev_assistant_memories: list[dict], user_msg: str) -> list[Op]:
    """Adoption / feedback about the PREVIOUS assistant turn (action -> result learning)."""
    ops: list[Op] = []
    targets = prev_assistant_memories[:3]
    if not targets:
        return ops
    if "adoption" in signals.user:
        for m in targets:
            if m.get("is_belief"):
                ops.append(Op(op="REINFORCE", memory_id=m["id"], speaker="user", evidence=_clean(user_msg, 200)))
    if "feedback_pos" in signals.user or "feedback_neg" in signals.user:
        ok = "feedback_pos" in signals.user and "feedback_neg" not in signals.user
        m = targets[0]
        what = m["content"].removeprefix("Assistant suggestion: ")
        ops.append(Op(op="OUTCOME", memory_id=m["id"], result="success" if ok else "failure",
                      lesson=(f"Worked: {what}" if ok else f"Did not work: {what}")[:300],
                      evidence=_clean(user_msg, 200)))
    return ops
