from datetime import datetime, timezone

from app.memory.salience import salience
from app.write.dates import parse_due
from app.write.extractor import extract_heuristic, outcome_ops
from app.write.ops import Op, validate
from app.write.signals import detect

NOW = datetime(2026, 9, 28, 10, tzinfo=timezone.utc)  # a Monday


def _extract(user: str, assistant: str = "Sure.") -> list[Op]:
    return extract_heuristic(user, assistant, detect(user, assistant), NOW)


def test_stack_and_deadline_extraction():
    ops = _extract("We're building the API in FastAPI with Postgres; deadline is Oct 15.")
    triples = {(o.subj, o.pred, o.obj) for o in ops}
    assert ("project", "uses_database", "postgres") in triples
    assert ("project", "uses_framework", "fastapi") in triples
    deadline = next(o for o in ops if o.pred == "deadline")
    assert deadline.due_at and deadline.due_at.month == 10 and deadline.due_at.day == 15


def test_switch_produces_functional_predicate():
    ops = _extract("Actually we switched to MySQL.")
    assert any(o.pred == "uses_database" and o.obj == "mysql" for o in ops)


def test_questions_about_tech_are_not_decisions():
    assert not any(o.pred == "uses_database" for o in _extract("Should we use MySQL or Postgres?"))


def test_identity_goes_to_user_scope():
    ops = _extract("My name is Manoj and I live in Bangalore.")
    assert {o.pred for o in ops if o.scope == "user"} >= {"name", "lives_in"}


def test_assistant_recommendation_is_a_belief():
    ops = _extract("How should I cache sessions?", "I recommend using Redis with a 30 minute TTL.")
    rec = [o for o in ops if o.speaker == "assistant"]
    assert rec and validate(rec, set())[0].confidence <= 0.6


def test_adoption_and_feedback_target_previous_suggestion():
    prev = [{"id": "m1", "content": "Assistant suggestion: use Redis", "is_belief": True}]
    ops = outcome_ops(detect("Sounds good, let's do that", ""), prev, "Sounds good, let's do that")
    assert [(o.op, o.memory_id, o.speaker) for o in ops] == [("REINFORCE", "m1", "user")]
    ops = outcome_ops(detect("That worked, thanks!", ""), prev, "That worked, thanks!")
    assert ops[0].op == "OUTCOME" and ops[0].result == "success" and ops[0].lesson.startswith("Worked")
    ops = outcome_ops(detect("Still failing with the same error", ""), prev, "Still failing")
    assert ops[0].result == "failure"


def test_validate_guards():
    ops = [
        Op(op="REINFORCE", memory_id="unknown"),                          # unknown id -> dropped
        Op(op="UPDATE", memory_id="unknown", content="New info"),         # unknown id -> becomes ADD
        Op(op="ADD", content="weak", confidence=0.1),                     # too uncertain -> dropped
        Op(op="ADD", content="x", kind="nonsense", subj="We"),            # kind/subj normalised
        Op(op="NOOP"),
    ]
    out = validate(ops, allowed_ids={"known"})
    assert [(o.op, o.kind) for o in out] == [("ADD", "fact"), ("ADD", "fact")]
    assert out[1].subj == "project"


def test_salience_gates_chitchat():
    chit = salience(detect("lol nice, what's the weather like", "It's sunny."), novelty=1.0)
    decision = salience(detect("Let's go with Postgres for the database", ""), novelty=1.0)
    remember = salience(detect("Remember that the staging URL is staging.acme.dev", ""), novelty=0.0)
    assert chit < 0.25 <= decision and remember >= 0.9


def test_due_dates():
    assert parse_due("tomorrow", NOW).day == 29
    assert parse_due("by Friday", NOW).weekday() == 4
    assert parse_due("2026-12-01", NOW).month == 12
    assert parse_due("no date here", NOW) is None


def test_social_facts_are_salient_and_extracted():
    msg = "Alice owns the payments service and she wants Stripe webhooks to be retried."
    sig = detect(msg, "Noted.")
    assert "social" in sig.user and salience(sig, novelty=1.0) >= 0.25
    ops = _extract(msg, "Noted.")
    assert any(o.subj == "Alice" and "Alice" in o.entities for o in ops)
