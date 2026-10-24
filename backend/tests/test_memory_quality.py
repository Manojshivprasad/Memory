"""Regression guards for failures the evaluation harness found (see eval/)."""

from app.runtime.intents import constraint_domains, query_attributes, query_domains
from app.textutil import embedding_text
from app.write.ops import Op, canonical_pred, grounded, validate
from app.write.signals import informative


def test_small_models_omitting_content_still_store_the_fact():
    o = Op.from_dict({"op": "ADD", "subj": "user", "pred": "monthly_budget", "obj": "60000 INR", "due_at": "Dec 2027"})
    assert o and o.content and "60000 INR" in o.content and o.due_at is None  # bad date doesn't drop the fact


def test_free_form_predicates_are_canonical():
    assert canonical_pred("moved_to") == canonical_pred("location") == "lives_in"
    assert canonical_pred("joined") == "works_at"
    assert canonical_pred("weekly_mileage") == "weekly_distance"


def test_assistant_cannot_assert_facts_only_propose():
    ops = [Op(op="ADD", kind="fact", content="Lima is the capital of Peru.", speaker="assistant"),
           Op(op="ADD", kind="decision", content="Assistant suggestion: use Celery.", speaker="assistant")]
    assert [o.kind for o in validate(ops, set())] == ["decision"]


def test_scope_rules():
    novel = Op(op="ADD", content="The protagonist is Elena.", subj="user", pred="protagonist", obj="Elena", scope="user")
    pref = Op(op="ADD", kind="preference", content="The user wants short answers.", scope="project")
    home = Op(op="ADD", content="The user lives in Bangalore.", subj="user", pred="moved_to", obj="Bangalore",
              scope="project")
    task = Op(op="ADD", kind="commitment", content="Send the portfolio to Tom.", scope="user")
    out = validate([novel, pref, home, task], set(), source="In general, keep answers short for every project.")
    assert [o.scope for o in out] == ["project", "user", "user", "project"]


def test_third_party_names_never_become_the_users_name():
    o = Op(op="ADD", content="The user is Alice.", subj="user", pred="name", obj="Alice")
    assert validate([o], set(), source="Alice owns the payments service and Bob handles DevOps.") == []
    ok = Op(op="ADD", content="The user's name is Manoj.", subj="user", pred="name", obj="Manoj")
    assert validate([ok], set(), source="My name is Manoj by the way.")


def test_change_events_dropped_when_new_state_present():
    ops = [Op(op="ADD", content="Switched database from Postgres to MySQL."),
           Op(op="ADD", content="The project uses MySQL.", pred="uses_database", obj="MySQL")]
    assert [o.content for o in validate(ops, set())] == ["The project uses MySQL."]


def test_grounding_blocks_resurrected_values():
    assert not grounded("15,000 INR", "I don't want to invest in crypto.")
    assert grounded("20,000 INR", "I increased my SIP to 20,000 a month.")
    assert grounded("2027-03-03", "her birthday is March 3rd")


def test_informative_gate():
    assert informative("The protagonist is Elena, a lighthouse keeper.")
    assert informative("She loves pottery. Maybe a pottery class?")
    assert not informative("what's the capital of Peru?")
    assert not informative("haha nice, thanks")


def test_first_person_embedding_text():
    assert embedding_text("The user lives in Bangalore.") == "I live in Bangalore."
    assert embedding_text("The user's sister is Priya.") == "my sister is Priya."


def test_standing_constraints_and_attribute_intents():
    assert constraint_domains("The user is severely allergic to peanuts.") == ["food"]
    assert constraint_domains("The user has a left knee injury.") == ["exercise"]
    assert constraint_domains("The user wants answers short with bullets.", "preference") == ["always"]
    assert constraint_domains("The user lives in Bangalore.") is None
    assert "food" in query_domains("What can I make for dinner tonight?")
    assert "works_at" in query_attributes("Which company do I work for right now?")
    assert "teacher" in query_attributes("Who is Aarav's teacher?")


def test_preference_reversals():
    from app.memory.polarity import contradicts

    assert contradicts("The user hates playing chess.", "The user enjoys playing chess.")
    assert contradicts("The user likes to play cricket instead of chess.",
                       "The user is interested in improving their chess skills.")
    assert contradicts("The user loves chess again.", "The user hates playing chess.")
    assert not contradicts("The user is not going to play chess today.", "The user enjoys playing chess.")
    assert not contradicts("The user is severely allergic to peanuts.", "The user is lactose intolerant.")
    assert not contradicts("The user likes Python.", "The user does not like Java.")
