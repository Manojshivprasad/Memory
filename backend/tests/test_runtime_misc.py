from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.auth import jwt as tokens
from app.auth.deps import principal_from_token
from app.runtime.gate import should_retrieve
from app.runtime.graph import ProjectGraph
from app.runtime.temporal import parse

NOW = datetime(2026, 9, 28, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize("q", ["hi", "Thanks!", "ok", "  ", "good morning there"])
def test_gate_skips_phatic(q):
    assert should_retrieve(q)[0] is False


@pytest.mark.parametrize("q", ["what db did we pick?", "continue where we left off", "fix the login bug"])
def test_gate_passes_information_needs(q):
    assert should_retrieve(q)[0] is True


def test_personal_vs_general_knowledge_prompts():
    from app.runtime.gate import is_personal

    assert is_personal("what DB did we pick?") and is_personal("any open todos?")
    assert is_personal("remind me what my deadline is")
    assert not is_personal("explain how quicksort works")
    assert not is_personal("write a regex to validate emails")


def test_temporal_cues():
    y = parse("what did we do yesterday?", NOW)
    assert y.has_window and y.start.day == 27 and y.historical
    assert parse("what did we use before we switched?", NOW).historical
    assert parse("let's continue where we left off", NOW).last_chat
    assert not parse("add a login page", NOW).has_window


def test_graph_match_and_spreading_activation():
    g = ProjectGraph(names={"postgres": "e1", "billing service": "e2", "alice": "e3"},
                     display={"e1": "Postgres", "e2": "Billing service", "e3": "Alice"},
                     adj={"e2": [("e3", 1.0), ("e1", 0.5)], "e3": [("e2", 1.0)], "e1": [("e2", 0.5)]})
    assert g.match("Who owns the Billing Service?") == ["e2"]
    act = g.spread({"e2": 1.0})
    assert act["e2"] == 1.0 and act["e3"] == 0.5 and act["e1"] == 0.25


def test_jwt_roundtrip_and_type_checks():
    tid, uid = uuid4(), uuid4()
    tok = tokens.issue("subject", {"tid": str(tid), "sub": "u1", "uid": str(uid), "scopes": ["prefetch"]}, 60)
    p = principal_from_token(tok)
    assert p.kind == "subject" and p.user_id == uid and p.tenant_id == tid
    with pytest.raises(HTTPException):
        p.require_scope("memory:write")
    refresh = tokens.issue("refresh", {"tid": str(tid), "sub": "x"}, 60)
    with pytest.raises(HTTPException):
        principal_from_token(refresh)  # refresh tokens can't call the API
    with pytest.raises(HTTPException):
        principal_from_token(tok[:-2] + "xx")  # tampered


def test_subject_token_project_restriction():
    pid = uuid4()
    tok = tokens.issue("subject", {"tid": str(uuid4()), "sub": "u", "uid": str(uuid4()), "scopes": ["chat"],
                                   "pids": [str(pid)]}, 60)
    p = principal_from_token(tok)
    p.check_project(pid)
    with pytest.raises(HTTPException):
        p.check_project(uuid4())
