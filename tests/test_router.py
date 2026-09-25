"""Tests for the Jev ADK model router (Jev itself is stubbed; no network)."""

from types import SimpleNamespace

import pytest

from jev_client import JevError
from router import (
    DEEP_MODEL,
    FAST_MODEL,
    ROUTE_QUESTION,
    ROUTE_QUESTION_ID,
    last_user_text,
    make_jev_model_router,
    route_prompt,
)


def stub_decide(tier=None, probabilities=None, exc=None, raw_answer=None):
    """Build a fake decide_fn. Captures (state, questions, model) per call."""
    calls = []

    def _fake(state, questions, model=None):
        calls.append({"state": state, "questions": questions, "model": model})
        if exc is not None:
            raise exc
        answer = raw_answer if raw_answer is not None else {
            "type": "choice",
            "choice": tier,
            "probabilities": probabilities or {},
        }
        return {"answers": {ROUTE_QUESTION_ID: answer}}, 6.0

    _fake.calls = calls
    return _fake


def fake_request(prompt, model=FAST_MODEL):
    part = SimpleNamespace(text=prompt)
    content = SimpleNamespace(role="user", parts=[part])
    return SimpleNamespace(model=model, contents=[content])


# --- routing decisions -------------------------------------------------------


def test_fast_prompt_routes_to_fast_model():
    cb = make_jev_model_router(decide_fn=stub_decide("fast", {"fast": 0.91, "deep": 0.09}))
    req = fake_request("What is the capital of France?")
    assert cb(None, req) is None  # never short-circuits the model call
    assert req.model == FAST_MODEL


def test_deep_prompt_routes_to_deep_model():
    cb = make_jev_model_router(decide_fn=stub_decide("deep", {"fast": 0.2, "deep": 0.8}))
    req = fake_request("Plan a product launch with tradeoffs.")
    assert cb(None, req) is None
    assert req.model == DEEP_MODEL


def test_custom_choices_mapping_is_respected():
    choices = {"fast": "my-cheap-model", "deep": "my-smart-model"}
    cb = make_jev_model_router(decide_fn=stub_decide("deep", {"fast": 0.1, "deep": 0.9}), choices=choices)
    req = fake_request("something hard")
    cb(None, req)
    assert req.model == "my-smart-model"


def test_unknown_tier_keeps_default_model(caplog):
    cb = make_jev_model_router(decide_fn=stub_decide("ultra", {"ultra": 1.0}))
    req = fake_request("something", model=FAST_MODEL)
    with caplog.at_level("WARNING"):
        assert cb(None, req) is None
    assert req.model == FAST_MODEL  # unchanged
    assert "unknown tier" in caplog.text


def test_jev_outage_keeps_default_model(caplog):
    cb = make_jev_model_router(decide_fn=stub_decide(exc=JevError("timeout")))
    req = fake_request("something", model=FAST_MODEL)
    with caplog.at_level("WARNING"):
        assert cb(None, req) is None
    assert req.model == FAST_MODEL
    assert "Jev routing failed" in caplog.text


def test_empty_prompt_keeps_default_model_no_jev_call(caplog):
    fake = stub_decide("deep", {"fast": 0.0, "deep": 1.0})
    cb = make_jev_model_router(decide_fn=fake)
    req = SimpleNamespace(model=FAST_MODEL, contents=[])
    with caplog.at_level("WARNING"):
        assert cb(None, req) is None
    assert req.model == FAST_MODEL
    assert fake.calls == []


# --- route_prompt helper -----------------------------------------------------


def test_route_prompt_returns_model_and_info():
    model_name, info = route_prompt(
        "Summarize this.", decide_fn=stub_decide("fast", {"fast": 0.88, "deep": 0.12})
    )
    assert model_name == FAST_MODEL
    assert info["tier"] == "fast"
    assert info["probabilities"] == {"fast": 0.88, "deep": 0.12}
    assert info["latency_ms"] == 6.0


def test_route_prompt_raises_on_unknown_tier():
    with pytest.raises(JevError, match="unknown tier"):
        route_prompt("x", decide_fn=stub_decide("mystery", {}))


# --- request shape sent to Jev -----------------------------------------------


def test_state_is_the_user_prompt_text():
    fake = stub_decide("fast", {"fast": 1.0, "deep": 0.0})
    cb = make_jev_model_router(decide_fn=fake)
    cb(None, fake_request("Extract the invoice total."))
    assert fake.calls[0]["state"] == "Extract the invoice total."


def test_question_is_typed_choice_with_least_expensive_instruction():
    fake = stub_decide("fast", {"fast": 1.0, "deep": 0.0})
    cb = make_jev_model_router(decide_fn=fake)
    cb(None, fake_request("hi"))
    q = fake.calls[0]["questions"][ROUTE_QUESTION_ID]
    assert q["type"] == "choice"
    assert "least expensive" in q["instructions"]
    assert set(q["criteria"]) == {"fast", "deep"}


# --- last_user_text ----------------------------------------------------------


def test_last_user_text_picks_latest_user_message():
    def content(role, text):
        return SimpleNamespace(role=role, parts=[SimpleNamespace(text=text)])

    contents = [
        content("user", "first question"),
        content("model", "first answer"),
        content("user", "follow-up question"),
    ]
    assert last_user_text(contents) == "follow-up question"


def test_last_user_text_empty_when_no_user_content():
    assert last_user_text([]) == ""
    assert last_user_text(None) == ""
    assert last_user_text([SimpleNamespace(role="model", parts=[SimpleNamespace(text="hi")])]) == ""
