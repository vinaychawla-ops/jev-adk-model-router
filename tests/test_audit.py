"""Tests for audit logging: the log itself plus router wiring."""

import json
from types import SimpleNamespace

from audit import AuditLog, prompt_hash, prompt_preview
from jev_client import JevError
from router import DEEP_MODEL, FAST_MODEL, ROUTE_QUESTION_ID, make_jev_model_router


def stub_decide(tier=None, probabilities=None, exc=None):
    def _fake(state, questions, model=None):
        if exc is not None:
            raise exc
        return ({"answers": {ROUTE_QUESTION_ID: {
            "type": "choice", "choice": tier, "probabilities": probabilities or {}}}}, 7.0)

    return _fake


def fake_request(prompt, model=FAST_MODEL):
    part = SimpleNamespace(text=prompt)
    return SimpleNamespace(model=model, contents=[SimpleNamespace(role="user", parts=[part])])


# --- AuditLog core -----------------------------------------------------------


def test_record_stamps_ts_and_appends():
    log = AuditLog()
    evt = log.record({"component": "router", "verdict": "routed-fast"})
    assert evt["ts"]
    assert len(log) == 1


def test_jsonl_sink_writes_parseable_lines(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path)
    log.record({"component": "router", "verdict": "routed-deep", "tier": "deep"})
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    assert lines[0]["tier"] == "deep"
    assert "ts" in lines[0]


def test_summary_counts_by_verdict():
    log = AuditLog()
    log.record({"verdict": "routed-fast"})
    log.record({"verdict": "routed-deep"})
    log.record({"verdict": "routed-fast"})
    assert log.summary() == {"total": 3, "by_verdict": {"routed-fast": 2, "routed-deep": 1}}


def test_prompt_preview_truncates_and_hash_is_stable():
    assert len(prompt_preview("x" * 500)) == 200
    assert prompt_hash("abc") == prompt_hash("abc")
    assert prompt_hash("abc") != prompt_hash("abd")


# --- router wiring -----------------------------------------------------------


def test_routing_decision_is_audited():
    log = AuditLog()
    cb = make_jev_model_router(decide_fn=stub_decide("deep", {"fast": 0.15, "deep": 0.85}), audit=log)
    req = fake_request("Plan a product launch with tradeoffs.")
    assert cb(None, req) is None
    assert req.model == DEEP_MODEL
    assert len(log) == 1
    evt = log.events[0]
    assert evt["component"] == "router"
    assert evt["verdict"] == "routed-deep"
    assert evt["tier"] == "deep"
    assert evt["probabilities"] == {"fast": 0.15, "deep": 0.85}
    assert evt["previous_model"] == FAST_MODEL
    assert evt["chosen_model"] == DEEP_MODEL
    assert evt["latency_ms"] == 7.0
    assert evt["jev_model"] == "typesafe/jev-1.13"
    assert "product launch" in evt["prompt_preview"]
    assert evt["prompt_hash"] == prompt_hash("Plan a product launch with tradeoffs.")


def test_fallback_is_audited_with_reason():
    log = AuditLog()
    cb = make_jev_model_router(decide_fn=stub_decide(exc=JevError("timeout")), audit=log)
    req = fake_request("something")
    assert cb(None, req) is None
    evt = log.events[0]
    assert evt["verdict"] == "fallback-default"
    assert evt["chosen_model"] == FAST_MODEL  # default kept
    assert "timeout" in evt["reason"]


def test_no_audit_log_means_no_overhead():
    cb = make_jev_model_router(decide_fn=stub_decide("deep", {"fast": 0.0, "deep": 1.0}))
    req = fake_request("hard task")
    assert cb(None, req) is None
    assert req.model == DEEP_MODEL
