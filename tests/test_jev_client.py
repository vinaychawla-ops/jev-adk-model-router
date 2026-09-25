"""Tests for the stdlib Jev Decisions API client (all HTTP mocked)."""

import json
import os
from unittest import mock

import pytest

import jev_client
from jev_client import JevError, choice_selection, decide


def _fake_urlopen(captured, body):
    def _fake(req, timeout=60):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))

        class _Resp:
            def __init__(self):
                self._raw = json.dumps(body).encode("utf-8") if isinstance(body, dict) else body
                self._sent = False

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=-1):
                if self._sent:
                    return b""
                self._sent = True
                return self._raw if n is None or n < 0 else self._raw[:n]

        return _Resp()

    return _fake


CHOICE_BODY = {
    "answers": {
        "route": {
            "type": "choice",
            "choice": "deep",
            "confidence": 0.81,
            "probabilities": {"fast": 0.19, "deep": 0.81},
        }
    },
    "usage": {"input_tokens": 60, "output_tokens": 3, "cost": 0.000021},
    "provider": "typesafe",
    "model": "typesafe/jev-1.13",
}


def test_posts_to_decisions_endpoint_with_bearer_auth():
    captured = {}
    questions = {"route": {"type": "choice", "instructions": "pick", "criteria": {"fast": "f", "deep": "d"}}}
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-123"}), mock.patch(
        "urllib.request.urlopen", _fake_urlopen(captured, CHOICE_BODY)
    ):
        resp, ms = decide("some state", questions)

    assert captured["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert captured["headers"]["Authorization"] == "Bearer test-key-123"
    assert captured["payload"]["model"] == "typesafe/jev-1.13"
    assert captured["payload"]["state"] == "some state"
    assert captured["payload"]["questions"]["route"]["type"] == "choice"
    assert resp["answers"]["route"]["choice"] == "deep"
    assert ms >= 0


def test_model_override_is_sent():
    captured = {}
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}), mock.patch(
        "urllib.request.urlopen", _fake_urlopen(captured, CHOICE_BODY)
    ):
        decide("s", {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "b"}}}, model="typesafe/jev-latest")
    assert captured["payload"]["model"] == "typesafe/jev-latest"


def test_missing_key_raises_jev_error():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENROUTER_API_KEY", None)
        with mock.patch.object(jev_client, "_HAVE_VAULT", False):
            with pytest.raises(JevError, match="OPENROUTER_API_KEY"):
                decide("s", {"q": {"type": "noul", "instructions": "?"}})


def test_api_error_body_raises_jev_error():
    captured = {}
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}), mock.patch(
        "urllib.request.urlopen", _fake_urlopen(captured, {"error": {"message": "bad model"}})
    ):
        with pytest.raises(JevError, match="api error"):
            decide("s", {"q": {"type": "noul", "instructions": "?"}})


def test_choice_selection_extracts_winner_and_probabilities():
    tier, probs = choice_selection(CHOICE_BODY, "route")
    assert tier == "deep"
    assert probs == {"fast": 0.19, "deep": 0.81}


def test_choice_selection_missing_answer_raises():
    with pytest.raises(JevError):
        choice_selection({"answers": {}}, "route")
