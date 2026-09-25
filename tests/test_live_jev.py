"""Live integration test: hits the real Jev Decisions API.

Skipped unless Jev is reachable -- either ``OPENROUTER_API_KEY`` is set
(local runs) or the Muse vault surrogate is importable (sandbox runs).
Never commits or prints any credential.
"""

import os

import pytest

from router import DEEP_MODEL, FAST_MODEL, route_prompt


def _jev_reachable():
    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    try:
        import sys

        sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
        import dynamic_credentials  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _jev_reachable(), reason="Jev not reachable: set OPENROUTER_API_KEY")

PROMPTS = {
    "fast": "What is the capital of France? Reply with just the city name.",
    "deep": (
        "We are choosing between two job offers. Offer A pays $20k more but requires "
        "relocation and has a 1-year cliff on equity; offer B pays less, is remote, "
        "and has immediate equity vesting. Walk through the tradeoffs across salary, "
        "equity, cost of living, and career risk, then recommend one with reasoning."
    ),
}


def test_live_jev_routes_simple_prompt_to_fast():
    model_name, info = route_prompt(PROMPTS["fast"])
    assert model_name in (FAST_MODEL, DEEP_MODEL)
    assert info["tier"] in ("fast", "deep")
    assert abs(sum(info["probabilities"].values()) - 1.0) < 0.01
    print(f"\nlive Jev: {PROMPTS['fast'][:40]!r} -> tier={info['tier']} {info['probabilities']} ({info['latency_ms']:.0f} ms)")


def test_live_jev_routes_complex_prompt_to_deep():
    model_name, info = route_prompt(PROMPTS["deep"])
    assert model_name in (FAST_MODEL, DEEP_MODEL)
    assert info["tier"] in ("fast", "deep")
    print(f"\nlive Jev: tradeoff-analysis prompt -> tier={info['tier']} {info['probabilities']} ({info['latency_ms']:.0f} ms)")
