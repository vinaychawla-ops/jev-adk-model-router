#!/usr/bin/env python3
"""Demo Jev model routing without (or with) live API calls.

Default: scripted stand-in for Jev routes a few sample prompts.
Live:   python demo.py --live   (needs OPENROUTER_API_KEY, or run inside Muse)
"""

import argparse
import json
import sys
from types import SimpleNamespace

from audit import AuditLog
from router import DEEP_MODEL, FAST_MODEL, last_user_text, make_jev_model_router

SAMPLES = [
    "What is the capital of France?",
    "Summarize this paragraph in one sentence: Jev is a decision model that returns calibrated probabilities.",
    "We're choosing between two job offers with different salaries, equity, and relocation costs. Walk through the tradeoffs and recommend one with reasoning.",
    "Plan a 3-day product launch: sequencing, owners, risks, and rollback criteria.",
]

# scripted tiers for the mock run
MOCK_TIER = {
    SAMPLES[0]: "fast",
    SAMPLES[1]: "fast",
    SAMPLES[2]: "deep",
    SAMPLES[3]: "deep",
}


def mock_decide(state, questions, **kwargs):
    tier = MOCK_TIER.get(state, "fast")
    probs = {"fast": 0.9, "deep": 0.1} if tier == "fast" else {"fast": 0.15, "deep": 0.85}
    return ({"answers": {"route": {"type": "choice", "choice": tier, "probabilities": probs}}}, 4.0)


def fake_request(prompt):
    part = SimpleNamespace(text=prompt)
    content = SimpleNamespace(role="user", parts=[part])
    return SimpleNamespace(model=FAST_MODEL, contents=[content])


def run(callback, label):
    print(f"\n--- {label} ---")
    for prompt in SAMPLES:
        req = fake_request(prompt)
        assert callback(None, req) is None  # router never short-circuits
        routed = "FLASH (cheap)" if req.model == FAST_MODEL else "PRO (deep)"
        deep = req.model == DEEP_MODEL
        print(f"  [{'DEEP' if deep else 'fast':>4}] {routed:14} <- {prompt[:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="call the real Jev API")
    ap.add_argument("--audit", metavar="PATH", default=None,
                    help="write a JSONL audit log of every routing decision to PATH")
    ns = ap.parse_args()

    audit = AuditLog(path=ns.audit) if ns.audit else None
    if ns.live:
        run(make_jev_model_router(audit=audit), "LIVE Jev routing (typesafe/jev-1.13 via OpenRouter Decisions API)")
    else:
        run(make_jev_model_router(decide_fn=mock_decide, audit=audit), "MOCK Jev routing (no API calls)")
        print("\nTip: re-run with --live to route the same prompts with real Jev.")
    if audit is not None:
        print(f"\naudit: {len(audit)} routing decisions recorded -> {ns.audit}")
        print(f"summary: {json.dumps(audit.summary())}")


if __name__ == "__main__":
    sys.exit(main())
