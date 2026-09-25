# AGENTS.md -- jev-adk-model-router

## What this repo is
Jev (TypeSafe decision model) as a per-request model router for Google ADK
agents: a `before_model_callback` asks Jev's typed `choice` question which
tier should handle the prompt, then rewrites `llm_request.model`. Public demo
repo. The ADK equivalent of LangChain's `ModelRouterMiddleware`.

## Conventions
- `jev_client.py` stays **stdlib-only** (urllib, no requests/httpx). Keep it that way.
- Jev model is pinned to `typesafe/jev-1.13` in `jev_client.MODEL`. Only change
  the pin deliberately and note it in the README.
- Never commit API keys. Auth is `OPENROUTER_API_KEY` env var locally; inside
  Muse the `custom.openrouter` vault surrogate is used (see `jev_client.py`).
- Tests must run with zero network by default. Live API tests live in
  `tests/test_live_jev.py` and skip when Jev is unreachable.
- Router callbacks always return `None` (reroute, never short-circuit). The
  fallback on any Jev failure is the agent's cheap default model, with a
  warning logged.

## ADK mechanics (verified against installed google-adk; do not regress)
- `before_model_callback(callback_context, llm_request)`: return `None` to
  proceed (request may be mutated), return an `LlmResponse` to skip the call.
- The flow populates `llm_request.model` from the agent in preprocessing
  (`flows/llm_flows/basic.py`), then runs the callback; the Gemini backend
  (`models/google_llm.py`) sends `llm_request.model` as the model name. That
  is why mutating `llm_request.model` in the callback genuinely reroutes.
- `canonical_model_async` resolves the `BaseLlm` wrapper from the agent, not
  from the request -- routing works because the model *name* comes from the
  request at call time. If a future ADK version changes this, routing must be
  re-verified, not assumed.
- The routing state is the latest user message text, extracted duck-typed
  from `llm_request.contents` (`router.last_user_text`).

## Jev API facts (verified, don't regress)
- Endpoint: `POST https://openrouter.ai/api/alpha/decisions` (NOT chat/completions)
- Payload: `{"model", "state", "questions"}`; typed questions:
  `{"name": {"type": "choice", "instructions": "...", "criteria": {"k": "desc"}}}`
- Answers are dicts: `resp["answers"][name]["choice"]` -> winning key,
  `resp["answers"][name]["probabilities"]` -> per-option floats. Not attributes.
- `state` may be a plain string.

## Running
- `python demo.py` -- mock demo, no keys
- `python demo.py --live` -- real Jev (needs `OPENROUTER_API_KEY` or Muse vault)
- `pytest -q` -- full suite; live tests skip without a key
- `pytest -q tests/test_router.py tests/test_jev_client.py` -- offline only

## Tuning
- Tier criteria live in `router.ROUTE_QUESTION`; keep the keys ("fast"/"deep")
  in sync with the `choices` mapping passed to `make_jev_model_router`.
- To add tiers (e.g. "medium"), extend both the criteria and the choices map.
