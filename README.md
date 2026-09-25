# Jev Model Router for Google ADK

Let [Jev](https://typesafe.ai) (TypeSafe's specialized decision model) decide,
per request, which model should handle it -- the ADK-native version of
LangChain's `ModelRouterMiddleware`.

Instead of burning a flagship model on every prompt (or asking an LLM to
route, which is slow and uncalibrated), a `before_model_callback` sends the
user's prompt to Jev with one typed `choice` question. Jev returns the winning
tier plus per-tier probabilities -- no text generated -- and the callback
rewrites `llm_request.model` before the call goes out.

## How it works

```
user: "What is the capital of France?"
  -> before_model_callback
  -> Jev choice: "Choose the least expensive model that can reliably solve the task."
       fast: "Straightforward extraction, rewriting, lookup, or short-answer tasks..."
       deep: "Ambiguous or high-stakes tasks requiring planning, tradeoff analysis..."
  -> tier=fast (P=0.91) -> llm_request.model = "gemini-2.5-flash" -> call proceeds

user: "Walk through the tradeoffs between these two job offers and recommend one."
  -> tier=deep (P=0.85) -> llm_request.model = "gemini-2.5-pro" -> call proceeds
```

This genuinely reroutes the request: ADK populates `llm_request.model` from
the agent during preprocessing, runs `before_model_callback` (return `None` =
proceed with the possibly-mutated request), and the Gemini backend sends
`llm_request.model` as the model name. The callback never short-circuits --
it only reroutes.

## Quickstart

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=<your key>   # get one at openrouter.ai
python demo.py            # mock run, no API calls
python demo.py --live     # real Jev routing decisions
pytest                    # unit tests (mocked) + live tests if key is set
```

Minimal wiring in your own agent:

```python
from google.adk.agents import Agent
from router import make_jev_model_router

agent = Agent(
    name="jev_routed_agent",
    model="gemini-2.5-flash",  # cheap default; router upgrades per request
    instruction="You are a helpful assistant.",
    before_model_callback=make_jev_model_router(),
)
```

Custom tiers:

```python
make_jev_model_router(choices={
    "fast": "gemini-2.5-flash-lite",
    "deep": "gemini-2.5-pro",
})
```

If Jev is unreachable or returns an unknown tier, the agent's configured
(cheap default) model is kept and a warning is logged -- routing degrades to
"always fast", never to an error.

## Jev API notes

- Endpoint is `POST https://openrouter.ai/api/alpha/decisions` -- the
  **Decisions** API, not `/api/v1/chat/completions`. Chat SDKs will not work.
- Request: `{"model": ..., "state": <string|object>, "questions": {...}}`.
- A `choice` answer arrives as `response["answers"][name]["choice"]` (the
  winning key) plus `response["answers"][name]["probabilities"]` (dict
  access, not attributes).
- The client (`jev_client.py`) is stdlib-only: no dependency beyond Python 3.

## Files

- `jev_client.py` -- stdlib-only Jev Decisions API client
- `router.py` -- `make_jev_model_router()`: the ADK `before_model_callback`
- `agent.py` -- working ADK agent (flash default, Jev-routed per request)
- `demo.py` -- mock or `--live` end-to-end demo
- `tests/` -- mocked unit tests + live integration tests (skipped without a key)
