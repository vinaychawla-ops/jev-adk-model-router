"""ADK agent with Jev-powered per-request model routing.

The agent's configured model is the cheap default (``gemini-2.5-flash``).
Before every model call, :func:`router.make_jev_model_router` asks Jev which
tier can reliably solve the user's prompt and rewrites ``llm_request.model``
accordingly -- upgrading to ``gemini-2.5-pro`` for ambiguous, high-stakes, or
multi-step tasks, staying on flash for straightforward ones.

A live run needs ``OPENROUTER_API_KEY`` (Jev) and ``GOOGLE_API_KEY`` (Gemini).
Run ``python demo.py`` for a keyless mock demo.
"""

from router import FAST_MODEL, make_jev_model_router

try:
    from google.adk.agents import Agent

    _HAVE_ADK = True
except ImportError:  # pragma: no cover - lets tests import without google-adk
    _HAVE_ADK = False

jev_before_model_callback = make_jev_model_router()

if _HAVE_ADK:
    root_agent = Agent(
        name="jev_routed_agent",
        model=FAST_MODEL,  # cheap default; the router upgrades per request
        instruction=(
            "You are a helpful assistant. Answer the user's request directly "
            "and concisely."
        ),
        before_model_callback=jev_before_model_callback,
    )
else:  # pragma: no cover
    root_agent = None
