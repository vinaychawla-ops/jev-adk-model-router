"""Jev-powered model router for Google ADK agents.

The ADK-native equivalent of LangChain's ``ModelRouterMiddleware``: instead of
asking an LLM which model should handle a request, a ``before_model_callback``
sends the user's prompt to Jev with one typed ``choice`` question. Jev returns
the winning model tier plus per-option probabilities -- no text generated --
and the callback rewrites ``llm_request.model`` before the call goes out.

Why this works in ADK: the flow populates ``llm_request.model`` from the
agent during preprocessing, then runs ``before_model_callback``; a callback
that returns ``None`` lets the call proceed with the (possibly mutated)
request, and the Gemini backend sends ``llm_request.model`` as the model
name. So mutating ``llm_request.model`` genuinely reroutes the request.

Usage::

    from router import make_jev_model_router

    agent = Agent(
        ...
        model="gemini-2.5-flash",          # cheap default
        before_model_callback=make_jev_model_router(),  # upgrades to pro when needed
    )
"""

import logging
from typing import Callable, Dict, Optional, Tuple

from jev_client import JevError, choice_selection, decide

logger = logging.getLogger(__name__)

FAST_MODEL = "gemini-2.5-flash"
DEEP_MODEL = "gemini-2.5-pro"

ROUTE_QUESTION_ID = "route"

# Tier keys ("fast"/"deep") must match the keys of the ``choices`` mapping
# passed to ``make_jev_model_router``.
ROUTE_QUESTION = {
    ROUTE_QUESTION_ID: {
        "type": "choice",
        "instructions": "Choose the least expensive model that can reliably solve the task.",
        "criteria": {
            "fast": (
                "Straightforward extraction, rewriting, lookup, or short-answer "
                "tasks that do not require multi-step reasoning."
            ),
            "deep": (
                "Ambiguous or high-stakes tasks requiring planning, tradeoff "
                "analysis, or multi-step reasoning."
            ),
        },
    }
}

DEFAULT_CHOICES = {"fast": FAST_MODEL, "deep": DEEP_MODEL}


def last_user_text(contents) -> str:
    """Extract the latest user message text from an LlmRequest's contents.

    Duck-typed (getattr) so it works with real ``types.Content`` objects and
    with simple test doubles.
    """
    for content in reversed(list(contents or [])):
        if getattr(content, "role", None) != "user":
            continue
        parts = []
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return ""


def route_prompt(prompt: str, decide_fn: Callable = decide,
                 choices: Dict[str, str] = None, model: Optional[str] = None) -> Tuple[str, dict]:
    """Ask Jev which tier should handle ``prompt``.

    Returns ``(model_name, info)`` where ``info`` carries the winning tier,
    per-tier probabilities, and latency. Raises :class:`JevError` on failure
    so callers can decide their own fallback.
    """
    choices = choices or DEFAULT_CHOICES
    kwargs = {"model": model} if model else {}
    response, latency_ms = decide_fn(prompt, ROUTE_QUESTION, **kwargs)
    tier, probabilities = choice_selection(response, ROUTE_QUESTION_ID)
    if tier not in choices:
        raise JevError(f"Jev returned unknown tier {tier!r}; expected one of {sorted(choices)}")
    return choices[tier], {
        "tier": tier,
        "probabilities": probabilities,
        "latency_ms": latency_ms,
    }


def make_jev_model_router(
    decide_fn: Callable = None,
    choices: Dict[str, str] = None,
    model: Optional[str] = None,
):
    """Build a ``before_model_callback(callback_context, llm_request)``.

    On every model call the callback asks Jev to pick the cheapest tier that
    can reliably solve the user's prompt, then sets ``llm_request.model``
    accordingly and returns ``None`` so the call proceeds. If Jev is
    unreachable or returns something unusable, the agent's configured
    (cheap default) model is kept and a warning is logged.

    Args:
        decide_fn: Jev call ``(state, questions, model=...) ->
            (response_dict, latency_ms)``. Defaults to the real
            :func:`jev_client.decide`; inject a stub in tests.
        choices: maps Jev tier keys to model names. Keys must match the
            ``criteria`` keys of the route question (``"fast"``/``"deep"``).
        model: override the Jev model pin (default ``typesafe/jev-1.13``).
    """
    _decide = decide_fn or decide
    _choices = dict(choices or DEFAULT_CHOICES)

    def jev_before_model_callback(callback_context, llm_request):
        prompt = last_user_text(getattr(llm_request, "contents", None))
        if not prompt:
            logger.warning("Jev router: no user text found; keeping default model")
            return None
        try:
            model_name, info = route_prompt(prompt, decide_fn=_decide, choices=_choices, model=model)
        except JevError as exc:
            logger.warning("Jev routing failed (%s); keeping default model", exc)
            return None
        previous = getattr(llm_request, "model", None)
        llm_request.model = model_name
        probs = ", ".join(f"{k}={v:.2f}" for k, v in info["probabilities"].items())
        logger.info(
            "JEV ROUTE: tier=%s (%s) %s -> %s (%.0f ms)",
            info["tier"], probs, previous, model_name, info["latency_ms"],
        )
        return None  # proceed with the rerouted request

    return jev_before_model_callback


# A ready-made callback using the real Jev API.
jev_model_router = make_jev_model_router()
