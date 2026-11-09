"""One function to call any model on the Capital One gateway: `chat(...)`.

Inside Capital One every model is reached through the internal inference
client, which follows the OpenAI chat-completions schema. So there is one
code path here, not three, and `provider` is carried through only because
Weave records it as `gen_ai.provider.name`.

    from c1.aiml.genai.inference import Client as C1Client
    client.chat.completions.create(model=..., messages=..., api_version=2)

`api_version=2` is required by the gateway.

Unlike the public `openai` / `anthropic` clients, the gateway client is NOT
auto-instrumented by Weave, so `chat` and `chat_with_tools` are `@weave.op`s
themselves. That is what puts the model call, its arguments and its token
counts in the trace — the same thing the auto-patch does elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

try:
    # This is the import you'll use — Capital One's internal wrapper.
    from c1_aiml_aem import weave
except ImportError:
    import weave  # presenter laptop only; same API surface

import config


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    response_model: str = ""


@lru_cache(maxsize=None)
def _client():
    """The gateway client. Auth and routing are handled inside C1 — there is
    no API key to set, which is why .env.example has no provider keys."""
    from c1.aiml.genai.inference import Client as C1Client

    return C1Client()


def _create(**kwargs):
    """Every gateway call goes through here so `api_version=2` is never
    forgotten, and so TEMPERATURE is applied in exactly one place."""
    if config.TEMPERATURE is not None:
        kwargs["temperature"] = float(config.TEMPERATURE)
    return _client().chat.completions.create(api_version=2, **kwargs)


@weave.op()
def chat(provider: str, model: str, system: str, user: str,
         max_tokens: int = 700) -> LLMResult:
    """Single-turn chat call. `provider` is informational; the gateway routes
    on `model`. `max_tokens` is accepted for call-site compatibility and not
    sent — the gateway applies its own cap."""
    if config.uses_mock(provider):  # BACKEND=mock, or no gateway in this env
        from mock_llm import mock_chat

        return mock_chat(provider, model, system, user)

    resp = _create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    usage = resp.usage
    return LLMResult(
        text=resp.choices[0].message.content or "",
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        response_model=getattr(resp, "model", model) or model,
    )


# ---------------------------------------------------------------------------
# Tool calling, used by 01_trace_conversations.py's agent loop.
#
# Messages use one provider-neutral shape:
#   {"role": "user" | "assistant", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict}]}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}
# Tools: [{"name", "description", "parameters": <JSON schema>}]
# ---------------------------------------------------------------------------


@dataclass
class ToolTurn:
    text: str
    tool_calls: list[dict]          # [{"id", "name", "arguments": dict}]
    input_tokens: int = 0
    output_tokens: int = 0
    response_model: str = ""


@weave.op()
def chat_with_tools(provider: str, model: str, system: str, messages: list[dict],
                    tools: list[dict], max_tokens: int = 700) -> ToolTurn:
    """One model step: either a final answer (no tool_calls) or tool requests."""
    if config.uses_mock(provider):
        from mock_llm import mock_chat_with_tools

        return mock_chat_with_tools(provider, model, system, messages, tools)

    import json

    msgs: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "tool":
            msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                         "content": m["content"]})
        elif m.get("tool_calls"):
            msgs.append({"role": "assistant", "content": m.get("content") or None,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": json.dumps(c["arguments"])}}
                                        for c in m["tool_calls"]]})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})

    resp = _create(
        model=model,
        messages=msgs,
        tools=[{"type": "function", "function": t} for t in tools],
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    calls = [{"id": c.id, "name": c.function.name,
              "arguments": json.loads(c.function.arguments or "{}")}
             for c in (msg.tool_calls or [])]
    usage = resp.usage
    return ToolTurn(text=msg.content or "", tool_calls=calls,
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    response_model=getattr(resp, "model", model) or model)
