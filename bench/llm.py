"""LLM backends. Two wire formats cover almost everything:
  - openai   : OpenAI, and any OpenAI-compatible endpoint (Ollama, Gemini, OpenRouter, vLLM)
  - anthropic: Claude models

Each backend keeps its own provider-native message history and exposes:
  step() -> Turn(text, tool_calls[(id, name, args_or_None, raw_args)], in_tok, out_tok, latency_s)
  add_tool_results([(id, result_json_str)])
  add_user(text)
"""
import asyncio
import json
import time
from dataclasses import dataclass, field

from .interfaces import openai_tools, anthropic_tools


@dataclass
class Turn:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    in_tok: int = 0
    cached_tok: int = 0
    out_tok: int = 0
    latency_s: float = 0.0
    retries: int = 0


class FatalAPIError(RuntimeError):
    """Quota/auth problems: retrying or continuing the batch is pointless."""


FATAL_MARKERS = ("insufficient_quota", "invalid_api_key", "billing", "exceeded your current quota",
                 "model_not_found", "does not exist")


class OpenAIBackend:
    def __init__(self, model, system, first_user, use_tools, base_url=None,
                 api_key=None, temperature=0.7, max_tokens=1024, reasoning_effort=None):
        from openai import AsyncOpenAI
        kw = {}
        if base_url:
            kw["base_url"] = base_url
        if api_key:
            kw["api_key"] = api_key
        self.client = AsyncOpenAI(**kw)
        self.model = model
        self.use_tools = use_tools
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.dropped = []            # params the endpoint rejected (report in Method section)
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": first_user}]

    async def step(self) -> Turn:
        kw = dict(model=self.model, messages=self.messages)
        if self.temperature is not None and "temperature" not in self.dropped:
            kw["temperature"] = self.temperature
        if self.max_tokens:
            kw["max_completion_tokens"] = self.max_tokens
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        if self.use_tools:
            kw["tools"] = openai_tools()
            kw["tool_choice"] = "auto"
        retries = 0
        while True:
            t0 = time.monotonic()
            try:
                r = await self.client.chat.completions.create(**kw)
                lat = time.monotonic() - t0
                break
            except Exception as ex:
                msg = str(ex)
                low = msg.lower()
                if any(k in low for k in FATAL_MARKERS):
                    raise FatalAPIError(msg) from ex
                # adapt to endpoint quirks, then retry immediately
                if "max_completion_tokens" in msg and "max_completion_tokens" in kw:
                    kw["max_tokens"] = kw.pop("max_completion_tokens")
                    continue
                if "temperature" in low and "temperature" in kw:
                    kw.pop("temperature")
                    self.dropped.append("temperature")
                    continue
                if "reasoning_effort" in low and kw.get("reasoning_effort") != "none":
                    kw["reasoning_effort"] = self.reasoning_effort = "none"
                    continue
                retries += 1
                if retries > 4:
                    raise
                await asyncio.sleep(2 ** retries)
        m = r.choices[0].message
        u = r.usage
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        turn = Turn(text=m.content or "", latency_s=lat, retries=retries,
                    in_tok=getattr(u, "prompt_tokens", 0) or 0, cached_tok=cached,
                    out_tok=getattr(u, "completion_tokens", 0) or 0)
        asst = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            asst["tool_calls"] = []
            for tc in m.tool_calls:
                raw = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = None
                turn.tool_calls.append((tc.id, tc.function.name, args, raw))
                asst["tool_calls"].append({"id": tc.id, "type": "function",
                                           "function": {"name": tc.function.name, "arguments": raw}})
        self.messages.append(asst)
        return turn

    def add_tool_results(self, results):
        for tc_id, content in results:
            self.messages.append({"role": "tool", "tool_call_id": tc_id, "content": content})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})


class AnthropicBackend:
    def __init__(self, model, system, first_user, use_tools, api_key=None,
                 temperature=0.7, max_tokens=1024, **_):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        self.model = model
        self.system = system
        self.use_tools = use_tools
        self.temperature = temperature
        self.max_tokens = max_tokens or 1024
        self.messages = [{"role": "user", "content": first_user}]

    async def step(self) -> Turn:
        kw = dict(model=self.model, system=self.system, messages=self.messages,
                  max_tokens=self.max_tokens)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.use_tools:
            kw["tools"] = anthropic_tools()
        retries = 0
        while True:
            t0 = time.monotonic()
            try:
                r = await self.client.messages.create(**kw)
                lat = time.monotonic() - t0
                break
            except Exception:
                retries += 1
                if retries > 4:
                    raise
                await asyncio.sleep(2 ** retries)
        cached = getattr(r.usage, "cache_read_input_tokens", 0) or 0
        turn = Turn(latency_s=lat, retries=retries, cached_tok=cached,
                    in_tok=r.usage.input_tokens + cached, out_tok=r.usage.output_tokens)
        content = []
        texts = []
        for b in r.content:
            if b.type == "text":
                texts.append(b.text)
                content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                turn.tool_calls.append((b.id, b.name, dict(b.input), json.dumps(b.input)))
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        turn.text = "\n".join(texts)
        if not content:
            content = [{"type": "text", "text": "(no output)"}]
        self.messages.append({"role": "assistant", "content": content})
        return turn

    def add_tool_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": i, "content": c} for i, c in results]})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})


PRESETS = {
    # name: (backend, base_url, api_key_env)
    "openai": ("openai", None, None),
    "ollama": ("openai", "http://localhost:11434/v1", "ollama"),
    "gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai/", None),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", None),
    "anthropic": ("anthropic", None, None),
}


def make_backend(provider, model, system, first_user, use_tools, temperature=0.7,
                 base_url=None, api_key=None, reasoning_effort=None):
    import os
    kind, default_url, key = PRESETS[provider]
    if provider == "ollama":
        api_key = api_key or "ollama"
    elif provider == "gemini":
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
    elif provider == "openrouter":
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    url = base_url or default_url
    if kind == "openai":
        if reasoning_effort is None and provider == "openai" and model.startswith(("gpt-5", "gpt-6")):
            # GPT-5.x/6 reject function tools on chat.completions unless effort is 'none';
            # 'none' is also what we want for real-time control latency.
            reasoning_effort = "none"
        return OpenAIBackend(model, system, first_user, use_tools, base_url=url,
                             api_key=api_key, temperature=temperature,
                             reasoning_effort=reasoning_effort)
    return AnthropicBackend(model, system, first_user, use_tools, api_key=api_key,
                            temperature=temperature)
