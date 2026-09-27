"""Provider-agnostic LLM wrapper.

Groq and Gemini both expose OpenAI-compatible endpoints, so one client covers
both. Switch providers with LLM_PROVIDER in .env; no code changes.
If no API key is set, chat_json() returns None and callers fall back to
rules-only mode (keeps the demo alive if the free tier rate-limits you).

Models get renamed/retired. If the configured model returns "model not found",
we try the fallbacks below once, remember the first that works, and if none
do, switch to rules-only for the rest of the run (one clear message, no spam).
"""
import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    # name: (base_url, env var for key, models to try in order; LLM_MODEL in .env goes first)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY",
             ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
               ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]),
}

_state = {"model": None, "dead": set(), "disabled": False, "no_json_mode": set(), "cooldown_until": 0.0,
          "last_failed": False, "model_cooldown": {}}
MAX_WAIT = int(os.getenv("LLM_MAX_WAIT", "65"))  # wait out per-minute limits up to this many seconds


def _retry_after(exc) -> float | None:
    """Seconds the provider asks us to wait ('Please try again in 1m2.5s' / '7.2s' / '450ms')."""
    text = str(exc)
    m = re.search(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?\s*(?:([\d.]+)ms)?", text, re.I)
    if not m or not any(m.groups()):
        return None
    h, mi, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec or 0) + float(ms or 0) / 1000


def _is_rate_limit(exc) -> bool:
    t = str(exc).lower()
    return "429" in t or "rate limit" in t or "rate_limit" in t or "quota" in t


def last_call_failed() -> bool:
    """True if the most recent chat_json() wanted the LLM but couldn't get an answer (rate limit, outage)."""
    return _state["last_failed"]


def _provider():
    return os.getenv("LLM_PROVIDER", "groq").lower()


def _providers() -> list[str]:
    """Your LLM_PROVIDER first, then every other provider you've put a key in .env for (automatic backup)."""
    first = _provider()
    order = [first] + [p for p in PROVIDERS if p != first]
    return [p for p in order if p in PROVIDERS and os.getenv(PROVIDERS[p][1])]


def _provider_of(model: str) -> str:
    if model.startswith("gemini") or model == os.getenv("GEMINI_MODEL"):
        return "gemini"
    for name, (_, _, models) in PROVIDERS.items():
        if model in models:
            return name
    return _provider()  # a custom LLM_MODEL belongs to your main provider


_discovered = {}


def _models_for(provider: str) -> list[str]:
    """The models to try for a provider. For Gemini, ask Google which models THIS key may use (new keys can't use
    retired ones like gemini-2.5-flash) and take its "flash" models, newest first; fall back to the fixed list."""
    fixed = PROVIDERS[provider][2]
    own = [os.getenv(f"{provider.upper()}_MODEL")] if os.getenv(f"{provider.upper()}_MODEL") else []
    if provider != "gemini":
        return own + fixed
    if provider not in _discovered:
        found = []
        try:
            for m in _client(provider).models.list():
                mid = str(getattr(m, "id", "")).removeprefix("models/")
                if "flash" in mid and not re.search(r"tts|image|audio|live|embed|exp|preview|thinking", mid):
                    found.append(mid)
        except Exception:
            found = []
        ver = lambda mid: [float(x) for x in re.findall(r"\d+(?:\.\d+)?", mid)[:1]] or [0.0]
        found.sort(key=lambda mid: (ver(mid), "lite" not in mid), reverse=True)
        _discovered[provider] = found
    return own + (_discovered[provider] or fixed)


def _candidates():
    first = [os.getenv("LLM_MODEL")] if os.getenv("LLM_MODEL") else []
    models = []
    for p in _providers() or [_provider()]:
        models += _models_for(p)
    seen, out = set(), []
    for m in first + models:
        if m and m not in seen and m not in _state["dead"]:
            seen.add(m)
            out.append(m)
    return out


_clients = {}


def _client(provider: str | None = None):
    """An API client for `provider` (default: the first one with a key), or None when there's no key."""
    if _state["disabled"] or os.getenv("JOBPILOT_RULES_ONLY") == "1":
        return None
    provider = provider or (_providers() or [None])[0]
    if provider is None:
        return None
    base_url, key_var, _ = PROVIDERS[provider]
    key = os.getenv(key_var)
    if not key:
        return None
    if (provider, key) not in _clients:
        from openai import OpenAI  # imported lazily so rules-only mode needs no SDK

        _clients[(provider, key)] = OpenAI(base_url=base_url, api_key=key, max_retries=0)
    return _clients[(provider, key)]


def _parse_json(text: str):
    text = re.sub(r"(?is)<think>.*?</think>", "", text or "").strip()  # Qwen-style models may "think out loud" first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate ```json fences / reasoning text
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def llm_available() -> bool:
    return _client() is not None


def llm_label() -> str:
    if _client() is None:
        return "rules-only (LLM disabled)" if _state["disabled"] else "rules-only (no API key)"
    model = _state["model"] or (_candidates() or ["?"])[0]
    backups = [p for p in _providers() if p != _provider_of(model)]
    return f"{_provider_of(model)}:{model}" + (f" (backup: {', '.join(backups)})" if backups else "")


def _is_model_missing(exc) -> bool:
    msg = str(exc).lower()
    return ("model_not_found" in msg or "does not exist" in msg or "decommissioned" in msg or "not found" in msg
            or "no longer available" in msg or "not supported" in msg or "error code: 404" in msg)


def _call(client, model, system, user, temperature, max_tokens=None):
    kwargs = dict(model=model, temperature=temperature,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if model not in _state["no_json_mode"]:
        kwargs["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model and model not in _state.setdefault("no_effort", set()):
        # reasoning models spend most tokens "thinking"; low effort is plenty for triage and short drafts,
        # and keeps us under free-tier tokens-per-minute limits
        kwargs["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT", "low")
    kwargs["max_tokens"] = max_tokens or int(os.getenv("LLM_MAX_TOKENS", "700"))
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        # some models reject JSON mode; retry once without it (the prompt still asks for JSON)
        msg = str(exc).lower()
        if "reasoning_effort" in kwargs and "reasoning_effort" in msg:
            _state["no_effort"].add(model)
            kwargs.pop("reasoning_effort")
            return client.chat.completions.create(**kwargs)
        if "response_format" in kwargs and "response_format" in msg:
            _state["no_json_mode"].add(model)
            kwargs.pop("response_format")
            return client.chat.completions.create(**kwargs)
        raise


def _available(now: float) -> list[str]:
    models = [m for m in ([_state["model"]] if _state["model"] else []) + _candidates()
              if _state["model_cooldown"].get(m, 0) <= now]
    return list(dict.fromkeys(models))


def chat_json(system: str, user: str, temperature: float = 0.1, retries: int = 2, max_tokens: int | None = None):
    """Return a parsed JSON dict from the model, or None if unavailable/failed.

    Free limits are per model, so when one model says "slow down" we switch to another that's free instead of
    sleeping (7 models' limits used side by side). We only wait when EVERY model is resting and the soonest one
    is back within LLM_MAX_WAIT seconds; longer than that, the item is left for the next run."""
    _state["last_failed"] = False
    client = _client()
    if client is None:
        return None
    if time.time() < _state["cooldown_until"]:
        _state["last_failed"] = True  # still inside a long provider limit: don't hammer, don't spam
        return None
    while True:
        models = _available(time.time())
        if not models:
            resting = [_state["model_cooldown"].get(m, 0) for m in _candidates()]
            soonest = min(resting) if resting else time.time() + 600
            if soonest - time.time() <= MAX_WAIT:
                time.sleep(max(0.0, soonest - time.time()) + 1)
                continue
            _state["cooldown_until"] = soonest
            print(f"[llm] every model is at its limit (next one back around "
                  f"{time.strftime('%H:%M', time.localtime(soonest))}). Unsure items are left for the next run.")
            _state["last_failed"] = True
            return None
        for model in models:
            attempt = 0
            while attempt <= retries:
                try:
                    resp = _call(_client(_provider_of(model)) or client, model, system, user, temperature,
                                 *([max_tokens] if max_tokens else []))
                    _state["model"] = model
                    parsed = _parse_json(resp.choices[0].message.content or "")
                    if parsed is not None:
                        return parsed
                    attempt += 1
                except Exception as exc:
                    if _is_model_missing(exc):
                        _state["dead"].add(model)
                        if _state["model"] == model:
                            _state["model"] = None
                        print(f"[llm] model '{model}' unavailable, trying the next one")
                        break  # next model
                    busy = "503" in str(exc) or "overloaded" in str(exc).lower() or "high demand" in str(exc).lower()
                    if _is_rate_limit(exc) or busy:
                        wait = _retry_after(exc) or (30 if busy else 600)
                        _state["model_cooldown"][model] = time.time() + wait
                        if _state["model"] == model:
                            _state["model"] = None
                        if wait > MAX_WAIT:  # a long (hourly/daily) limit is worth a line; per-minute ones aren't
                            print(f"[llm] {model} is resting until about "
                                  f"{time.strftime('%H:%M', time.localtime(time.time() + wait))}; using other models")
                        break  # try the next model right away
                    attempt += 1
                    if attempt > retries:
                        print(f"[llm] call failed, using rules for this item: {str(exc)[:160]}")
                        _state["last_failed"] = True
                        return None
                    time.sleep(2 ** attempt)  # network blip
            else:
                continue  # this model returned unparseable text every time: try the next one
        # went through every available model without an answer: loop (they may all be resting now)
        if not _candidates():
            break
        if all(_state["model_cooldown"].get(m, 0) <= time.time() for m in _candidates()):
            break  # nobody is resting, they just failed: give up on this item
    if not _candidates():
        _state["disabled"] = True
        print("[llm] no working model for this provider/key; continuing rules-only. "
              "Set LLM_MODEL in .env to a model your account can use.")
    _state["last_failed"] = True
    return None