"""Groq provider (official groq SDK). PROTOTYPE.
Role: INDEPENDENT second extractor, validator, adjudicator and reranker (not a dead backup).

Includes a client-side token-bucket limiter because the free tier caps tokens/minute.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import deque

from src.llm.base import LLMProvider, PermanentError, RawResponse, TransientError


def parse_groq_response(resp) -> RawResponse:
    """Pure parser for a groq ChatCompletion (or a mock with the same shape)."""
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise TransientError("no choices in Groq response", kind="empty_response")
    msg = choices[0].message
    text = getattr(msg, "content", None) or ""
    if not text.strip():
        raise TransientError("empty Groq content", kind="empty_response")
    u = getattr(resp, "usage", None)
    details = getattr(u, "completion_tokens_details", None) if u else None
    return RawResponse(
        text=text,
        input_tokens=getattr(u, "prompt_tokens", None) if u else None,
        output_tokens=getattr(u, "completion_tokens", None) if u else None,
        reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
        model=getattr(resp, "model", None),
    )


def _retry_after(e) -> float | None:
    resp = getattr(e, "response", None)
    h = getattr(resp, "headers", None) or {}
    v = h.get("retry-after") if hasattr(h, "get") else None
    try:
        return float(v) + 0.5 if v else None
    except ValueError:
        m = re.search(r"try again in ([\d.]+)s", str(e))
        return float(m.group(1)) + 0.5 if m else None


class TokenBudget:
    """Sliding 60 s window of consumed tokens; blocks until an estimated request fits."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.events: deque = deque()
        self.lock = threading.Lock()

    def acquire(self, estimate: int) -> None:
        while True:
            with self.lock:
                now = time.time()
                while self.events and now - self.events[0][0] > 60:
                    self.events.popleft()
                used = sum(t for _, t in self.events)
                if used + estimate <= self.per_minute or not self.events:
                    self.events.append((now, estimate))
                    return
                wait = 60 - (now - self.events[0][0]) + 0.2
            time.sleep(max(wait, 0.5))

    def correct(self, estimate: int, actual: int | None) -> None:
        if actual is None:
            return
        with self.lock:
            for i in range(len(self.events) - 1, -1, -1):
                if self.events[i][1] == estimate:
                    self.events[i] = (self.events[i][0], actual)
                    break


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, cfg: dict, **kw):
        super().__init__(cfg, **kw)
        self._client = None
        self.budget = TokenBudget(int(cfg.get("tokens_per_minute_budget", 7000)))

    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise PermanentError("GROQ_API_KEY not configured", kind="no_credentials")
            self._client = Groq(api_key=key, timeout=float(self.cfg.get("timeout_s", 90)), max_retries=0)
        return self._client

    def _call(self, model, system, prompt, schema, schema_name) -> RawResponse:
        import groq
        estimate = int((len(system) + len(prompt)) / 3.2) + 1400  # input chars/token + output allowance
        self.budget.acquire(estimate)
        kwargs = dict(model=model, temperature=self.cfg.get("temperature", 0.0),
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                      response_format={"type": "json_schema",
                                       "json_schema": {"name": schema_name, "strict": True, "schema": schema}})
        if self.cfg.get("reasoning_effort") and "gpt-oss" in model:
            kwargs["reasoning_effort"] = self.cfg["reasoning_effort"]
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except groq.RateLimitError as e:
            if re.search(r"per day|TPD|RPD", str(e)):
                raise PermanentError("daily quota exhausted", kind="daily_quota_exhausted")
            raise TransientError("rate limited", retry_after=_retry_after(e), kind="rate_limit")
        except (groq.APITimeoutError, groq.APIConnectionError):
            raise TransientError("timeout/connection", kind="timeout")
        except groq.InternalServerError:
            raise TransientError("server error", kind="unavailable")
        except groq.BadRequestError as e:
            # strict json_schema generation failures surface as 400 json_validate_failed -> retry once
            if "json_validate_failed" in str(e) or "Failed to generate JSON" in str(e):
                raise TransientError("json generation failed", kind="schema_generation_failed")
            raise PermanentError("bad request", kind="http_400")
        except groq.APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code and code >= 500:
                raise TransientError("server error", kind="unavailable")
            raise PermanentError(f"http {code}", kind=f"http_{code}")
        raw = parse_groq_response(resp)
        raw.extra["api_latency_ms"] = (time.perf_counter() - t0) * 1000  # excludes local rate-limit waiting
        total = (raw.input_tokens or 0) + (raw.output_tokens or 0)
        self.budget.correct(estimate, total or None)
        return raw
