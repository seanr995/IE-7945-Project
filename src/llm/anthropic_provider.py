"""Anthropic / Claude provider (official anthropic SDK). PROTOTYPE.
Role: OPTIONAL third extractor for a separate experiment arm on the same frozen postings.
It is not part of the Gemini/Groq baseline and is never used as a silent fallback.

Same contract as the other providers: the shared JSON schema is sent as structured output
(output_config.format = json_schema) and the response is re-validated locally with pydantic.
The key is read only from the ANTHROPIC_API_KEY environment variable (.env); it is never logged.
Server-side refusal fallbacks are deliberately NOT enabled: a different model answering would
confound the provider comparison, so a refusal is recorded as a failed call instead.
"""
from __future__ import annotations

import os
import time

from src.llm.base import LLMProvider, PermanentError, RawResponse, TransientError


def parse_anthropic_response(resp) -> RawResponse:
    """Pure parser for an anthropic Message (or a mock with the same shape)."""
    stop = getattr(resp, "stop_reason", None)
    if stop == "refusal":
        raise PermanentError("model declined the request", kind="refusal")
    text = "".join(getattr(b, "text", "") or "" for b in (getattr(resp, "content", None) or [])
                   if getattr(b, "type", None) == "text")
    if not text.strip():
        raise TransientError(f"empty Claude response (stop_reason={stop})", kind="empty_response")
    u = getattr(resp, "usage", None)
    return RawResponse(text=text, input_tokens=getattr(u, "input_tokens", None) if u else None,
                       output_tokens=getattr(u, "output_tokens", None) if u else None,
                       model=getattr(resp, "model", None), extra={"stop_reason": stop})


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, cfg: dict, **kw):
        super().__init__(cfg, **kw)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise PermanentError("ANTHROPIC_API_KEY not configured", kind="no_credentials")
            # retries are handled by LLMProvider.generate_json (shared backoff + usage logging)
            self._client = anthropic.Anthropic(api_key=key, timeout=float(self.cfg.get("timeout_s", 180)), max_retries=0)
        return self._client

    def _call(self, model, system, prompt, schema, schema_name) -> RawResponse:
        import anthropic
        output_config = {"format": {"type": "json_schema", "schema": schema}}
        if self.cfg.get("effort"):
            output_config["effort"] = self.cfg["effort"]
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=model, max_tokens=int(self.cfg.get("max_tokens", 16000)), system=system,
                messages=[{"role": "user", "content": prompt}], output_config=output_config)
        except anthropic.RateLimitError as e:
            ra = e.response.headers.get("retry-after") if getattr(e, "response", None) is not None else None
            raise TransientError("rate limited", retry_after=float(ra) + 0.5 if ra else None, kind="rate_limit")
        except (anthropic.APITimeoutError, anthropic.APIConnectionError):
            raise TransientError("timeout/connection", kind="timeout")
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            raise PermanentError("authentication/permission error", kind="auth")
        except anthropic.NotFoundError:
            raise PermanentError("model or endpoint not found", kind="http_404")
        except anthropic.BadRequestError:
            raise PermanentError("bad request", kind="http_400")
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:  # includes 529 overloaded
                raise TransientError("server error", kind="unavailable")
            raise PermanentError(f"http {e.status_code}", kind=f"http_{e.status_code}")
        raw = parse_anthropic_response(resp)
        raw.extra["api_latency_ms"] = (time.perf_counter() - t0) * 1000
        return raw
