"""Gemini provider (official google-genai SDK). PROTOTYPE. Role: PRIMARY extractor."""
from __future__ import annotations

import os

from src.llm.base import LLMProvider, PermanentError, RawResponse, TransientError


def parse_gemini_response(resp) -> RawResponse:
    """Pure parser for a google-genai GenerateContentResponse (or a mock with the same shape)."""
    text = getattr(resp, "text", None)
    if not text:
        cands = getattr(resp, "candidates", None) or []
        parts = []
        for c in cands:
            content = getattr(c, "content", None)
            for p in (getattr(content, "parts", None) or []):
                if getattr(p, "text", None) and not getattr(p, "thought", False):
                    parts.append(p.text)
        text = "".join(parts)
    if not text:
        raise TransientError("empty Gemini response", kind="empty_response")
    u = getattr(resp, "usage_metadata", None)
    return RawResponse(
        text=text,
        input_tokens=getattr(u, "prompt_token_count", None) if u else None,
        output_tokens=getattr(u, "candidates_token_count", None) if u else None,
        reasoning_tokens=getattr(u, "thoughts_token_count", None) if u else None,
        model=getattr(resp, "model_version", None),
    )


def classify_gemini_error(e: Exception) -> Exception:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    msg = str(e)
    if code == 429 or "RESOURCE_EXHAUSTED" in msg:
        return TransientError("rate limited", kind="rate_limit")
    if code in (500, 502, 503, 504) or "UNAVAILABLE" in msg or "DEADLINE" in msg:
        return TransientError("unavailable", kind="unavailable")
    if code in (400, 401, 403, 404):
        return PermanentError(f"client error {code}", kind=f"http_{code}")
    return TransientError(type(e).__name__, kind="network")


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, cfg: dict, **kw):
        super().__init__(cfg, **kw)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise PermanentError("GEMINI_API_KEY not configured", kind="no_credentials")
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=int(self.cfg.get("timeout_s", 120)) * 1000))
        return self._client

    def _call(self, model, system, prompt, schema, schema_name) -> RawResponse:
        from google.genai import types
        conf = types.GenerateContentConfig(
            system_instruction=system, temperature=self.cfg.get("temperature", 0.0),
            response_mime_type="application/json", response_json_schema=schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        if self.cfg.get("thinking_level") and model.startswith("gemini-3"):
            conf.thinking_config = types.ThinkingConfig(thinking_level=self.cfg["thinking_level"])
        try:
            resp = self.client.models.generate_content(model=model, contents=prompt, config=conf)
        except (TransientError, PermanentError):
            raise
        except Exception as e:  # SDK errors -> retry classification
            raise classify_gemini_error(e)
        raw = parse_gemini_response(resp)
        raw.model = model
        return raw
