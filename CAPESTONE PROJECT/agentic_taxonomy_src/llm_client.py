"""Claude client: plain Messages API + JSON text parsing. NO tool calling, NO sampling parameters.

Every request is built by build_request() and contains EXACTLY: model, max_tokens, system, messages.
temperature / top_p / top_k / tools / tool_choice / thinking / stop_sequences are never sent; the transport re-checks
this immediately before sending.

call(): topic-count guard -> cache lookup -> budget.authorize (persisted before sending) -> request ->
        JSON extraction (deterministic cleanup only) -> pydantic validation -> ONE repair request if invalid and the
        reserve allows -> cache store (valid responses only).
API 400 errors stop the run with the server message. 429/5xx/connection errors get one retry from the reserve.
Nothing is written to registry/assignments unless a response validated, so failures cannot corrupt outputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from .budget import ApiBudget, BudgetExceeded
from .cache import ResponseCache, cache_key
from .config import StopRun
from .provenance import append_jsonl, log, read_json, sha256_text, utc, write_json
from .schemas import PROMPT_VERSIONS, SCHEMAS, assert_no_topic_count_hint

ALLOWED_REQUEST_KEYS = frozenset({"model", "max_tokens", "system", "messages"})
FORBIDDEN_REQUEST_KEYS = frozenset({"temperature", "top_p", "top_k", "tools", "tool_choice", "thinking",
                                    "stop_sequences", "metadata", "service_tier"})


class ClaudeAPIError(RuntimeError):
    def __init__(self, status, kind, message, retryable=False):
        super().__init__(f"Claude API {kind} (status {status}): {message}")
        self.status, self.kind, self.retryable = status, kind, retryable


def build_request(model: str, max_tokens: int, system: str, messages: list) -> dict:
    """The ONLY place a request is constructed."""
    req = {"model": model, "max_tokens": int(max_tokens), "system": system, "messages": messages}
    check_request(req)
    return req


def check_request(req: dict) -> None:
    keys = set(req)
    bad = keys & FORBIDDEN_REQUEST_KEYS
    if bad or keys != ALLOWED_REQUEST_KEYS:
        raise StopRun(f"request kwargs must be exactly {sorted(ALLOWED_REQUEST_KEYS)}; got {sorted(keys)}"
                      + (f" (forbidden: {sorted(bad)})" if bad else ""))


def response_text(resp) -> str:
    """Join text blocks only (any non-text block, e.g. thinking, is ignored)."""
    return "".join(getattr(b, "text", "") or "" for b in (getattr(resp, "content", None) or [])
                   if getattr(b, "type", None) == "text")


class AnthropicTransport:
    def __init__(self, api_key: str, timeout: float = 300.0):
        import anthropic
        self._a = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout)   # retries are budgeted here

    def validate_model(self, model: str) -> dict:
        """Metadata request (models.retrieve) - not a taxonomy call."""
        try:
            m = self.client.models.retrieve(model)
        except self._a.NotFoundError:
            raise StopRun(f"model '{model}' is not available to this API key")
        except (self._a.AuthenticationError, self._a.PermissionDeniedError):
            raise StopRun("Anthropic authentication failed - check ANTHROPIC_API_KEY (the key is never printed)")
        return {"id": m.id, "display_name": getattr(m, "display_name", ""), "created_at": str(getattr(m, "created_at", ""))}

    def create(self, req: dict) -> dict:
        check_request(req)
        a = self._a
        try:
            resp = self.client.messages.create(**req)
        except a.BadRequestError as e:
            raise ClaudeAPIError(400, "bad_request", _msg(e))
        except (a.AuthenticationError, a.PermissionDeniedError) as e:
            raise ClaudeAPIError(getattr(e, "status_code", 401), "authentication", "check ANTHROPIC_API_KEY")
        except a.NotFoundError as e:
            raise ClaudeAPIError(404, "not_found", _msg(e))
        except a.RateLimitError as e:
            raise ClaudeAPIError(429, "rate_limited", _msg(e), retryable=True)
        except (a.APITimeoutError, a.APIConnectionError) as e:
            raise ClaudeAPIError(None, "connection", type(e).__name__, retryable=True)
        except a.APIStatusError as e:
            code = getattr(e, "status_code", None)
            raise ClaudeAPIError(code, "server_error" if (code or 0) >= 500 else "api_error", _msg(e),
                                 retryable=(code or 0) >= 500)
        u = getattr(resp, "usage", None)
        return {"text": response_text(resp), "stop_reason": getattr(resp, "stop_reason", None),
                "model": getattr(resp, "model", req["model"]),
                "usage": {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)}}


def _msg(e):
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        return str(body.get("error", {}).get("message", body))[:500]
    return str(e)[:500]


class MockTransport:
    """Test double. `responder(req) -> str | Exception`; records every request it receives."""

    def __init__(self, responder=None, queue=None):
        self.responder, self.queue, self.requests = responder, list(queue or []), []

    def validate_model(self, model):
        return {"id": model, "display_name": "mock", "created_at": ""}

    def create(self, req):
        check_request(req)
        self.requests.append(req)
        out = self.queue.pop(0) if self.queue else self.responder(req)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps(out)
        return {"text": text, "stop_reason": "end_turn", "model": req["model"],
                "usage": {"input_tokens": len(req["messages"][-1]["content"]) // 4, "output_tokens": len(text) // 4}}


# ---------------------------------------------------------------- JSON extraction
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str):
    """Deterministic cleanup only: strip fences / surrounding prose, drop trailing commas. Returns (obj, steps) or
    raises ValueError."""
    steps, t = [], (text or "").strip()
    if not t:
        raise ValueError("empty response text")
    m = _FENCE.search(t)
    if m:
        t, steps = m.group(1).strip(), steps + ["stripped_code_fence"]
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise ValueError("no JSON object found in response")
        t, steps = t[i:j + 1], steps + ["trimmed_surrounding_text"]
    try:
        return json.loads(t), steps
    except json.JSONDecodeError:
        t2 = _TRAILING_COMMA.sub(r"\1", t)
        try:
            return json.loads(t2), steps + ["removed_trailing_commas"]
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}")


def model_validation(transport, model, budget: ApiBudget, provenance_dir: Path) -> dict:
    """Validate the configured model once (metadata request, logged separately; never repeated for the same model)."""
    p = Path(provenance_dir) / "model_validation.json"
    prev = read_json(p)
    if prev and prev.get("model") == model and prev.get("status") == "ok":
        return prev
    info = transport.validate_model(model)
    budget.note_metadata(f"models.retrieve({model})")
    res = {"model": model, "status": "ok", "info": info, "utc": utc()}
    write_json(p, res)
    return res


class ClaudeClient:
    def __init__(self, transport, budget: ApiBudget, cache: ResponseCache, model: str, provenance_dir, max_tokens: int):
        if not model:
            raise StopRun("ANTHROPIC_MODEL is not configured")
        self.t, self.budget, self.cache, self.model, self.max_tokens = transport, budget, cache, model, max_tokens
        self.prov = Path(provenance_dir)
        self.calls_log = self.prov / "api_calls.jsonl"

    def _send(self, role, kind, system, messages, note):
        req = build_request(self.model, self.max_tokens, system, messages)
        n = self.budget.authorize(role, kind=kind, note=note)
        try:
            out = self.t.create(req)
        except ClaudeAPIError as e:
            self.budget.record(n, f"error:{e.kind}:{e.status}")
            append_jsonl(self.calls_log, {"role": role, "kind": kind, "call_no": n, "note": note, "error": str(e)})
            raise
        self.budget.record(n, "received", out.get("usage"))
        return n, out

    def call(self, role, system, instruction, data, contract, registry_version, note=""):
        """Returns (validated dict, info). Raises BudgetExceeded / ClaudeAPIError / StopRun."""
        assert_no_topic_count_hint(system + "\n" + instruction + "\n" + contract)
        user = f"{instruction}\n\n{data}\n\n{contract}"
        schema = SCHEMAS[role]
        ih = sha256_text(system + "\n" + user)
        key = cache_key(self.model, role, PROMPT_VERSIONS[role], ih, registry_version)
        hit = self.cache.get(role, key)
        if hit is not None:
            append_jsonl(self.calls_log, {"role": role, "cache_hit": True, "key": key, "note": note})
            return hit["data"], {"cache_hit": True, "key": key, "usage": hit.get("usage")}
        messages = [{"role": "user", "content": user}]
        kind, attempts = "normal", 0
        while True:
            try:
                n, out = self._send(role, kind, system, messages, note)
            except ClaudeAPIError as e:
                if e.retryable and attempts == 0 and self.budget.can_reserve():
                    attempts, kind = 1, "retry"
                    log(f"  {role}: {e.kind} - one retry from reserve")
                    continue
                if e.status == 400:
                    raise StopRun(f"Claude rejected the request (400) for role '{role}': {e}")
                raise
            try:
                obj, steps = extract_json(out["text"])
                data = schema.model_validate(obj).model_dump()
            except (ValueError, ValidationError) as err:
                problem = str(err)[:600]
                self.budget.record(n, "invalid_json")
                append_jsonl(self.calls_log, {"role": role, "kind": kind, "call_no": n, "note": note,
                                              "invalid": problem, "stop_reason": out.get("stop_reason")})
                if kind == "repair" or not self.budget.can_reserve():
                    raise BudgetExceeded(f"{role}: response was not valid JSON for the schema and no repair is "
                                         f"possible ({problem[:200]})")
                trunc = out.get("stop_reason") == "max_tokens"
                messages = messages[:1] + [
                    {"role": "assistant", "content": out["text"][:20000] or "(empty)"},
                    {"role": "user", "content": ("Your previous reply " + ("was cut off by the output limit. " if trunc else
                                                 "did not match the required JSON structure. ") +
                                                 f"Problem: {problem}\nReturn ONLY the complete corrected JSON"
                                                 + (", using shorter text fields" if trunc else "") +
                                                 " - no Markdown, no explanation.\n" + contract)}]
                kind = "repair"
                log(f"  {role}: invalid JSON ({problem[:120]}) - one repair request from reserve")
                continue
            entry = {"model": out.get("model", self.model), "role": role, "prompt_version": PROMPT_VERSIONS[role],
                     "input_hash": ih, "registry_version": registry_version, "data": data, "usage": out.get("usage"),
                     "cleanup_steps": steps, "attempt_kind": kind, "created_utc": utc()}
            self.cache.put(role, key, entry)
            self.budget.record(n, "ok", out.get("usage"))
            append_jsonl(self.calls_log, {"role": role, "kind": kind, "call_no": n, "cache_hit": False, "key": key,
                                          "note": note, "usage": out.get("usage"), "stop_reason": out.get("stop_reason"),
                                          "cleanup_steps": steps})
            return data, {"cache_hit": False, "key": key, "usage": out.get("usage"), "stop_reason": out.get("stop_reason")}
