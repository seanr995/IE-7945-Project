"""Gemini runtime (google-genai) behind a small provider abstraction.

batch(): cache lookup -> one structured-output generation request for a whole batch -> Pydantic + business validation per record ->
ONE batched repair request for the failed ids only (global max 2) -> unresolved. Never one call per topic/task.
Rate limits: minimum spacing between requests, 429 handling with retry-after / exponential backoff (max 2 retries), daily-quota
detection, authentication detection, adaptive 20 -> 10 splitting. Every attempt is written to the ledger before it is sent."""
from __future__ import annotations

import json
import os
import re
import time

from .budget import Ledger
from .config import (MIN_SPLIT_BATCH, CallBudgetError, GeminiAuthError, GeminiQuotaError, Settings, StopRun, log, read_json, sha256_text, write_json)

PROMPT_VERSION = "sprint4_gemini_v1"


class TooLarge(Exception):
    pass


class GeminiTransport:
    """The only place that touches the Google GenAI SDK."""

    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str, schema, max_output_tokens: int) -> dict:
        t = self._types
        cfg = t.GenerateContentConfig(system_instruction=system, response_mime_type="application/json", response_schema=schema,
                                      temperature=0.0, max_output_tokens=max_output_tokens)
        r = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
        u = getattr(r, "usage_metadata", None)
        out_tok = (getattr(u, "candidates_token_count", None) or getattr(u, "response_token_count", None) or 0) + (getattr(u, "thoughts_token_count", None) or 0)
        fr = ""
        try:
            fr = str(r.candidates[0].finish_reason)
        except Exception:                                    # noqa: BLE001
            pass
        return {"text": getattr(r, "text", None) or "", "finish_reason": fr,
                "usage": {"input_tokens": getattr(u, "prompt_token_count", 0) or 0, "output_tokens": out_tok}}


class MockTransport:
    """Offline test double: no network, no SDK. responder(system, user) -> str (may raise to simulate API errors)."""

    def __init__(self, responder):
        self.responder, self.requests = responder, []

    def generate(self, system, user, schema, max_output_tokens) -> dict:
        self.requests.append({"system": system, "user": user})
        text = self.responder(system, user)
        return {"text": text, "finish_reason": "STOP", "usage": {"input_tokens": (len(system) + len(user)) // 4, "output_tokens": len(text) // 4}}


def classify(e: Exception):
    """-> (kind, retry_delay_seconds|None). kinds: auth, quota_daily, rate, too_large, server, other."""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    status = str(getattr(e, "status", "") or "").upper()
    msg = (str(getattr(e, "message", "") or "") + " " + str(e)).lower()
    if (code in (401, 403) or "UNAUTHENTICATED" in status or "PERMISSION_DENIED" in status or "api key not valid" in msg
            or "api_key_invalid" in msg or "invalid api key" in msg or "invalid x-api-key" in msg):
        return "auth", None
    if code == 429 or "RESOURCE_EXHAUSTED" in status or "quota" in msg and "exceed" in msg:
        if any(x in msg for x in ("perday", "per day", "per_day", "daily", "requestsperday")):
            return "quota_daily", None
        m = re.search(r"retry in ([0-9.]+)\s*s", msg) or re.search(r"retrydelay['\": ]+([0-9.]+)\s*s", msg) or re.search(r"retry-after['\": ]+([0-9.]+)", msg)
        return "rate", float(m.group(1)) if m else None
    if code == 400 and ("too large" in msg or "exceeds the maximum" in msg or ("token" in msg and "exceed" in msg)):
        return "too_large", None
    if (isinstance(code, int) and code >= 500) or "UNAVAILABLE" in status or "INTERNAL" in status:
        return "server", None
    return "other", None


def _json(text: str):
    t = (text or "").strip()
    if not t:
        raise ValueError("empty response")
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"invalid JSON: {e}")


class LLM:
    def __init__(self, s: Settings, ledger: Ledger, transport, sleep=time.sleep, clock=time.monotonic):
        self.s, self.ledger, self.t, self.sleep, self.clock = s, ledger, transport, sleep, clock
        self.last_call = None
        s.cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ one real request (retries on rate/server errors)
    def _wait(self):
        if self.last_call is not None and self.s.min_seconds > 0:
            gap = self.s.min_seconds - (self.clock() - self.last_call)
            if gap > 0:
                self.sleep(gap)

    def _real_call(self, role, system, schema, user, batch_id, n_items, kind):
        attempt = 0
        while True:
            self._wait()
            n = self.ledger.authorize(role, batch_id, n_items, kind if attempt == 0 else "retry")
            self.last_call = self.clock()
            try:
                out = self.t.generate(system, user, schema, self.s.max_output_tokens)
            except Exception as e:                                        # noqa: BLE001
                ck, delay = classify(e)
                self.ledger.finish(n, "error", error_type=ck)
                if ck == "auth":
                    raise GeminiAuthError("GEMINI_AUTHENTICATION_FAILED: Gemini rejected the API key. Check GEMINI_API_KEY in "
                                          "SPRINT4_READY/.env (or the project-folder .env). Completed cached work is preserved.")
                if ck == "quota_daily":
                    raise GeminiQuotaError("GEMINI_QUOTA_EXHAUSTED: daily/project quota exhausted. Cache and ledger are saved; rerun later to resume.")
                if ck == "too_large":
                    raise TooLarge(str(e)[:200])
                if ck in ("rate", "server"):
                    if attempt >= self.s.max_retries or not self.ledger.can_extra(1):
                        raise GeminiQuotaError(f"GEMINI_QUOTA_EXHAUSTED: {ck} errors persisted after {attempt} retries (or the call cap leaves no room). "
                                               f"Cache and ledger are saved; rerun later to resume.")
                    wait = min(delay + 1 if delay else max(15.0, self.s.min_seconds) * (2 ** attempt), 180.0)
                    log(f"  {role}: {ck} error - waiting {wait:.0f}s then retry {attempt + 1}/{self.s.max_retries}")
                    self.sleep(wait)
                    attempt += 1
                    continue
                raise StopRun(f"GEMINI_ERROR ({type(e).__name__}): {str(e)[:300]}")
            hit_max = "MAX_TOKENS" in (out.get("finish_reason") or "").upper()
            self.ledger.finish(n, "max_tokens" if hit_max else "ok", out.get("usage"), "max_tokens" if hit_max else "")
            if hit_max:
                raise TooLarge("output truncated (MAX_TOKENS)")
            return out

    # ------------------------------------------------------------ batch with validation, repair, adaptive split
    def batch(self, role, system, schema, ids, make_payload, extract, validate, batch_id, kind="normal", splittable=True):
        from .agents import user_prompt
        ids = list(ids)
        user = user_prompt(role, make_payload(ids, None))
        key = sha256_text("|".join([self.s.provider, self.s.model, role, PROMPT_VERSION, system, user]))
        cp = self.s.cache / role / f"{key}.json"
        hit = read_json(cp)
        if hit is not None:
            self.ledger.cache_hit(role, batch_id, len(ids))
            log(f"  {role} {batch_id}: cache hit (0 Gemini calls)")
            return hit["results"], hit["failed"]
        results, failed = {}, {}
        try:
            out = self._real_call(role, system, schema, user, batch_id, len(ids), kind)
            raw = self._parse(out, schema, extract)
            err_all = None
        except TooLarge:
            if splittable and len(ids) >= 2 and self.ledger.can_extra(2):
                half = (len(ids) + 1) // 2 if len(ids) > MIN_SPLIT_BATCH else (len(ids) + 1) // 2
                log(f"  {role} {batch_id}: request too large - splitting {len(ids)} -> {half}+{len(ids) - half}")
                r1, f1 = self.batch(role, system, schema, ids[:half], make_payload, extract, validate, batch_id + ".a", "split", splittable)
                r2, f2 = self.batch(role, system, schema, ids[half:], make_payload, extract, validate, batch_id + ".b", "split", splittable)
                results, failed = {**r1, **r2}, {**f1, **f2}
                write_json(cp, {"results": results, "failed": failed})
                return results, failed
            failed = {i: "request/response too large and no call budget left to split" for i in ids}
            write_json(cp, {"results": {}, "failed": failed})
            return {}, failed
        except ValueError as e:
            raw, err_all = None, str(e)[:200]
        self._validate_into(ids, raw, err_all, validate, results, failed)
        if failed and self.ledger.can_extra(1) and self.ledger.kind_count("repair") < self.s.max_repair_calls:
            fids = list(failed)
            errs = dict(failed)
            log(f"  {role} {batch_id}: {len(fids)} invalid record(s) - one batched repair request")
            ruser = user_prompt(role, make_payload(fids, errs))
            try:
                out = self._real_call(role, system, schema, ruser, batch_id + ".repair", len(fids), "repair")
                raw2 = self._parse(out, schema, extract)
                err2 = None
            except TooLarge:
                raw2, err2 = None, "repair request too large"
            except ValueError as e:
                raw2, err2 = None, str(e)[:200]
            res2, fail2 = {}, {}
            self._validate_into(fids, raw2, err2, validate, res2, fail2)
            results.update(res2)
            failed = fail2
        write_json(cp, {"results": results, "failed": failed})
        return results, failed

    def _parse(self, out, schema, extract):
        obj = schema.model_validate(_json(out["text"])).model_dump()
        return extract(obj)

    @staticmethod
    def _validate_into(ids, raw, err_all, validate, results, failed):
        for i in ids:
            if raw is None:
                failed[i] = err_all or "no response"
                continue
            if i not in raw:
                failed[i] = "missing from response"
                continue
            try:
                results[i] = validate(raw[i], i)
            except (ValueError, TypeError, KeyError) as e:
                failed[i] = str(e)[:200]
