"""Shared LLM client: Databricks Unity Gateway, unified Responses API (OpenAI-compatible).

Auth: a short-lived OAuth token minted with `databricks auth token --profile $LLM_PROFILE`
(default bops-prod-new), refreshed before expiry; or $DATABRICKS_TOKEN if set. No credential
is ever written to disk. Every call is cached in llm_cache/{model}.jsonl keyed by
sha256(model, input, params), so reruns are free and reproducible.
"""
import hashlib
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from openai import APIStatusError, APITimeoutError, APIConnectionError, OpenAI

from retrieval_eval.common import HERE

# Gateway model names (Unity Catalog system.ai models).
GPT_OSS = "system.ai.gpt-oss-120b"
QWEN = "system.ai.qwen35-122b-a10b"
LLAMA = "system.ai.meta-llama-3-3-70b-instruct"  # alternative HyDE generator (non-reasoning, open-weight)
CHAT_MODELS = {LLAMA}  # services that reject the Responses API: called via the unified Chat Completions API
PROFILE = os.environ.get("LLM_PROFILE", "bops-prod-new")
HOST = os.environ.get("DATABRICKS_HOST", "https://adb-7405609759660295.15.azuredatabricks.net").rstrip("/")
CACHE = HERE / "llm_cache"
WORKERS = int(os.environ.get("LLM_WORKERS", "8"))  # parallel requests in LLM.map


class _Token:
    """Thread-safe holder for the gateway bearer token; mints a new one via the Databricks CLI when needed."""

    def __init__(self):
        self.lock, self.value, self.expiry = threading.Lock(), None, 0.0

    def get(self):
        """Return a valid token: $DATABRICKS_TOKEN if set, else a cached OAuth token refreshed near expiry."""
        if os.environ.get("DATABRICKS_TOKEN"):
            return os.environ["DATABRICKS_TOKEN"]
        with self.lock:
            # Refresh 5 minutes before expiry so in-flight requests don't fail mid-call.
            if self.value is None or time.time() > self.expiry - 300:
                out = subprocess.run(["databricks", "auth", "token", "--profile", PROFILE],
                                     capture_output=True, text=True, check=True).stdout
                d = json.loads(out)
                self.value = d["access_token"]
                exp = d.get("expiry")
                # Fall back to a 30-minute lifetime if the CLI omits the expiry.
                self.expiry = (datetime.fromisoformat(exp).timestamp() if exp else time.time() + 1800)
            return self.value


_token = _Token()  # shared by all LLM instances and threads


class LLM:
    """Cached, retrying client for one gateway model. Call it like a function; use map() for batches."""

    def __init__(self, model):
        self.model = model
        CACHE.mkdir(exist_ok=True)
        self.path = CACHE / f"{model}.jsonl"
        self.cache = {}
        # Load the append-only JSONL cache from previous runs.
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial line from an interrupted write
                self.cache[r["key"]] = r
        self.wlock = threading.Lock()  # guards cache file appends and stats across worker threads
        self.stats = {"calls": 0, "cache_hits": 0, "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0,
                      "errors": 0, "seconds": 0.0}

    def _client(self):
        """New OpenAI-compatible client pointed at the gateway; built per call so it picks up a fresh token.
        max_retries=0 because __call__ does its own retry/backoff."""
        return OpenAI(api_key=_token.get(), base_url=f"{HOST}/ai-gateway/mlflow/v1", timeout=300, max_retries=0)

    def key(self, input_, params):
        """Cache key: sha256 of the canonical JSON of (model, input, params)."""
        blob = json.dumps({"model": self.model, "input": input_, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def __call__(self, input_, tag="", **params):
        """input_: str or list of Responses-API messages. Returns output text ('' on hard failure).
        A param named `_salt` is part of the cache key but not sent (e.g. HyDE sample index)."""
        k = self.key(input_, params)  # computed before _salt is stripped, so the salt affects the key
        params = {p: v for p, v in params.items() if p != "_salt"}
        if k in self.cache:
            with self.wlock:
                self.stats["cache_hits"] += 1
            return self.cache[k]["text"]
        # Up to 8 attempts with exponential backoff (capped at 60s) plus jitter.
        for attempt in range(8):
            try:
                t = time.perf_counter()
                if self.model in CHAT_MODELS:
                    # Chat Completions: string input becomes one user message; max_output_tokens -> max_tokens.
                    msgs = [{"role": "user", "content": input_}] if isinstance(input_, str) else input_
                    cp = {("max_tokens" if p == "max_output_tokens" else p): v for p, v in params.items()}
                    r = self._client().chat.completions.create(model=self.model, messages=msgs, **cp)
                    dt = time.perf_counter() - t
                    u = getattr(r, "usage", None)
                    rec = {"key": k, "tag": tag, "text": (r.choices[0].message.content if r.choices else "") or "",
                           "seconds": dt, "in": getattr(u, "prompt_tokens", 0) if u else 0,
                           "out": getattr(u, "completion_tokens", 0) if u else 0, "reasoning": 0}
                else:
                    r = self._client().responses.create(model=self.model, input=input_, **params)
                    dt = time.perf_counter() - t
                    u = getattr(r, "usage", None)
                    rec = {"key": k, "tag": tag, "text": r.output_text or "", "seconds": dt,
                           "in": getattr(u, "input_tokens", 0) if u else 0,
                           "out": getattr(u, "output_tokens", 0) if u else 0,
                           "reasoning": (getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0)
                           if u else 0}
                # Persist immediately so an interrupted run keeps every completed call.
                with self.wlock:
                    self.cache[k] = rec
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    s = self.stats
                    s["calls"] += 1; s["in_tokens"] += rec["in"]; s["out_tokens"] += rec["out"]
                    s["reasoning_tokens"] += rec["reasoning"]; s["seconds"] += dt
                return rec["text"]
            except APIStatusError as e:
                if e.status_code in (401, 403) and attempt == 0:
                    _token.expiry = 0  # force token refresh once
                    continue
                # Only retry transient statuses (timeouts, rate limits, server errors); others are bugs.
                if e.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    raise
            except (APITimeoutError, APIConnectionError):
                pass
            time.sleep(min(60, 2 ** attempt) + random.random())
        # All retries exhausted: count the failure and return empty text instead of crashing the run.
        with self.wlock:
            self.stats["errors"] += 1
        return ""

    def map(self, jobs, desc=""):
        """jobs: list of (input_, tag, params). Runs WORKERS-way parallel, preserves order."""
        t0 = time.perf_counter()
        done = [0]  # list so the nested function can mutate the counter

        def run(j):
            """Run one job and print progress every 500 completions."""
            out = self(j[0], j[1], **j[2])
            with self.wlock:
                done[0] += 1
                if done[0] % 500 == 0:
                    print(f"  [{self.model}] {desc}: {done[0]}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
            return out

        with ThreadPoolExecutor(WORKERS) as ex:
            return list(ex.map(run, jobs))

    def report(self, desc=""):
        """Print call/cache/token statistics and return a copy of them."""
        s = self.stats
        print(f"  [{self.model}] {desc}: {s['calls']} calls, {s['cache_hits']} cache hits, {s['errors']} errors, "
              f"tokens in {s['in_tokens']:,} / out {s['out_tokens']:,} (reasoning {s['reasoning_tokens']:,})", flush=True)
        return dict(s)
