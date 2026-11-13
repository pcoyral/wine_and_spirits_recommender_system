"""An LLM behind a disk cache (CLAUDE.md rule 2): a local model through
Ollama (casual test queries, S2) or Gemini's API (simulated queries, S2b).

Every answer is saved as data/llm_cache/<key>.json, where key is a hash of
the model, its digest, the options (temperature 0, fixed seed) and the
prompt. The cache files are committed: temperature 0 + seed gives the same
answer on the same machine and Ollama version, but that is not guaranteed
elsewhere, so the cache is the record of what the LLM wrote. A cached
prompt is never sent to the model again.

The LLM only writes text (casual queries now, explanations in S3); it never
ranks wines.
"""
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


class CachedLLM:
    def __init__(self, model, options, cache_dir, url, digest=None, call=None):
        """call: function prompt -> text. Left as None, Ollama's HTTP API at
        `url` is used; tests pass a fake function instead."""
        self.model = model
        self.digest = digest
        self.options = options
        self.cache_dir = Path(cache_dir)
        self.url = url
        self.call = call or self.ask_ollama
        self.n_calls = 0  # prompts actually sent to the model (not from cache)

    def key(self, prompt):
        data = {"model": self.model, "digest": self.digest, "options": self.options, "prompt": prompt}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def generate(self, prompt):
        path = self.cache_dir / f"{self.key(prompt)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.digest is None:
            # The digest is part of the cache key: pinning it later would orphan
            # every cached answer, and a re-run might write different queries.
            raise RuntimeError("Pin llm.digest in config.yaml (ID column of `ollama list`) "
                               "before generating anything new.")
        response = self.call(prompt)
        self.n_calls += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "digest": self.digest, "options": self.options,
                  "prompt": prompt, "response": response}
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return response

    def ask_ollama(self, prompt):
        body = {"model": self.model, "prompt": prompt, "stream": False, "options": self.options}
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())["response"]


def gemini_text(data):
    """The text of a Gemini generateContent answer; "" if Gemini gave no text
    (for example a blocked prompt), which the caller then flags."""
    candidates = data.get("candidates") or [{}]
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


def gemini_call(model, url, options, pause_seconds=0, key_name="GEMINI_API_KEY", waits=(30, 60, 120)):
    """A function prompt -> text that asks Gemini through its HTTP API.

    The API key is read from the environment variable (a GitHub secret in
    the workflow); it is never written to the repo or the cache. On "too
    many requests" (HTTP 429) or "busy" (503) it waits and tries again; if
    the quota is used up it stops, and a later run continues from the cache."""
    def call(prompt):
        key = os.environ.get(key_name)
        if not key:
            raise SystemExit(f"Set the environment variable {key_name} (GitHub secret) to call Gemini.")
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": options["temperature"], "seed": options["seed"]}}
        request = urllib.request.Request(url.format(model=model), data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})
        for wait in [*waits, None]:
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    text = gemini_text(json.loads(response.read()))
                time.sleep(pause_seconds)   # stay under the free tier's requests per minute
                return text
            except urllib.error.HTTPError as error:
                if error.code not in (429, 503) or wait is None:
                    reason = error.read().decode(errors="replace")[:300]   # Gemini's own explanation
                    raise SystemExit(f"Gemini refused the request (HTTP {error.code}): {reason}\nIf the free "
                                     "quota is used up, run again later: answers so far are in the cache.")
                log.warning("Gemini is busy (HTTP %d), waiting %d s", error.code, wait)
                time.sleep(wait)
    return call


def gemini_from_config(cfg, call=None):
    """The cached Gemini LLM for the simulated queries (config: simulation.llm)."""
    from sensoryrec.config import repo_path
    c = cfg["simulation"]["llm"]
    if not c["model"]:
        raise SystemExit("Set simulation.llm.model in config.yaml to the exact Gemini model name "
                         "(as in the course notebook) before `make simulate`.")
    options = {"temperature": c["temperature"], "seed": c["seed"]}
    call = call or gemini_call(c["model"], c["url"], options, c["pause_seconds"])
    # For an API model the pinned model name plays the role of the digest.
    return CachedLLM(c["model"], options, repo_path(cfg["llm"]["cache_dir"]), c["url"], digest=c["model"], call=call)


def from_config(cfg, call=None):
    from sensoryrec.config import repo_path
    c = cfg["llm"]
    return CachedLLM(c["model"], {"temperature": c["temperature"], "seed": c["seed"]},
                     repo_path(c["cache_dir"]), c["url"], digest=c["digest"], call=call)
