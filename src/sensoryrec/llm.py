"""A local LLM (Ollama) behind a disk cache (CLAUDE.md rule 2).

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
        if digest is None:
            log.warning("llm.digest is not pinned in config.yaml: copy it from `ollama list`")

    def key(self, prompt):
        data = {"model": self.model, "digest": self.digest, "options": self.options, "prompt": prompt}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def generate(self, prompt):
        path = self.cache_dir / f"{self.key(prompt)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
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


def from_config(cfg, call=None):
    from sensoryrec.config import repo_path
    c = cfg["llm"]
    return CachedLLM(c["model"], {"temperature": c["temperature"], "seed": c["seed"]},
                     repo_path(c["cache_dir"]), c["url"], digest=c["digest"], call=call)
