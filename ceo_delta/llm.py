"""LLM client for the vLLM OpenAI-compatible server.

Handles two quirks of the served reasoning model (Qwen3.6-35B-A3B):
  * the answer arrives in `message.content` but the model burns tokens in a
    separate `reasoning` channel first, so we request generous max_tokens;
  * JSON is requested in-prompt and extracted defensively (the model wraps it
    in prose / fences sometimes).

If the server is unreachable and config.llm_allow_stub is set, a deterministic
stub keeps the whole pipeline runnable offline (demos, CI, cold-start tests).
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
import os 
from .config import Config, DEFAULT


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or DEFAULT
        self._server_ok: Optional[bool] = None
        self.total_tokens = 0  # running cost meter (used by reflection budget)
        self._call_log: list = []  # accumulates during a run; flushed by Orchestrator

    def reset_call_log(self) -> None:
        self._call_log = []

    def flush_call_log(self) -> list:
        out = self._call_log
        self._call_log = []
        return out

    # -- public ---------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], *, max_tokens: int | None = None,
             temperature: float | None = None, tag: str = "") -> str:
        import time as _time
        t0 = _time.time()
        stub_used = False
        try:
            response = self._chat_remote(messages, max_tokens, temperature)
        except (urllib.error.URLError, urllib.error.HTTPError, LLMError, TimeoutError, OSError) as e:
            print("DEBUG ERROR:", type(e).__name__, str(e))
            if self.cfg.llm_allow_stub:
                response = self._chat_stub(messages)
                stub_used = True
            else:
                raise LLMError(str(e))
        latency = round(_time.time() - t0, 3)
        self._call_log.append({
            "tag":      tag,
            "latency_s": latency,
            "stub":     stub_used,
            "prompt":   [{"role": m["role"], "content": m["content"][:800]} for m in messages],
            "response": response[:1200],
        })
        return response

    def chat_json(self, messages: List[Dict[str, str]], *, max_tokens: int | None = None,
                  tag: str = "") -> Any:
        """Chat then parse JSON out of the reply, tolerant of fences/prose."""
        raw = self.chat(messages, max_tokens=max_tokens, temperature=0.0, tag=tag)
        return extract_json(raw)

    # -- remote ---------------------------------------------------------------
    def _chat_remote(self, messages, max_tokens, temperature) -> str:
    # Anthropic requires system prompt separated from messages array
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        payload = {
            "model": self.cfg.llm_model,
            "max_tokens": max_tokens or self.cfg.llm_max_tokens,
            "temperature": self.cfg.llm_temperature if temperature is None else temperature,
            "messages": filtered,
        }
        if system:
            payload["system"] = system

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.cfg.llm_base_url}/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as resp:
            data = json.loads(resp.read().decode())
        usage = data.get("usage") or {}
        self.total_tokens += int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        content = data["content"][0]["text"]
        if not content:
            raise LLMError("empty content from model")
        return content.strip()

    # -- stub -----------------------------------------------------------------
    def _chat_stub(self, messages) -> str:
        """Deterministic offline fallback. Returns minimal valid JSON when a JSON
        schema is requested, else a short canned string."""
        prompt = "\n".join(m["content"] for m in messages)
        low = prompt.lower()
        self.total_tokens += 200
        if "json" in low and "dag" in low:
            return json.dumps(_stub_dag(prompt))
        if "json" in low and "brief" in low:
            return json.dumps({
                "summary": "stub research brief",
                "key_findings": ["offline stub: no external retrieval"],
                "refined_intent": _first_line(prompt),
                "materiality": "low",
            })
        if "json" in low:
            return json.dumps({"note": "stub", "value": _first_line(prompt)})
        return "stub response (LLM server unreachable)"


# ---------------------------------------------------------------------------
def _first_line(text: str) -> str:
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            return ln[:160]
    return ""


def _stub_dag(prompt: str) -> Dict[str, Any]:
    return {
        "topology": "fan-out",
        "depth": 2,
        "why_topology": "stub: parallel retrieval assumed",
        "why_depth": "stub default depth",
        "nodes": [
            {"node_id": "n1", "intent": "retrieve relevant information",
             "dependencies": [], "structural": "fan-out", "functional": "retriever",
             "epistemic": "specialist"},
            {"node_id": "n2", "intent": "synthesize answer",
             "dependencies": ["n1"], "structural": "join", "functional": "synthesizer",
             "epistemic": "generalist"},
        ],
    }


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(raw: str) -> Any:
    """Pull the first JSON object/array out of a model reply."""
    if not raw:
        raise LLMError("empty reply")
    m = _JSON_FENCE.search(raw)
    candidate = m.group(1) if m else raw
    # try direct
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # find first balanced {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            c = candidate[i]
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"could not extract JSON from reply: {raw[:200]!r}")
