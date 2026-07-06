"""v3.1 — Retrieval grounding seam.

Retriever-role nodes call retrieve() instead of hallucinating plausible facts.
Today this is backed by Claude's server-side web search tool. The seam is
deliberately backend-agnostic: the planner, kernel, and EF machinery only ever
see retrieve(GroundedResult) — so the backend can later swap to a NANDA-
discovered retrieval agent (grounding_backend="nanda") without touching any of
them. That's the whole point of isolating grounding here.

Web-search version note: web_search_20260209 (dynamic filtering) requires
Opus 4.8/4.7/4.6 or Sonnet 5/4.6. This project runs Haiku 4.5, which uses the
basic web_search_20250305 (configurable via cfg.web_search_tool_version).

Server tool mechanics (per the Claude API):
  - declared in tools=[]; Claude runs the search server-side, no client loop
  - success: a web_search_tool_result block whose .content is a LIST of results
  - error:   the same block type whose .content is a single error OBJECT
  - text blocks carry a .citations array
  - a long server-tool loop can end with stop_reason == "pause_turn" — re-send
    to resume (no extra "continue" user message)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class GroundedResult:
    text: str
    citations: List[Dict] = field(default_factory=list)   # {url, title, cited_text}
    sources: List[str] = field(default_factory=list)       # deduped source URLs
    used_search: bool = False
    tokens: int = 0
    error: Optional[str] = None


_RETRIEVE_SYSTEM = (
    "You are a retrieval agent. Search the web for authoritative, current "
    "information on the requested topic and report concrete findings grounded "
    "in the sources you find — cite specifics (figures, named systems, dates). "
    "Do not speculate beyond what the sources support."
)


def retrieve(query: str, cfg, client_provider: Callable, *,
             max_continuations: int = 3) -> GroundedResult:
    """The grounding seam. Backend selected by cfg.grounding_backend:
      - "web_search": Claude server-side web search (default)
      - "none":       no grounding (caller falls back to plain generation)
      - future "nanda": dispatch to a discovered retrieval agent (client_provider
        would be replaced by a NANDA directory client; signature unchanged)
    """
    if not getattr(cfg, "grounding_enabled", True):
        return GroundedResult(text="", used_search=False)
    backend = getattr(cfg, "grounding_backend", "web_search")
    if backend == "none":
        return GroundedResult(text="", used_search=False)
    if backend == "web_search":
        return _web_search(query, cfg, client_provider, max_continuations)
    return GroundedResult(text="", used_search=False,
                          error=f"unknown grounding backend: {backend}")


def _web_search(query, cfg, client_provider, max_continuations) -> GroundedResult:
    # No key -> skip cleanly (keeps offline/stub runs from making a doomed call).
    if not getattr(cfg, "anthropic_api_key", ""):
        return GroundedResult(text="", used_search=False, error="no api key")
    try:
        client = client_provider()
    except Exception as e:
        return GroundedResult(text="", used_search=False, error=f"client init failed: {e}")

    tool = {
        "type": getattr(cfg, "web_search_tool_version", "web_search_20250305"),
        "name": "web_search",
        "max_uses": getattr(cfg, "web_search_max_uses", 5),
    }
    messages = [{"role": "user", "content": query}]
    text_parts: List[str] = []
    citations: List[Dict] = []
    sources: List[str] = []
    tokens = 0
    used = False

    try:
        for _ in range(max_continuations + 1):
            resp = client.messages.create(
                model=cfg.llm_model,
                max_tokens=cfg.llm_max_tokens,
                temperature=getattr(cfg, "llm_temperature", 0.0),
                system=_RETRIEVE_SYSTEM,
                tools=[tool],
                messages=messages,
            )
            usage = getattr(resp, "usage", None)
            if usage:
                tokens += getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
            for block in resp.content:
                bt = getattr(block, "type", None)
                if bt == "text":
                    text_parts.append(block.text)
                    for c in (getattr(block, "citations", None) or []):
                        citations.append({
                            "url": getattr(c, "url", ""),
                            "title": getattr(c, "title", ""),
                            "cited_text": getattr(c, "cited_text", ""),
                        })
                elif bt == "web_search_tool_result":
                    used = True
                    content = getattr(block, "content", None)
                    # success -> list of web_search_result; error -> single object
                    if isinstance(content, list):
                        for r in content:
                            u = getattr(r, "url", None)
                            if u:
                                sources.append(u)
            if getattr(resp, "stop_reason", None) == "pause_turn":
                messages = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": resp.content},
                ]
                continue
            break
    except Exception as e:
        return GroundedResult(
            text="\n".join(text_parts).strip(), citations=citations,
            sources=_dedup(sources), used_search=used, tokens=tokens, error=str(e),
        )

    return GroundedResult(
        text="\n".join(text_parts).strip(), citations=citations,
        sources=_dedup(sources), used_search=used, tokens=tokens,
    )


def _dedup(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out
