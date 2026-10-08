"""Knowledge router: an agent that delegates each question to specialist
sub-agents (agent-as-tool), each answering from its own LightRAG silo.

The turn's citations must keep resolving as cite://N -> chunks[N-1] of the
merged LightRAG payload, so every specialist's local [n](cite://n) markers are
renumbered onto one turn-wide counter (see IACTTool.claim_citations).
Design: docs/superpowers/specs/2026-10-07-knowledge-router-design.md
"""
import asyncio
import re
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.streaming_utils import merge_lightrag_graph
from utils.logger import get_logger

logger = get_logger(__name__)

KNOWLEDGE_ROUTER_TOOL_NAME = "consultar_varios"

# Tool calls each specialist may make per question. A general question runs every
# specialist, each looping retrieval + coverage + retries on a slow local LLM:
# without a ceiling a single answer took minutes and timed out (observed 9-11 calls
# for the EN specialist on hard questions). 6 leaves room for load_skill + two
# coverage listings + 2-3 searches (enough to conclude "it does not exist"); 3 left
# ~2 useful searches. Starting value: re-measure latency vs recall before changing.
SPECIALIST_MAX_TOOL_CALLS = 6

KNOWLEDGE_ROUTER_INSTRUCTION = """<knowledge_router>
You answer by delegating to specialist agents; you have no knowledge base of your own.
Each specialist tool's description says which knowledge it covers.
- The language the question is written in says NOTHING about which specialist to use: the knowledge may be split by language while every question arrives in the same one.
- Use a single specialist only when the question explicitly restricts itself to that specialist's knowledge (it names its domain, language, product line...). Call that specialist's tool once with the user's question; its answer is shown to the user as-is.
- Otherwise (a general question, nothing restricts it, or it spans several specialists), call `consultar_varios` ONCE with the names of every relevant specialist (include all of them when unsure), then write the final answer from their replies.
- Your final answer must include the findings of EVERY specialist that answered, in one section per specialist (named after its knowledge area). Never drop a specialist's documents or conclusions because another specialist reached a similar one: if they agree, say so and still list each one's documents.
- Never call two specialist tools one after another for the same question; use `consultar_varios` instead.
- Keep every [N](cite://N) citation exactly as the specialists wrote it, next to the claim it supports. Never invent, renumber or drop citations.
</knowledge_router>"""

# [n](cite://n), also with the full-width brackets some models emit (【n】).
_CITE_RE = re.compile(r"[\[【](\d+)[\]】]\(cite://(\d+)\)")


def rebase_citations(text: str, delta: int, valid: Optional[int] = None) -> str:
    """Shift every cite marker by *delta*; the cite:// number is authoritative.

    *valid*: how many chunks back this text's own numbering (1..valid). Markers
    outside it are dropped: nothing backs them (e.g. copied from a nested agent
    whose chunks never reach the payload), and after shifting they would open
    another specialist's chunk instead.
    """
    def _shift(m: re.Match) -> str:
        n = int(m.group(2))
        if valid is not None and not 1 <= n <= valid:
            return ""
        return f"[{n + delta}](cite://{n + delta})"

    return _CITE_RE.sub(_shift, text)


def tag_graph_with_silo(graph: dict, silo_id: Optional[int], silo_name: Optional[str]) -> dict:
    """Stamp silo_id/silo_name on every entity, relationship and chunk (new dict)."""
    data = graph.get("data") or {}
    tag = {"silo_id": silo_id, "silo_name": silo_name}
    return {
        **graph,
        "data": {
            **data,
            **{k: [{**item, **tag} for item in data.get(k) or []] for k in ("entities", "relationships", "chunks")},
        },
    }


def graph_artifact(graph: Optional[dict]) -> Optional[List[Document]]:
    """Wrap a payload like a direct LightRAG tool's artifact, so the streaming
    accumulator and history reconstruction pick it up unchanged."""
    if not graph:
        return None
    return [Document(page_content="", metadata={"lightrag_raw_data": graph})]


class _ConsultarVariosInput(BaseModel):
    especialistas: List[str] = Field(description="Names of the specialist tools to ask")
    pregunta: str = Field(description="The user's question, as asked")


def create_consultar_varios_tool(specialists: List[Any], lock: asyncio.Lock) -> StructuredTool:
    """Ask several specialists the same question in parallel; merge in list order.

    *specialists* are router-mode IACTTools. Their sub-agents run concurrently,
    but citations are claimed afterwards in the order requested, so numbering
    never depends on which one finished first.
    """
    by_name = {t.name.lower(): t for t in specialists}
    valid_names = ", ".join(t.name for t in specialists)
    catalog = "\n".join(f"- {t.name}: {t.description}" for t in specialists)

    async def _consultar(especialistas: List[str], pregunta: str) -> Tuple[str, Optional[List[Document]]]:
        requested = list(dict.fromkeys(n.lower() for n in especialistas))
        chosen = [by_name[n] for n in requested if n in by_name]
        unknown = [n for n in dict.fromkeys(especialistas) if n.lower() not in by_name]
        if not chosen:
            return f"Unknown specialists {especialistas}. Valid names: {valid_names}", None
        async with lock:
            runs = await asyncio.gather(*(t.run_subagent(pregunta) for t in chosen), return_exceptions=True)
            parts: List[str] = []
            merged: Optional[dict] = None
            for t, run in zip(chosen, runs):
                if isinstance(run, BaseException):
                    logger.error("Knowledge router specialist %s failed: %s", t.name, run)
                    parts.append(f"### {t.name}\n(error: {run})")
                    continue
                text, graph = t.claim_citations(*run)
                parts.append(f"### {t.name}\n{text}")
                if graph:
                    merged = merge_lightrag_graph(merged, graph)
        if unknown:
            parts.append(f"(ignored unknown specialists: {', '.join(unknown)}. Valid: {valid_names})")
        return "\n\n".join(parts), graph_artifact(merged)

    return StructuredTool.from_function(
        coroutine=_consultar,
        name=KNOWLEDGE_ROUTER_TOOL_NAME,
        description=(
            "Ask several specialists the same question at once and get all their answers.\n"
            "Specialists:\n" + catalog
        ),
        args_schema=_ConsultarVariosInput,
        response_format="content_and_artifact",
    )
