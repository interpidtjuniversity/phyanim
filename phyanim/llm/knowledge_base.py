"""Manim knowledge base loader and retriever for RAG tool calling.

Defines the ``search_knowledge_base`` tool using LangChain's ``@tool``
decorator so it can be bound to any LangChain ChatModel and executed
by LangGraph's ``ToolNode``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_KB_DOCX_PATH = Path(__file__).resolve().parent / "rag" / "Manim_Knowledge_Base.docx"

# Loaded lazily on first access.
_KB_TEXT: str | None = None
_KB_CHUNKS: list[str] | None = None


def _load_kb() -> None:
    """Load and chunk the knowledge base from the .docx file."""
    global _KB_TEXT, _KB_CHUNKS
    if _KB_TEXT is not None:
        return

    if not _KB_DOCX_PATH.exists():
        logger.warning("Knowledge base file not found: %s", _KB_DOCX_PATH)
        _KB_TEXT = ""
        _KB_CHUNKS = []
        return

    try:
        from docx import Document

        doc = Document(str(_KB_DOCX_PATH))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        _KB_TEXT = "\n".join(paragraphs)

        # Chunk by paragraphs, grouping every ~10 paragraphs into one chunk.
        _KB_CHUNKS = []
        chunk_size = 10
        for i in range(0, len(paragraphs), chunk_size):
            chunk = "\n".join(paragraphs[i : i + chunk_size])
            if chunk:
                _KB_CHUNKS.append(chunk)

        logger.info(
            "Loaded knowledge base: %d chars, %d chunks from %s",
            len(_KB_TEXT),
            len(_KB_CHUNKS),
            _KB_DOCX_PATH,
        )
    except Exception as exc:
        logger.error("Failed to load knowledge base: %s", exc)
        _KB_TEXT = ""
        _KB_CHUNKS = []


def _do_search(query: str, top_k: int = 5) -> str:
    """Search the Manim knowledge base for content matching *query*."""
    _load_kb()
    if not _KB_CHUNKS:
        return "知识库为空或未加载。"

    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored: list[tuple[float, str]] = []
    for chunk in _KB_CHUNKS:
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split())
        overlap = len(query_words & chunk_words)
        for word in query_words:
            if len(word) > 2 and word in chunk_lower:
                overlap += 0.5
        if overlap > 0:
            scored.append((float(overlap), chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return f"未找到与 '{query}' 相关的知识库内容。"

    results = []
    for i, (_, chunk) in enumerate(scored[:top_k], 1):
        results.append(f"--- 知识库片段 {i} ---\n{chunk}")
    return "\n\n".join(results)


#@tool
def search_knowledge_base(query: str) -> str:
    """搜索 Manim 知识库，获取 Manim API 用法、常见错误解决方案、动画模式等相关知识。

    当需要查阅 Manim 具体用法、排查错误、或了解最佳实践时调用此工具。

    Args:
        query: 搜索关键词或问题描述，如 'TracedPath 用法' 或 'wait(0) 报错'
    """
    return _do_search(query)


# Export the tool for binding to LLM models.
KB_TOOLS = [search_knowledge_base]


print(search_knowledge_base("Circle"))