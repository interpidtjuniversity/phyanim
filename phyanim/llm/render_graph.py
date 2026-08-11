from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from phyanim.llm.chat_model import make_chat_model
from phyanim.llm.client import LLMConfig
from phyanim.llm.graph_nodes import (
    finalize_failure,
    finalize_success,
    make_analyze,
    make_analyze_error,
    make_generate_code,
    make_render,
    validate_code,
)
from phyanim.llm.graph_state import RenderGraphState
from phyanim.llm.knowledge_base import KB_TOOLS

logger = logging.getLogger(__name__)


def build_render_workflow(
    llm_config: LLMConfig,
    tts_config: dict,
    root_dir: str,
    code_dir: str,
    media_dir: str,
    max_retries: int = 2,
) -> Any:
    """Build and compile the LangGraph render workflow.

    Parameters
    ----------
    llm_config:
        LLM connection configuration (api_key, base_url, model, etc.).
    tts_config:
        TTS configuration dict injected into generated code.
    media_dir:
        Root media directory for Manim output.
    output_dir:
        Directory for generated .py scripts.
    max_retries:
        Maximum number of repair attempts on render failure.

    Returns
    -------
    Compiled LangGraph runnable.
    """
    model = make_chat_model(llm_config)

    graph = StateGraph(RenderGraphState)

    def make_appending_tool_node():
        tool_node = ToolNode(KB_TOOLS)
        def run_tools(state: RenderGraphState):
            output = tool_node.invoke(state)
            return {
                "messages": [
                    *state.get("messages", []),
                    *output.get("messages", []),
                ]
            }
        return run_tools


    # --- Add nodes ---
    graph.add_node("analyze", make_analyze(model))

    graph.add_node("generate_code", make_generate_code(model, tts_config, media_dir))
    graph.add_node("generate_code_tools", make_appending_tool_node())

    graph.add_node("validate_code", validate_code)

    graph.add_node("render", make_render(root_dir, code_dir))

    graph.add_node("analyze_error", make_analyze_error(model))
    graph.add_node("analyze_error_tools", make_appending_tool_node())

    graph.add_node("finalize_success", finalize_success)
    graph.add_node("finalize_failure", finalize_failure)

    graph.add_node("clear_code_generate_messages", lambda s: {"messages": []})
    graph.add_node("clear_analyze_error_messages", lambda s: {"messages": []})

    # --- Set entry point ---
    graph.set_entry_point("analyze")

    # --- Edges ---

    # analyze: tool_calls → analyze_tools, else → generate_code
    graph.add_edge("analyze", "generate_code")

    # generate_code: tool_calls → generate_code_tools, else → validate_code
    graph.add_conditional_edges(
        "generate_code",
        tools_condition,
        {"tools": "generate_code_tools", END: "clear_code_generate_messages"},
    )
    graph.add_edge("generate_code_tools", "generate_code")

    # 清空上下文
    graph.add_edge("clear_code_generate_messages", "validate_code")

    graph.add_conditional_edges(
        "validate_code",
        lambda s: "render" if not s.get("validate_error") else "generate_code",
        {"render": "render", "generate_code": "generate_code"},
    )

    # render → finalize_success (ok) or analyze_error (failed)
    graph.add_conditional_edges(
        "render",
        lambda s: "finalize_success" if s.get("returncode", 1) == 0 else "analyze_error",
        {
            "finalize_success": "finalize_success",
            "analyze_error": "analyze_error",
        },
    )

    # analyze_error: tool_calls → analyze_error_tools, else → retry router
    def _retry_router(state: RenderGraphState) -> str:
        count = state.get("retry_count", 0)
        if count > max_retries:
            return "finalize_failure"
        return "generate_code"

    graph.add_conditional_edges(
        "analyze_error",
        tools_condition,
        {"tools": "analyze_error_tools", END: "clear_analyze_error_messages"},
    )
    graph.add_edge("analyze_error_tools", "analyze_error")

    # 清空上下文
    graph.add_edge("clear_analyze_error_messages", "retry_router")

    # Retry router node (no-op, just for routing decision)
    graph.add_node("retry_router", lambda s: {"enter_generate_code_source": "retry_router"})
    graph.add_conditional_edges(
        "retry_router",
        _retry_router,
        {"generate_code": "generate_code", "finalize_failure": "finalize_failure"},
    )

    # --- Terminal nodes ---
    graph.add_edge("finalize_success", END)
    graph.add_edge("finalize_failure", END)

    return graph.compile()
