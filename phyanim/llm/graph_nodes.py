from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ChatMessage
from langchain_core.runnables import history
from langchain_openai import ChatOpenAI

from phyanim.llm.knowledge_base import KB_TOOLS
from phyanim.llm.planner import (
    ANALYSIS_SYSTEM_PROMPT,
    CodeValidationError,
    _extract_code,
    _inject_force_exit_hook,
    _inject_headless_config,
    _inject_media_dir,
    _inject_tts_config,
    _rename_scene_class,
    _validate_python,
)
from phyanim.llm.prompt_builder import build_code_generate_system_prompt
from phyanim.llm.runner import start_render_process
from phyanim.llm.graph_state import RenderGraphState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node 1: Analyze problem and generate animation plan (no code)
# ---------------------------------------------------------------------------

def make_analyze(model: ChatOpenAI):

    def analyze(state: RenderGraphState) -> dict[str, Any]:
        prompt = state["prompt"]
        history_messages = state.get("history_messages", [])

        scene_name = state.get("scene_name", "")
        logger.info("[%s] Generating animation plan", scene_name)

        # 这个msg当前是一次性的，不会进入后续节点上下文
        msgs = [
            *history_messages,
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = model.invoke(msgs)

        analysis_info = response.content
        logger.info(
            "[%s] Animation plan generated (%d chars)",
            state.get("scene_name", ""),
                len(analysis_info),
        )
        result: dict[str, Any] = {}
        result["analysis_info"] = analysis_info
        result["enter_generate_code_source"] = "analyze"
        return result

    return analyze


# ---------------------------------------------------------------------------
# Node 2: Generate code via LLM
# ---------------------------------------------------------------------------

def make_generate_code(model: ChatOpenAI, tts_config: dict, media_dir: str):
    """Return a node function that generates code.

    The model has ``search_knowledge_base`` bound as a tool.
    """

    model_with_tools = model.bind_tools(KB_TOOLS)

    def generate_code(state: RenderGraphState) -> dict[str, Any]:
        prompt = state["prompt"]
        history_messages = state.get("history_messages", [])
        analysis_info = state.get("analysis_info", "")
        scene_name = state.get("scene_name")
        messages = state.get("messages", [])
        enter_generate_code_source = state.get("enter_generate_code_source", "")

        if messages:
            response = model_with_tools.invoke(messages)
            new_messages = messages + [response]
        else:
            system_prompt = build_code_generate_system_prompt()
            msgs = [
                *history_messages,
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"用户原始输入为：{prompt}"),
                HumanMessage(content=f"具体细节：{analysis_info}"),
            ]
            if enter_generate_code_source == "analyze":
                pass
            elif enter_generate_code_source == "validate_code":
                source_code = state.get("source_code", "")
                validate_error = state.get("validate_error", "")
                msgs.append(HumanMessage(content=f"你上次生成的代码为：{source_code} \n 它不符合python语法：{validate_error}"))
            elif enter_generate_code_source == "retry_router":
                source_code = state.get("source_code", "")
                render_error = state.get("render_error", "")
                render_error_analysis = state.get("render_error_analysis", "")
                msgs.append(HumanMessage(content=f"你上次生成的代码为：{source_code} \n manim v0.18.1 渲染失败：{render_error}"))
                msgs.append(HumanMessage(content=f"修改建议为：{render_error_analysis}"))

            response = model_with_tools.invoke(msgs)
            new_messages = msgs + [response]

        result: dict[str, Any] = {"messages": new_messages}
        if isinstance(response, AIMessage) and not response.tool_calls:
            raw = response.content
            source_code = _extract_code(raw)

            code = _inject_tts_config(source_code, tts_config)
            if media_dir:
                code = _inject_media_dir(code, media_dir)
            code = _inject_headless_config(code)
            code = _inject_force_exit_hook(code)
            if scene_name:
                code = _rename_scene_class(code, scene_name)

            result["source_code"] = source_code
            result["code"] = code
            #result["messages"] = []

        return result

    return generate_code


# ---------------------------------------------------------------------------
# Node 3: Validate generated code
# ---------------------------------------------------------------------------

def validate_code(state: RenderGraphState) -> dict[str, Any]:
    """Check that generated code parses as valid Python."""
    source_code = state.get("source_code", "")
    try:
        _validate_python(source_code)
    except CodeValidationError as exc:
        return {"validate_error": str(exc), "enter_generate_code_source": "validate_code"}
    return {
        "validate_error": "",
        "enter_generate_code_source": "",
    }


# ---------------------------------------------------------------------------
# Node 4: Execute render subprocess
# ---------------------------------------------------------------------------

def make_render(root_dir: str, code_dir: str):
    """Return a node function that runs the Manim render subprocess."""

    def render(state: RenderGraphState) -> dict[str, Any]:
        code = state["code"]
        scene_name = state.get("scene_name", "generated_animation")

        script_path, process = start_render_process(
            code,
            root_dir=root_dir,
            script_name=f"{scene_name}.py",
            code_dir=code_dir,
        )
        logger.info("[%s] Rendering with PID %s", scene_name, process.pid)
        stdout, stderr = process.communicate()
        returncode = process.returncode

        if returncode == 0:
            return {
                "returncode": returncode,
                "render_error": "",
                # 视频路径被封装在manager里无状态的node无法直接调用，这里返回ok表示渲染成功
                "video_path": "ok",
            }

        return {
            "returncode": returncode,
            "render_error": (stderr or stdout or "").strip(),
            "video_path": "not_ok",
        }

    return render


# ---------------------------------------------------------------------------
# Node 5: Analyze render error via LLM (with knowledge base tool)
# ---------------------------------------------------------------------------

def make_analyze_error(model: ChatOpenAI):
    """Return a node function that uses the LLM to analyze render errors.

    The model has ``search_knowledge_base`` bound as a tool.
    """

    model_with_tools = model.bind_tools(KB_TOOLS)

    def analyze_error(state: RenderGraphState) -> dict[str, Any]:
        messages = state.get("messages", [])

        if messages:
            response = model_with_tools.invoke(messages)
            new_messages = messages + [response]
        else:
            prompt = state.get("prompt", "")
            analysis_info = state.get("analysis_info", "")
            source_code = state.get("source_code", "")
            render_error = state.get("render_error", "")
            analysis_prompt = (
                "你是 Manim 渲染错误分析专家。Manim 基于v0.18.1版本，分析以下渲染错误，"
                "指出根本原因和修复建议（简要回答）：\n"
                f"错误信息:\n{render_error}\n"
                "你可以调用知识库搜索工具查阅相关 Manim 文档。\n"
                "请输出：\n"
                "1. 错误类型（如：类型错误、属性错误、导入错误、方法错误等）\n"
                "2. 根本原因\n"
                "3. 修复建议（具体的代码修改方向）"
            )
            msgs = [
                SystemMessage(content=analysis_prompt),
                HumanMessage(content=f"用户原始输入为：{prompt}"),
                HumanMessage(content=f"具体细节：{analysis_info}"),
                HumanMessage(content=f"报错代码为：{source_code}"),
            ]
            response = model_with_tools.invoke(msgs)
            new_messages = msgs + [response]

        result: dict[str, Any] = {
            "messages": new_messages,
        }

        if isinstance(response, AIMessage) and not response.tool_calls:
            retry_count = state.get("retry_count", 0)
            analysis = response.content
            logger.info(
                "[Error Analysis] retry %d: %s", retry_count, analysis[:2000]
            )
            result["render_error_analysis"] = analysis
            result["retry_count"] = retry_count + 1
            #result["messages"] = []

        return result

    return analyze_error


# ---------------------------------------------------------------------------
# Node 6: Finalize (success or failure)
# ---------------------------------------------------------------------------

def finalize_success(state: RenderGraphState) -> dict[str, Any]:
    return {"video_path": state.get("video_path", "")}


def finalize_failure(state: RenderGraphState) -> dict[str, Any]:
    return {
        "video_path": "not_ok",
        "render_error": state.get("render_error", ""),
        "render_error_analysis": state.get(
            "render_error_analysis", ""
        ),
    }
