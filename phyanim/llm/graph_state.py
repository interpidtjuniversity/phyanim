"""LangGraph state definition for the code-generation → render → retry pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class RenderGraphState(TypedDict, total=False):

    # 用户请求
    prompt: str
    # 历史消息包括analysis和code
    history_messages: list[Any]
    # 图片url
    image_urls: list[str]
    # 场景名称
    scene_name: str
    # 分析信息
    analysis_info: str
    # 源代码
    source_code: str
    # 注入后的代码
    code: str
    # 验证错误，这个不需要再开一个llm节点单独分析
    validate_error: str
    # 渲染错误
    render_error: str
    # 渲染错误分析
    render_error_analysis: str
    # 重试次数
    retry_count: int
    # 视频路径
    video_path: str
    # render返回值
    returncode: int
    # 节点消息
    messages: list[BaseMessage]
    # 进入生成代码节点的来源
    enter_generate_code_source: str
