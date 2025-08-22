"""Định nghĩa cấu trúc state dùng trong LangGraph.

Sử dụng TypedDict (hoặc có thể chuyển sang pydantic BaseModel nếu muốn validation runtime).
"""
from __future__ import annotations
from typing import TypedDict, List, Any, Optional


class AgentState(TypedDict, total=False):
    # Đầu vào ban đầu từ người dùng
    query: str
    # Tool được router chọn
    selected_tool: Optional[str]
    # Kết quả tạm thời / trung gian
    intermediate: List[str]
    # Đầu ra cuối
    output: Optional[str]
    # Trace đơn giản để debug
    trace: List[str]
    # Lưu arg cho tool_call LLM (map tool_name -> args dict)
    tool_args: dict


def init_state(query: str) -> AgentState:
    return {
        "query": query,
        "selected_tool": None,
        "intermediate": [],
        "output": None,
        "trace": [f"INIT: {query}"]
    }
