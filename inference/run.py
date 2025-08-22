"""Hàm public để invoke agent graph từ API hoặc script."""
from __future__ import annotations
from agent.graph import build_graph
from state.state import init_state
from model.schemas import ToolResult


def invoke_agent(query: str):
    graph = build_graph()
    state = init_state(query)
    # LangGraph runner nhận dict state, trả (yield) events hoặc kết quả.
    # Sử dụng .invoke để đơn giản.
    new_state = graph.invoke(state)
    tool = new_state.get("selected_tool")
    tool_res_list = []
    if tool:
        tool_res_list.append(
            ToolResult(tool_name=tool, output=new_state.get("output", ""), raw_args=new_state.get("tool_args", {}).get(tool) if new_state.get("tool_args") else None).model_dump()
        )
    return {
        "output": new_state.get("output"),
        "trace": new_state.get("trace", []),
        "intermediate": new_state.get("intermediate", []),
        "selected_tool": new_state.get("selected_tool"),
        "tool_results": tool_res_list or None,
    }
