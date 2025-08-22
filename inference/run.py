"""Hàm public để invoke agent graph từ API hoặc script."""
from __future__ import annotations
from agent.graph import build_graph, GeminiAgentGraph  # type: ignore
from state.state import init_state
from model.schemas import ToolResult
from typing import Generator, AsyncGenerator

try:  # Gemini message class
    from langchain_core.messages import HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    HumanMessage = None  # type: ignore


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


def stream_agent(query: str):
    """Streaming phiên bản đơn giản: chạy router + tool qua full graph (để lấy tool result),
    sau đó stream phần finalize Gemini token-by-token (nếu có), fallback chia nhỏ từ của output.

    Yield chuỗi (token) đã sẵn sàng gửi tới client (không bao gồm 'data:' prefix)."""
    # Dùng lệnh invoke bình thường để có base output/tool selection
    result = invoke_agent(query)
    base_output = result.get("output") or ""

    # Chuẩn bị Gemini model cho phần cải thiện câu trả lời
    try:
        g = GeminiAgentGraph()
        llm = getattr(g, "gemini_llm", None)
    except Exception:  # pragma: no cover
        llm = None

    if llm is not None and HumanMessage is not None:
        prompt = (
            "Bạn là trợ lý. Người dùng hỏi: '" + query + "'.\n"
            "Kết quả trung gian/tool: '" + base_output + "'.\n"
            "Hãy trả lời ngắn gọn và rõ ràng bằng tiếng Việt."
        )
        try:
            last = ""
            for chunk in llm.stream([HumanMessage(content=prompt)]):  # type: ignore[attr-defined]
                full = getattr(chunk, "content", None) if chunk is not None else None
                if not full:
                    continue
                # Nhiều implementation trả toàn bộ nội dung tích luỹ -> lấy delta
                if full.startswith(last):
                    delta = full[len(last):]
                else:
                    # fallback nếu không phải chuỗi tích luỹ
                    delta = full
                last = full
                for ch in delta:
                    yield ch
            return
        except Exception:  # pragma: no cover
            pass

    # Fallback: stream từng ký tự của base_output
    for ch in base_output:
        yield ch
