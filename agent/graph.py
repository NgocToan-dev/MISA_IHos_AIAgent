"""Xây dựng LangGraph: định nghĩa nodes và compile graph.

Kiến trúc (đơn giản hoá để demo):
    (input state) -> router_node -> tool_node -> finalize_node -> END

Refactor thành lớp `GeminiAgentGraph` để:
    - Dễ mở rộng (thêm node, subgraph, memory, checkpoint)
    - Dễ cấu hình (model, gemini model name, temperature)
    - Dễ test (có thể khởi tạo nhiều instance độc lập)

Public API giữ nguyên hàm module-level `build_graph()` để không phá vỡ mã hiện tại.
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END
from typing import Any
import os

# Tích hợp Gemini qua langchain-google-genai
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.messages import HumanMessage
except ImportError:  # Chưa cài gói gemini
    ChatGoogleGenerativeAI = None  # type: ignore
    HumanMessage = None  # type: ignore
from prompt.system_prompt import IHOS_SYSTEM_PROMPT
from state.state import AgentState
from tools.registry import ALL_TOOLS  # @tool based
import datetime


class GeminiAgentGraph:
    """Đóng gói logic xây dựng và vận hành LangGraph agent (Cách 1: dùng system_instruction)."""

    def __init__(
        self,
        gemini_model_name: str | None = None,
        temperature: float = 0.0,
        enable_gemini: bool = True,
        tools: list | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.gemini_model_name = gemini_model_name or os.getenv(
            "GEMINI_MODEL", "gemini-2.5-flash"
        )
        self.temperature = temperature
        self.compiled_graph = None
        self.gemini_llm = None
        self.tools = tools if tools is not None else ALL_TOOLS

        current_date = "Hôm nay là ngày " + str(datetime.datetime.today())
        default_prompt = (
            "Bạn là trợ lý AI của nền tảng IHOS của công ty cổ phần MISA. "
            "Nhiệm vụ: hiểu câu hỏi tiếng Việt, chọn đúng tool (ví dụ: ihos_doc_search khi cần tra cứu tài liệu), "
            "trả lời súc tích, chính xác, giữ nguyên đơn vị đo, và cảnh báo khi thiếu dữ liệu. "
            "Trả ra kết quả dạng markdown trực quan để người dùng nhìn."
        )
        self.system_prompt = system_prompt or default_prompt + f"\n{current_date}"

        if (
            enable_gemini
            and ChatGoogleGenerativeAI is not None
            and os.getenv("GOOGLE_API_KEY")
        ):
            try:  # pragma: no cover
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model=self.gemini_model_name,
                    temperature=temperature,
                    system_instruction=self.system_prompt,
                )
            except Exception:
                self.gemini_llm = None

    # ---------------- Nodes ---------------- #
    def router_node(self, state: AgentState) -> AgentState:
        """Dùng Gemini tool calling để chọn tool; fallback echo."""
        query = state.get("query", "")
        trace = state.setdefault("trace", [])
        if self.gemini_llm is not None and HumanMessage is not None:
            try:
                llm_tools = self.gemini_llm.bind_tools(self.tools)
                msgs = [HumanMessage(content="Câu hỏi: " + query)]
                ai_msg = llm_tools.invoke(msgs)
                tool_calls = getattr(ai_msg, "tool_calls", []) or []
                if tool_calls:
                    first = tool_calls[0]
                    tool_name = first.get("name")
                    args = first.get("args", {})
                    state["selected_tool"] = tool_name
                    tool_args = state.get("tool_args") or {}
                    tool_args[tool_name] = args
                    state["tool_args"] = tool_args
                    trace.append(f"LLM_TOOL_SELECT -> {tool_name} {args}")
                    return state
                state["selected_tool"] = "echo"
                trace.append("LLM_TOOL_SELECT_NONE -> echo")
                return state
            except Exception as e:  # pragma: no cover
                trace.append(f"LLM_TOOL_ROUTE_ERROR: {e}")
        state["selected_tool"] = "echo"
        trace.append("ROUTER_DEFAULT -> echo")
        return state

    def tool_node(self, state: AgentState) -> AgentState:
        tool_name = state.get("selected_tool")
        query = state.get("query", "")
        # Map tool name -> tool object (dùng getattr fallback để tránh lỗi static typing)
        tool_map: dict[str, Any] = {
            getattr(t, "name", getattr(t, "__name__", f"tool_{i}")): t
            for i, t in enumerate(self.tools)
        }
        if tool_name in tool_map:
            args_map = state.get("tool_args", {}).get(tool_name, {}) or {}
            # Auto-fill: nếu tool chưa có args và chỉ có 1 param -> dùng query
            if not args_map:
                try:
                    schema = getattr(tool_map[tool_name], "args", None)
                    if schema and len(schema) == 1:
                        sole = list(schema.keys())[0]
                        args_map = {sole: query}
                except Exception:  # pragma: no cover
                    pass
            if not args_map:
                # fallback theo tên phổ biến
                name_map = {"echo": "text", "internet_search": "query"}
                param = name_map.get(tool_name or "")
                if param:
                    args_map = {param: query}
            # persist
            if args_map:
                tool_args_all = state.get("tool_args") or {}
                tool_args_all[tool_name] = args_map
                state["tool_args"] = tool_args_all
            tool_obj = tool_map[tool_name]
            try:
                # LangChain tool object có .invoke; nếu không thì gọi như hàm thường
                if hasattr(tool_obj, "invoke"):
                    result = tool_obj.invoke(args_map)  # type: ignore[attr-defined]
                else:
                    result = tool_obj(**args_map)  # type: ignore[misc]
            except Exception as e:  # pragma: no cover
                result = f"ToolError: {e}"
        else:
            result = f"ECHO: {query}" if tool_name == "echo" else f"Unknown tool: {tool_name}"
        interm = state.setdefault("intermediate", [])
        interm.append(result)
        trace = state.setdefault("trace", [])
        trace.append(f"TOOL {tool_name}: {result}")
        state["output"] = result
        return state

    def finalize_node(self, state: AgentState) -> AgentState:
        base_output = state.get("output") or "(no output)"
        query = state.get("query", "")
        enhanced = base_output
        if self.gemini_llm is not None and HumanMessage is not None:
            try:
                prompt = (
                    "Người dùng hỏi: "
                    + query
                    + "\nKết quả trung gian/tool: "
                    + base_output
                    + "\nHãy tổng hợp và trả lời ngắn gọn bằng tiếng Việt, giữ nguyên thuật ngữ chuyên môn khi cần."
                )
                msg = self.gemini_llm.invoke([HumanMessage(content=prompt)])
                enhanced = getattr(msg, "content", base_output)
            except Exception as e:  # pragma: no cover
                trace = state.setdefault("trace", [])
                trace.append(f"GEMINI_FALLBACK: {e}")
                enhanced = base_output
        trace = state.setdefault("trace", [])
        trace.append("FINALIZE")
        state["output"] = enhanced
        return state

    # ---------------- Build / Access graph ------------- #
    def build_graph(self):
        if self.compiled_graph is not None:
            return self.compiled_graph
        graph = StateGraph(AgentState)
        graph.add_node("router", self.router_node)
        graph.add_node("tool", self.tool_node)
        graph.add_node("finalize", self.finalize_node)
        graph.set_entry_point("router")
        graph.add_edge("router", "tool")
        graph.add_edge("tool", "finalize")
        graph.add_edge("finalize", END)
        self.compiled_graph = graph.compile()

        return self.compiled_graph


# Singleton mặc định (giữ tương thích với inference.run import build_graph)
_DEFAULT_INSTANCE: GeminiAgentGraph | None = None


def build_graph():  # noqa: D401 - giữ API cũ
    """Trả về compiled graph mặc định (singleton)."""
    global _DEFAULT_INSTANCE
    if _DEFAULT_INSTANCE is None:
        _DEFAULT_INSTANCE = GeminiAgentGraph(system_prompt=IHOS_SYSTEM_PROMPT.format(current_date=datetime.datetime.today()))
    return _DEFAULT_INSTANCE.build_graph()
