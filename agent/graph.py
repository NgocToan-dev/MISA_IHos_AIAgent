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
from typing import Callable, Optional, Any
import os

# Tích hợp Gemini qua langchain-google-genai
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.messages import HumanMessage
except ImportError:  # Chưa cài gói gemini
    ChatGoogleGenerativeAI = None  # type: ignore
    HumanMessage = None  # type: ignore
from state.state import AgentState
from tools.registry import ALL_TOOLS  # @tool based


class GeminiAgentGraph:
    """Đóng gói logic xây dựng và vận hành LangGraph agent.

    Thuộc tính chính:
        gemini_model_name: tên model Gemini (ví dụ: gemini-1.5-flash).
        gemini_llm: instance ChatGoogleGenerativeAI hoặc None nếu không sẵn sàng.
        compiled_graph: graph đã compile (cache). Dùng lazy compile.

    Mở rộng trong tương lai:
        - Thêm checkpoint (Redis/Postgres) -> lưu ở thuộc tính saver/store
        - Thêm memory (short / long term)
        - Thêm node động (subgraph, branch theo condition)
    """

    def __init__(
        self,
        gemini_model_name: str | None = None,
        temperature: float = 0.0,
        enable_gemini: bool = True,
        tools: list | None = None,
    ) -> None:
        self.gemini_model_name = gemini_model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self.compiled_graph = None
        self.gemini_llm = None
        self.tools = tools if tools is not None else ALL_TOOLS
        if enable_gemini and ChatGoogleGenerativeAI is not None and os.getenv("GOOGLE_API_KEY"):
            try:
                self.gemini_llm = ChatGoogleGenerativeAI(model=self.gemini_model_name, temperature=temperature)
            except Exception:
                self.gemini_llm = None

    # ---------------- Node definitions ---------------- #
    def router_node(self, state: AgentState) -> AgentState:
        """Nếu có Gemini + tool calling: để LLM tự chọn, ngược lại fallback route cũ."""
        query = state.get("query", "")
        trace = state.setdefault("trace", [])
        if self.gemini_llm is not None and HumanMessage is not None:
            try:
                llm_tools = self.gemini_llm.bind_tools(self.tools)
                ai_msg = llm_tools.invoke([HumanMessage(content=query)])
                tool_calls = getattr(ai_msg, "tool_calls", []) or []
                if tool_calls:
                    first = tool_calls[0]
                    tool_name = first.get("name")
                    args = first.get("args", {})
                    state["selected_tool"] = tool_name
                    tool_args = state.get("tool_args")
                    if tool_args is None:
                        tool_args = {}
                        state["tool_args"] = tool_args
                    tool_args[tool_name] = args
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
        tool_map = {t.name: t for t in self.tools}
        if tool_name in tool_map and self.gemini_llm is not None:
            args_map = state.get("tool_args", {}).get(tool_name, {}) or {}
            # Auto-fill missing required single argument with query
            if tool_name in tool_map:
                tool_obj = tool_map[tool_name]
                try:
                    schema = getattr(tool_obj, "args_schema", None)
                    if schema is not None:
                        fields = getattr(schema, "model_fields", {})
                        required = [k for k, f in fields.items() if getattr(f, "is_required", False)]
                        if not required:
                            required = [k for k, f in fields.items() if getattr(f, "default", object()) is ...]
                        if (not args_map) and len(required) == 1:
                            args_map = {required[0]: query}
                        elif not args_map:
                            name_map = {"echo": "text", "calculator": "expression", "hospital_list": "keyword"}
                            param_name = name_map.get(tool_name or "")
                            if param_name:
                                args_map = {param_name: query}
                except Exception:  # pragma: no cover
                    pass
            # Persist back
            if args_map:
                tool_args_all = state.get("tool_args") or {}
                tool_args_all[tool_name] = args_map
                state["tool_args"] = tool_args_all
            try:
                result = tool_map[tool_name].invoke(args_map)
            except Exception as e:  # pragma: no cover
                result = f"ToolError: {e}"
        elif tool_name == "echo":
            result = f"ECHO: {query}"
        else:
            result = f"Unknown tool: {tool_name}"
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
                    "Bạn là trợ lý. Người dùng hỏi: '" + query + "'.\n"
                    "Kết quả trung gian/tool: '" + base_output + "'.\n"
                    "Hãy trả lời ngắn gọn và rõ ràng bằng tiếng Việt."
                )
                msg = self.gemini_llm.invoke([HumanMessage(content=prompt)])
                enhanced = getattr(msg, "content", base_output)
            except Exception as e:  # pragma: no cover - fallback
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
        _DEFAULT_INSTANCE = GeminiAgentGraph()
    return _DEFAULT_INSTANCE.build_graph()
