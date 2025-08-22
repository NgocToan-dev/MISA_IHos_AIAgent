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
import json
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
    def llm_structured_node(self, state: AgentState) -> AgentState:
        """
        Use Gemini's structured output API when available to extract a tool call for booking.
        Expected structured output: {"tool_name": "book_meeting_room", "args": { ... }}
        On success this sets state['selected_tool'] and state['tool_args'][tool_name] and
        returns the updated state. On failure it falls back to router_node.
        """
        query = state.get("query", "")
        trace = state.setdefault("trace", [])
        if not (self.gemini_llm and HumanMessage is not None):
            trace.append("LLM_STRUCTURED_NOT_AVAILABLE -> fallback router")
            return self.router_node(state)

        prompt = (
            "Người dùng muốn ĐẶT PHÒNG. Hãy phân tích câu hỏi dưới đây và trả về "
            "một JSON duy nhất có 2 trường: 'tool_name' (ví dụ: book_meeting_room) "
            "và 'args' (object với các tham số). Trả chỉ JSON, không giải thích thêm.\n\n"
            f"Câu hỏi: {query}"
        )
        try:
            # Prefer structured API if available
            if hasattr(self.gemini_llm, "with_structured_output"):
                try:
                    structured_llm = self.gemini_llm.with_structured_output(output_schema=dict)
                    msg = structured_llm.invoke([HumanMessage(content=prompt)])
                    structured = getattr(msg, "structured_output", None) or getattr(msg, "output", None)
                    if structured is None:
                        content = getattr(msg, "content", None)
                        if isinstance(content, str):
                            try:
                                structured = json.loads(content)
                            except Exception:
                                structured = None
                    if isinstance(structured, str):
                        try:
                            structured = json.loads(structured)
                        except Exception:
                            structured = None
                    if isinstance(structured, dict):
                        tool_name = structured.get("tool_name")
                        args = structured.get("args", {}) or {}
                        if tool_name:
                            tool_args = state.get("tool_args") or {}
                            tool_args[tool_name] = args
                            state["tool_args"] = tool_args
                            state["selected_tool"] = tool_name
                            trace.append(f"LLM_STRUCTURED -> {tool_name} {args}")
                            return state
                except Exception as e:
                    trace.append(f"LLM_STRUCTURED_ERROR: {e}")

            # Fallback: plain invoke and try to parse JSON content
            llm_tools = self.gemini_llm.bind_tools(self.tools)
            msg = llm_tools.invoke([HumanMessage(content=prompt)])
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        tool_name = parsed.get("tool_name")
                        args = parsed.get("args", {}) or {}
                        if tool_name:
                            tool_args = state.get("tool_args") or {}
                            tool_args[tool_name] = args
                            state["tool_args"] = tool_args
                            state["selected_tool"] = tool_name
                            trace.append(f"LLM_STRUCTURED_FALLBACK -> {tool_name} {args}")
                            return state
                except Exception:
                    pass
        except Exception as e:
            trace.append(f"LLM_STRUCTURED_FATAL: {e}")

        trace.append("LLM_STRUCTURED_NO_RESULT -> fallback router")
        return self.router_node(state)

    def dispatch_node(self, state: AgentState) -> AgentState:
        """
        Dispatch/condition node: if query indicates booking intent, route to
        `llm_structured_node` to force structured LLM parsing; otherwise use
        `router_node` for general tool routing.
        """
        query = (state.get("query") or "").lower()
        booking_triggers = ["đặt phòng", "đặt cho tôi", "đặt phòng giúp", "đặt hộ phòng", "đặt cuộc họp", "đoặt"]
        if any(tok in query for tok in booking_triggers):
            return self.llm_structured_node(state)
        return self.router_node(state)

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
        graph.add_node("dispatch", self.dispatch_node)
        graph.add_node("router", self.router_node)
        graph.add_node("tool", self.tool_node)
        graph.add_node("finalize", self.finalize_node)
        # Entry point is dispatch which will route to router or llm_structured_node
        graph.set_entry_point("dispatch")
        # dispatch (which may call llm_structured_node or router_node) then goes to tool
        graph.add_edge("dispatch", "tool")
        # keep router -> tool for backwards compatibility when router is used directly
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
        # IHOS_SYSTEM_PROMPT may contain braces/placeholders that cause str.format() to raise
        # KeyError if keys are missing. Try to format with current_date, but fall back to
        # the raw prompt if formatting fails.
        try:
            system_prompt = IHOS_SYSTEM_PROMPT.format(current_date=datetime.datetime.today())
        except Exception:
            system_prompt = IHOS_SYSTEM_PROMPT
        _DEFAULT_INSTANCE = GeminiAgentGraph(system_prompt=system_prompt)
    return _DEFAULT_INSTANCE.build_graph()
