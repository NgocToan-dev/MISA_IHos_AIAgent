"""Registry các tool dùng @tool để LLM (Gemini) tự chọn.

Lưu ý: Tool arguments nên đơn giản (kiểu nguyên thủy) để model dễ sinh.
"""
from __future__ import annotations
from typing import List, Dict
import math, json
from services.api.business_service import search_hospitals_by_keyword
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Tính toán biểu thức toán học Python cơ bản (hàm math, + - * / **). Tham số: expression."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
    code = compile(expression, '<calc>', 'eval')
    for name in code.co_names:
        if name not in allowed:
            raise ValueError(f"Tên không hợp lệ: {name}")
    return str(eval(code, {"__builtins__": {}}, allowed))


def _serialize_hospital(h) -> Dict[str, str]:
    return {
        "id": h.id,
        "name": h.name,
        "province": h.province,
        "specialties": h.specialties,
        "level": h.level,
    }


@tool
def hospital_list(keyword: str) -> str:
    """Tra cứu bệnh viện theo từ khóa (tên, tỉnh, chuyên khoa). Tham số: keyword (ví dụ 'Ha Noi tim mach'). Trả JSON."""
    hospitals = search_hospitals_by_keyword(keyword)
    items = [_serialize_hospital(h) for h in hospitals]
    return json.dumps({"count": len(items), "items": items}, ensure_ascii=False)


@tool
def echo(text: str = "") -> str:
    """Trả lại nguyên văn input (debug). Tham số text (optional)."""
    return f"ECHO: {text}" if text else "ECHO: (empty)"


ALL_TOOLS = [calculator, hospital_list, echo]

__all__ = ["calculator", "hospital_list", "echo", "ALL_TOOLS"]
