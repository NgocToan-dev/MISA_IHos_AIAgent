"""Data object (Pydantic) definitions cho input/output của agent.

Không phải AI model. Dùng để chuẩn hoá:
 - Tham số vào agent
 - Kết quả tool
 - Kết quả cuối agent
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Hospital(BaseModel):
    id: str
    name: str
    province: str
    specialties: str
    level: str


class ToolResult(BaseModel):
    tool_name: str = Field(..., description="Tên tool được gọi")
    output: str = Field(..., description="Kết quả tool dạng text/JSON string")
    raw_args: Optional[Dict[str, Any]] = Field(None, description="Args gốc truyền vào tool (nếu có)")


class AgentResponse(BaseModel):
    output: str
    selected_tool: Optional[str]
    intermediate: List[str]
    trace: List[str]
    tool_results: Optional[List[ToolResult]] = None
    session_id: Optional[str] = Field(None, description="ID phiên hội thoại để duy trì memory")


class AgentRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi / lệnh đầu vào")
    session_id: Optional[str] = Field(None, description="ID phiên (client giữ và gửi lại để lưu ngữ cảnh)")


class Room(BaseModel):
    id: str
    employee_name: str
    room_name: str
    time_start: str
    time_end: str
    booking_date: str
    purpose: str = ""


__all__ = [
    "Hospital",
    "ToolResult",
    "Room",
    "AgentResponse",
    "AgentRequest",
]
