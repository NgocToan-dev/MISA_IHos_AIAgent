"""Registry các tool dùng @tool để LLM (Gemini) tự chọn.

Lưu ý: Tool arguments nên đơn giản (kiểu nguyên thủy) để model dễ sinh.
"""
from __future__ import annotations
from typing import List, Dict
import math, json
from services.api.business_service import search_hospitals_by_keyword
from services.mongo.mongo_repo import find_many
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

@tool
def all_employees_with_departments_info() -> str:
    """
    Trả về thông tin tất cả nhân viên và phòng ban liên quan.
    Trả JSON gồm danh sách nhân viên và phòng ban.
    Nhân viên bao gồm các thông tin: tên, tuổi, ngày sinh, số điện thoại, vị trí làm việc
    Phòng ban bao gồm các thông tin: Tên phòng ban, số giường bệnh, trạng thái hoạt động
    """
    employees = find_many("employees")
    results = []
    for emp in employees:
        departments = find_many("departments", {"_id": emp.get("department_id")}, limit=1)
        dept = departments[0] if departments else None
        results.append({
            "employee": {
                "employee_name": emp.get("employee_name"),
                "employee_age": emp.get("employee_age"),
                "employee_birthday": str(emp.get("employee_birthday")),
                "phone_number": emp.get("phone_number"),
                "job_position_name": emp.get("job_position_name"),
            },
            "department": {
                "department_name": dept.get("department_name") if dept else None,
                "no_of_beds": dept.get("no_of_beds") if dept else None,
                "active_status": dept.get("active_status") if dept else None,
            } if dept else None
        })
    return json.dumps({"count": len(results), "items": results}, ensure_ascii=False)

@tool
def employees_by_department_names(department_names: List[str]) -> str:
    """
    Lấy danh sách nhân viên theo các phòng ban chỉ định bằng tên phòng ban.
    Tham số: department_names (danh sách tên phòng ban).
    Trả về JSON gồm danh sách nhân viên và phòng ban liên quan.
    """
    results = []
    for dept_name in department_names:
        departments = find_many("departments", {"department_name": dept_name}, limit=1)
        dept = departments[0] if departments else None
        if not dept:
            continue
        employees = find_many("employees", {"department_id": dept.get("_id")})
        for emp in employees:
            results.append({
                "employee": {
                    "employee_name": emp.get("employee_name"),
                    "employee_age": emp.get("employee_age"),
                    "employee_birthday": str(emp.get("employee_birthday")),
                    "phone_number": emp.get("phone_number"),
                    "job_position_name": emp.get("job_position_name"),
                },
                "department": {
                    "department_name": dept.get("department_name"),
                    "no_of_beds": dept.get("no_of_beds"),
                    "active_status": dept.get("active_status"),
                }
            })
    return json.dumps({"count": len(results), "items": results}, ensure_ascii=False)

ALL_TOOLS = [calculator, hospital_list, echo, all_employees_with_departments_info, employees_by_department_names]

__all__ = [
    "calculator", "hospital_list", "echo",
    "all_employees_with_departments_info", "employees_by_department_names", "ALL_TOOLS"
]
