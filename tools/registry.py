"""Registry các tool dùng @tool để LLM (Gemini) tự chọn.

Lưu ý: Tool arguments nên đơn giản (kiểu nguyên thủy) để model dễ sinh.
"""
from __future__ import annotations
from typing import List, Dict
import math, json
from services.api.business_service import search_hospitals_by_keyword
from services.mongo.mongo_repo import find_many
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


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

duckduckgo_search = DuckDuckGoSearchRun()

@tool
def internet_search(query: str) -> str:
    """
    Tìm kiếm thông tin trên internet bằng DuckDuckGo.
    Tham số: query (nội dung cần tìm kiếm).
    Trả về kết quả tìm kiếm dạng văn bản.
    """
    return duckduckgo_search.invoke(query)

@tool
def available_meeting_rooms(time_start: str, time_end: str, booking_date: str) -> str:
    """
    Tìm kiếm phòng họp còn trống trong bệnh viện theo khoảng thời gian và ngày đặt.
    Tham số:
        time_start: thời gian bắt đầu (HH:MM)
        time_end: thời gian kết thúc (HH:MM)
        booking_date: ngày đặt phòng (DD-MM-YYYY)
    Trả về JSON danh sách phòng họp còn trống.
    """
    rooms = find_many("meeting_rooms")
    bookings = find_many(
        "schedule_booking",
        {
            "booking_date": booking_date,
            "$or": [
                {"start_time": {"$lt": time_end, "$gte": time_start}},
                {"end_time": {"$gt": time_start, "$lte": time_end}},
                {"start_time": {"$lte": time_start}, "end_time": {"$gte": time_end}}
            ]
        }
    )
    booked_names = set(b["room_name"] for b in bookings)
    available_rooms = [
        {
            "room_name": r.get("room_name"),
            "room_code": r.get("room_code"),
            "location": r.get("location"),
            "capacity": r.get("capacity"),
            "equipment_list": r.get("equipment_list"),
            "is_available": r.get("is_available"),
        }
        for r in rooms
        if r.get("is_available", True) and r.get("room_name") not in booked_names
    ]
    return json.dumps({"count": len(available_rooms), "items": available_rooms}, ensure_ascii=False)

@tool
def book_meeting_room(
    employee_name: str,
    room_name: str,
    time_start: str,
    time_end: str,
    booking_date: str,
    purpose: str = ""
) -> str:
    """
    Đặt phòng họp trong bệnh viện.
    Tham số:
        employee_name: tên người đặt phòng
        room_name: tên phòng họp
        time_start: thời gian bắt đầu (HH:MM)
        time_end: thời gian kết thúc (HH:MM)
        booking_date: ngày đặt phòng (DD-MM-YYYY)
        purpose: mục đích họp (tùy chọn)
    Trả về kết quả đặt phòng (JSON).
    """
    rooms = find_many("meeting_rooms", {"room_name": room_name}, limit=1)
    if not rooms:
        return json.dumps({"success": False, "message": "Không tìm thấy phòng họp."}, ensure_ascii=False)
    bookings = find_many(
        "schedule_booking",
        {
            "room_name": room_name,
            "booking_date": booking_date,
            "$or": [
                {"start_time": {"$lt": time_end, "$gte": time_start}},
                {"end_time": {"$gt": time_start, "$lte": time_end}},
                {"start_time": {"$lte": time_start}, "end_time": {"$gte": time_end}}
            ]
        }
    )
    if bookings:
        return json.dumps({"success": False, "message": "Phòng đã bị đặt trong khoảng thời gian này."}, ensure_ascii=False)
    booking_info = {
        "employee_name": employee_name,
        "room_name": room_name,
        "booking_date": booking_date,
        "start_time": time_start,
        "end_time": time_end,
        "purpose": purpose
    }
    from services.mongo.mongo_repo import insert_one
    insert_one("schedule_booking", booking_info)
    return json.dumps({"success": True, "message": "Đặt phòng thành công.", "booking": booking_info}, ensure_ascii=False)

@tool
def list_meeting_rooms() -> str:
    """
    Lấy danh sách thông tin các phòng họp của bệnh viện.
    Trả về JSON gồm các trường: room_name, room_code, location, capacity, equipment_list, is_available, created_at.
    """
    rooms = find_many("meeting_rooms")
    items = [
        {
            "room_name": r.get("room_name"),
            "room_code": r.get("room_code"),
            "location": r.get("location"),
            "capacity": r.get("capacity"),
            "equipment_list": r.get("equipment_list"),
            "is_available": r.get("is_available"),
            "created_at": r.get("created_at"),
        }
        for r in rooms
    ]
    return json.dumps({"count": len(items), "items": items}, ensure_ascii=False)

ALL_TOOLS = [
    calculator,
    hospital_list,
    echo,
    all_employees_with_departments_info,
    employees_by_department_names,
    internet_search,
    available_meeting_rooms,
    book_meeting_room,
    list_meeting_rooms  # Thêm tool mới vào danh sách
]

__all__ = [
    "calculator", "hospital_list", "echo",
    "all_employees_with_departments_info", "employees_by_department_names",
    "internet_search", "available_meeting_rooms", "book_meeting_room", "list_meeting_rooms", "ALL_TOOLS"
]
