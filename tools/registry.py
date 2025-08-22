"""Registry các tool dùng @tool để LLM (Gemini) tự chọn.

Lưu ý: Tool arguments nên đơn giản (kiểu nguyên thủy) để model dễ sinh.
"""
from __future__ import annotations
from typing import List, Dict, Optional
import json
from services.api.business_service import search_hospitals_by_keyword
from services.mongo.mongo_repo import find_many
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from services.milvus.milvus_repo import (
    compute_embedding,
    search_embeddings,
    load_collection,
    create_index,
)
import os
from collections import Counter, defaultdict
from datetime import datetime
from datetime import timedelta
import traceback
from model.schemas import Room


@tool
def echo(text: str = "") -> str:
    """Trả lại nguyên văn input (debug). Tham số text (optional)."""
    return f"ECHO: {text}" if text else "ECHO: (empty)"

@tool
def all_employees() -> str:
    """
    Trả về thông tin tất cả nhân viên.
    Trả JSON gồm danh sách nhân viên.
    Nhân viên bao gồm các thông tin: tên, tuổi, ngày sinh, số điện thoại, vị trí làm việc
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


@tool
def ihos_doc_search(query: str, collection: Optional[str] = None) -> str:
    """Tìm kiếm semantic các đoạn văn bản phù hợp nhất (top 3) trong Milvus (ihos_documents).
    Tham số: query (câu hỏi). Trả JSON gồm: query, matches (danh sách tối đa 3), và best_match (phần tử đầu để tương thích cũ).
    Mỗi phần tử: {doc_id, chunk_index, score, text_truncated}.
    """
    coll = collection or os.getenv("MILVUS_TEXT_COLLECTION", "ihos_documents")
    try:
        # bảo đảm collection load
        try:
            load_collection(collection=coll)
        except Exception:
            try:
                create_index(collection=coll)
                load_collection(collection=coll)
            except Exception:
                pass
        emb = compute_embedding(query)
        hits = search_embeddings(emb, k=3, collection=coll, output_fields=["doc_id", "text", "chunk_index"])
        if not hits:
            return json.dumps({"query": query, "matches": [], "best_match": None, "message": "Không tìm thấy tài liệu phù hợp"}, ensure_ascii=False)
        matches = []
        for h in hits:
            full_text = h.get("text") or ""
            truncated = full_text[:500] + ("..." if len(full_text) > 500 else "")
            matches.append({
                "doc_id": h.get("doc_id"),
                "chunk_index": h.get("chunk_index"),
                "score": h.get("distance"),
                "text_truncated": truncated,
            })
        return json.dumps({"query": query, "matches": matches, "best_match": matches[0]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"MilvusSearchError: {e}"}, ensure_ascii=False)

ALL_TOOLS = [echo, all_employees, employees_by_department_names, ihos_doc_search]
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
def chart_employees_by_department() -> str:
    """Thống kê số lượng nhân viên theo phòng ban (dùng cho Bar/Pie chart)."""
    employees = find_many("employees")
    dept_counts: Counter[str] = Counter()
    for emp in employees:
        dept = emp.get("department_name") or emp.get("department_id") or "Unknown"
        dept_counts[str(dept)] += 1
    data = [{"label": k, "value": v} for k, v in dept_counts.most_common()]
    return json.dumps({"chart": "employees_by_department", "description": "Số nhân viên theo phòng ban", "data": data}, ensure_ascii=False)

@tool
def chart_employees_by_gender() -> str:
    """Thống kê số lượng nhân viên theo giới tính (Bar/Pie)."""
    employees = find_many("employees")
    gender_counts: Counter[str] = Counter()
    for emp in employees:
        gender = emp.get("gender") or "Khác"  # 'Nam', 'Nữ', ...
        gender_counts[str(gender)] += 1
    data = [{"label": k, "value": v} for k, v in gender_counts.most_common()]
    return json.dumps({"chart": "employees_by_gender", "description": "Số nhân viên theo giới tính", "data": data}, ensure_ascii=False)

@tool
def chart_age_distribution(buckets: Optional[List[int]] = None) -> str:
    """Phân bố độ tuổi nhân viên theo nhóm (Histogram). Tham số tùy chọn: buckets (danh sách mốc tuổi)."""
    employees = find_many("employees")
    ages: List[int] = []
    for emp in employees:
        val = emp.get("employee_age")
        if isinstance(val, (int, float)):
            ages.append(int(val))
    if not ages:
        return json.dumps({"chart": "age_distribution", "description": "Không có dữ liệu tuổi", "data": []}, ensure_ascii=False)
    buckets = sorted(set(buckets)) if buckets else [20,25,30,35,40,45,50,55,60]
    # Build ranges
    ranges = []
    prev = 0
    for b in buckets:
        ranges.append((prev, b))
        prev = b
    ranges.append((prev, 200))  # tail
    counts = []
    for start, end in ranges:
        label = f"{start}-{end-1}" if end != 200 else f">= {start}"
        c = 0
        for a in ages:
            if start <= a < end:
                c += 1
        if c > 0:
            counts.append({"range": label, "count": c})
    return json.dumps({"chart": "age_distribution", "description": "Phân bố độ tuổi", "data": counts}, ensure_ascii=False)

@tool
def chart_salary_by_department() -> str:
    """Thống kê lương trung bình, min, max theo phòng ban (Bar chart với avg hoặc Box-like data)."""
    employees = find_many("employees")
    agg: Dict[str, List[float]] = defaultdict(list)
    for emp in employees:
        dept = emp.get("department_name") or emp.get("department_id") or "Unknown"
        salary = emp.get("salary")
        if isinstance(salary, (int, float)):
            agg[str(dept)].append(float(salary))
    data = []
    for dept, vals in agg.items():
        avg = sum(vals)/len(vals) if vals else 0
        data.append({
            "department": dept,
            "avg_salary": round(avg,2),
            "min_salary": min(vals),
            "max_salary": max(vals),
            "count": len(vals)
        })
    data.sort(key=lambda x: x["avg_salary"], reverse=True)
    return json.dumps({"chart": "salary_by_department", "description": "Lương theo phòng ban", "data": data}, ensure_ascii=False)

@tool
def chart_avg_salary_by_experience() -> str:
    """Lương trung bình theo số năm kinh nghiệm (Line/Bar)."""
    employees = find_many("employees")
    exp_map: Dict[int, List[float]] = defaultdict(list)
    for emp in employees:
        yrs = emp.get("years_of_experience")
        sal = emp.get("salary")
        if isinstance(yrs, (int, float)) and isinstance(sal, (int, float)):
            exp_map[int(yrs)].append(float(sal))
    data = []
    for yrs, vals in sorted(exp_map.items(), key=lambda x: x[0]):
        avg = sum(vals)/len(vals)
        data.append({"years_of_experience": yrs, "avg_salary": round(avg,2), "count": len(vals)})
    return json.dumps({"chart": "avg_salary_by_experience", "description": "Lương trung bình theo kinh nghiệm", "data": data}, ensure_ascii=False)

@tool
def chart_education_level_distribution() -> str:
    """Phân bố trình độ học vấn (Bar/Pie)."""
    employees = find_many("employees")
    edu_counts: Counter[str] = Counter()
    for emp in employees:
        edu = emp.get("education_level") or "Khác"
        edu_counts[str(edu)] += 1
    data = [{"label": k, "value": v} for k, v in edu_counts.most_common()]
    return json.dumps({"chart": "education_level_distribution", "description": "Phân bố trình độ học vấn", "data": data}, ensure_ascii=False)

@tool
def chart_headcount_by_hire_year() -> str:
    """Xu hướng số lượng tuyển dụng theo năm (Line/Bar)."""
    employees = find_many("employees")
    year_counts: Counter[int] = Counter()
    for emp in employees:
        hire = emp.get("hire_date")
        year = None
        if isinstance(hire, datetime):
            year = hire.year
        elif isinstance(hire, str):
            # thử định dạng DD-MM-YYYY
            for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                try:
                    year = datetime.strptime(hire, fmt).year
                    break
                except Exception:
                    continue
        if year:
            year_counts[year] += 1
    data = [{"year": y, "count": year_counts[y]} for y in sorted(year_counts.keys())]
    return json.dumps({"chart": "headcount_by_hire_year", "description": "Số nhân viên theo năm tuyển dụng", "data": data}, ensure_ascii=False)

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
    room: Room
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
    # Helper: normalize time strings and booking_date tokens
    from services.mongo.mongo_repo import insert_one
    booking_info = {
        "employee_name": room.employee_name,
        "room_name": room.room_name,
        "time_start": room.time_start,
        "time_end": room.time_end,
        "booking_date": room.booking_date,
        "purpose": room.purpose,
    }
    try:
        insert_one("schedule_booking", booking_info)
    except Exception as e:
        print(f"[book_meeting_room] insert error for booking={booking_info}: {e}")
        traceback.print_exc()
        return json.dumps({"success": False, "message": f"Lỗi hệ thống khi lưu booking: {e}"}, ensure_ascii=False)
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

# Cập nhật lại danh sách ALL_TOOLS hợp nhất
ALL_TOOLS = [
    echo,
    all_employees,
    employees_by_department_names,
    ihos_doc_search,
    internet_search,
    chart_employees_by_department,
    chart_employees_by_gender,
    chart_age_distribution,
    chart_salary_by_department,
    chart_avg_salary_by_experience,
    chart_education_level_distribution,
    chart_headcount_by_hire_year,
    available_meeting_rooms,
    book_meeting_room,
    list_meeting_rooms,
]

__all__ = [
    "echo",
    "all_employees", "employees_by_department_names", "ihos_doc_search",
    "internet_search", "ALL_TOOLS",
    "chart_employees_by_department",
    "chart_employees_by_gender",
    "chart_age_distribution",
    "chart_salary_by_department",
    "chart_avg_salary_by_experience",
    "chart_education_level_distribution",
    "chart_headcount_by_hire_year",
    "available_meeting_rooms",
    "book_meeting_room",
    "list_meeting_rooms",
]
