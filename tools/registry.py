"""Registry các tool dùng @tool để LLM (Gemini) tự chọn.

Lưu ý: Tool arguments nên đơn giản (kiểu nguyên thủy) để model dễ sinh.
"""
from __future__ import annotations
from typing import List, Dict, Optional
import json
from services.api.business_service import search_hospitals_by_keyword
from services.mongo.mongo_repo import find_many
from langchain_core.tools import tool
from services.milvus.milvus_repo import (
    compute_embedding,
    search_embeddings,
    load_collection,
    create_index,
)
import os


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

ALL_TOOLS = [hospital_list, echo, all_employees_with_departments_info, employees_by_department_names, ihos_doc_search]

__all__ = [
    "hospital_list", "echo",
    "all_employees_with_departments_info", "employees_by_department_names", "ihos_doc_search", "ALL_TOOLS"
]
