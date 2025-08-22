"""Script test đẩy dữ liệu hospitals vào Milvus bằng Gemini embedding.

Chạy:
  python -m tests.milvus_push_test --limit 10 --query "tim mach Ha Noi"

Yêu cầu biến môi trường:
  - GOOGLE_API_KEY
  - MILVUS_URL, MILVUS_TOKEN (hoặc MILVUS_USER + MILVUS_PASSWORD)
  - MILVUS_COLLECTION

Luồng:
 1. Load .env (nếu có)
 2. Lấy hospitals từ Mongo; nếu rỗng tạo vài mẫu.
 3. Index vào Milvus (Gemini-only).
 4. Thực hiện search thử.
"""
from __future__ import annotations

import argparse
import os
from typing import List

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from services.api.business_service import (
    search_hospitals,
    create_hospital,
    bulk_create_hospitals,
)
from model.schemas import Hospital
from services.milvus.milvus_repo import index_hospitals, search_text, ensure_collection, create_index, load_collection


def ensure_sample_data(min_count: int = 5) -> List[Hospital]:
    hospitals = search_hospitals(limit=min_count)
    if len(hospitals) >= min_count:
        return hospitals
    samples = [
        Hospital(id="temp1", name="Bệnh viện Tim Hà Nội", province="Ha Noi", specialties="tim mach", level="A"),
        Hospital(id="temp2", name="Bệnh viện Phổi Trung ương", province="Ha Noi", specialties="ho hap", level="Trung ương"),
        Hospital(id="temp3", name="Bệnh viện Ung Bướu HCM", province="TP HCM", specialties="ung buou", level="Chuyên khoa"),
        Hospital(id="temp4", name="Bệnh viện Nhi Đồng 1", province="TP HCM", specialties="nhi khoa", level="Trung ương"),
        Hospital(id="temp5", name="Bệnh viện Da liễu", province="Da Nang", specialties="da lieu", level="B"),
    ]
    created = index = []
    # Chuyển về dict để remove id trước khi insert (Mongo sinh _id)
    created_ids = bulk_create_hospitals([s.model_dump() for s in samples])
    return search_hospitals(limit=min_count)


def run(limit: int, query: str, build_index: bool):
    data = ensure_sample_data()
    subset = data[:limit]
    print(f"Indexing {len(subset)} hospitals vào Milvus ...")
    count = index_hospitals(subset)
    print(f"Đã index {count} hospitals.")
    if build_index:
        # Lấy dimension từ embedding đầu tiên (đã index) bằng ensure_collection (no op nếu tồn tại)
        try:
            create_index()
            load_collection()
            print("Index vector đã tạo.")
        except Exception as e:
            print("Tạo index thất bại (tiếp tục):", e)
    if query:
        print(f"Search thử với query: {query}")
        try:
            hits = search_text(query, k=5)
            for i, h in enumerate(hits, 1):
                print(f"{i}. hospital_id={h.get('hospital_id')} distance={h.get('distance'):.4f} province={h.get('province')} level={h.get('level')}")
        except Exception as e:
            print("Search lỗi:", e)


def parse_args():
    p = argparse.ArgumentParser(description="Test push hospitals embeddings to Milvus")
    p.add_argument("--limit", type=int, default=5, help="Số hospital lấy để index")
    p.add_argument("--query", type=str, default="", help="Chuỗi search thử sau khi index")
    p.add_argument("--no-build-index", action="store_true", help="Không tạo index IVF_FLAT")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(limit=args.limit, query=args.query, build_index=not args.no_build_index)
