"""Business service layer.

Chứa các hàm thao tác dữ liệu domain (ví dụ: Hospital) sử dụng mongo_repo.
Tách riêng để controller / agent gọi mà không đụng trực tiếp vào driver.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Iterable, Union
from bson import ObjectId
from services.mongo.mongo_repo import (
	insert_one,
	insert_many,
	find_one,
	find_many,
	update_one,
	delete_one,
	ensure_indexes,
	ASCENDING,
	get_collection,
)
from model.schemas import Hospital

HOSPITAL_COLLECTION = "hospitals"


def init_business_indexes():
	"""Gọi khi khởi động để đảm bảo index cần thiết."""
	ensure_indexes(HOSPITAL_COLLECTION, [("province", ASCENDING), ("level", ASCENDING)])


def _hospital_to_doc(h: Union[Hospital, Dict[str, Any]]) -> Dict[str, Any]:
	if isinstance(h, Hospital):
		d = h.model_dump()
	else:
		d = dict(h)
	# Không lưu field id (dùng _id của Mongo)
	d.pop("id", None)
	return d


def _doc_to_hospital(doc: Optional[Dict[str, Any]]) -> Optional[Hospital]:
	if not doc:
		return None
	data = dict(doc)
	_id = data.pop("_id", None)
	data["id"] = str(_id) if _id is not None else None
	# Fill default None for missing fields to satisfy model if needed
	return Hospital(**data)


def create_hospital(hospital: Union[Hospital, Dict[str, Any]]) -> Hospital:
	doc = _hospital_to_doc(hospital)
	inserted_id = insert_one(HOSPITAL_COLLECTION, doc)
	doc["_id"] = ObjectId(inserted_id)
	return _doc_to_hospital(doc)  # type: ignore


def bulk_create_hospitals(items: Iterable[Union[Hospital, Dict[str, Any]]]) -> List[Hospital]:
	docs = [_hospital_to_doc(it) for it in items]
	ids = insert_many(HOSPITAL_COLLECTION, docs)
	hospitals: List[Hospital] = []
	for doc, _id in zip(docs, ids):
		doc["_id"] = ObjectId(_id)
		hospitals.append(_doc_to_hospital(doc))  # type: ignore
	return hospitals


def get_hospital_by_id(hospital_id: str) -> Optional[Hospital]:
	try:
		oid = ObjectId(hospital_id)
	except Exception:
		return None
	return _doc_to_hospital(find_one(HOSPITAL_COLLECTION, {"_id": oid}))


def search_hospitals(province: Optional[str] = None, level: Optional[str] = None, limit: int = 50) -> List[Hospital]:
	query: Dict[str, Any] = {}
	if province:
		query["province"] = province
	if level:
		query["level"] = level
	docs = find_many(HOSPITAL_COLLECTION, query, limit=limit)
	return [h for h in (_doc_to_hospital(d) for d in docs) if h]


def search_hospitals_by_keyword(keyword: str, limit: int = 5) -> List[Hospital]:
	"""Full-text (regex) tìm kiếm theo nhiều field. Đơn giản, chưa dùng Atlas Search.

	Mỗi token phải khớp ít nhất một trong các field (name, province, specialties, level).
	"""
	tokens = [t for t in keyword.strip().split() if t]
	col = get_collection(HOSPITAL_COLLECTION)
	if not tokens:
		cursor = col.find({}).limit(limit)
		return [h for h in (_doc_to_hospital(d) for d in cursor) if h]
	and_clauses = []
	for t in tokens:
		regex = {"$regex": t, "$options": "i"}
		and_clauses.append({
			"$or": [
				{"name": regex},
				{"province": regex},
				{"specialties": regex},
				{"level": regex},
			]
		})
	query = {"$and": and_clauses}
	cursor = col.find(query).limit(limit)
	return [h for h in (_doc_to_hospital(d) for d in cursor) if h]


def update_hospital(hospital_id: str, patch: Dict[str, Any]) -> Optional[Hospital]:
	try:
		oid = ObjectId(hospital_id)
	except Exception:
		return None
	modified = update_one(HOSPITAL_COLLECTION, {"_id": oid}, {"$set": patch})
	if modified:
		return get_hospital_by_id(hospital_id)
	return get_hospital_by_id(hospital_id)  # return current (even if unchanged)


def delete_hospital(hospital_id: str) -> bool:
	try:
		oid = ObjectId(hospital_id)
	except Exception:
		return False
	deleted = delete_one(HOSPITAL_COLLECTION, {"_id": oid})
	return deleted > 0


__all__ = [
	"init_business_indexes",
	"create_hospital",
	"bulk_create_hospitals",
	"get_hospital_by_id",
	"search_hospitals",
	"search_hospitals_by_keyword",
	"update_hospital",
	"delete_hospital",
]

