"""Mongo repository base utilities.

Kết nối tới MongoDB local: mongodb://localhost:27017/misa_ihos

Thiết kế đơn giản để business layer có thể gọi nhanh:

from services.mongo.mongo_repo import get_collection, insert_one, find_many

Có thể mở rộng thêm caching / logging sau này.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult
import threading
import os

_lock = threading.Lock()
_client: Optional[MongoClient] = None


def _build_uri() -> str:
	# Cho phép override bằng biến môi trường MONGO_URI
	return os.getenv("MONGO_URI", "mongodb://localhost:27017")


def get_client() -> MongoClient:
	global _client
	if _client is None:
		with _lock:
			if _client is None:
				_client = MongoClient(_build_uri(), uuidRepresentation="standard")
	return _client


def get_db(name: str = "misa_ihos") -> Database:
	return get_client()[name]


def get_collection(name: str, db: Optional[str] = None) -> Collection:
	database = get_db(db or "misa_ihos")
	return database[name]


# ---------- CRUD convenience wrappers ----------

def insert_one(col: str, doc: Dict[str, Any]) -> str:
	result: InsertOneResult = get_collection(col).insert_one(doc)
	return str(result.inserted_id)


def insert_many(col: str, docs: Iterable[Dict[str, Any]]) -> List[str]:
	result: InsertManyResult = get_collection(col).insert_many(list(docs))
	return [str(_id) for _id in result.inserted_ids]


def find_one(col: str, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
	return get_collection(col).find_one(query, projection)


def find_many(
	col: str,
	query: Optional[Dict[str, Any]] = None,
	projection: Optional[Dict[str, int]] = None,
	sort: Optional[List[tuple[str, int]]] = None,
	limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
	cursor = get_collection(col).find(query or {}, projection)
	if sort:
		cursor = cursor.sort(sort)
	if limit:
		cursor = cursor.limit(limit)
	return list(cursor)


def update_one(col: str, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> int:
	result: UpdateResult = get_collection(col).update_one(query, update, upsert=upsert)
	return result.modified_count


def delete_one(col: str, query: Dict[str, Any]) -> int:
	result: DeleteResult = get_collection(col).delete_one(query)
	return result.deleted_count


def ensure_indexes(col: str, indexes: List[tuple[str, int]]) -> None:
	collection = get_collection(col)
	for field, direction in indexes:
		collection.create_index([(field, direction)])


__all__ = [
	"get_client",
	"get_db",
	"get_collection",
	"insert_one",
	"insert_many",
	"find_one",
	"find_many",
	"update_one",
	"delete_one",
	"ensure_indexes",
	"ASCENDING",
	"DESCENDING",
]
