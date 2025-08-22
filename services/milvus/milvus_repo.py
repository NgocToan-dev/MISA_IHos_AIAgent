"""Milvus (Zilliz Cloud) base connection & helper functions.

Env variables sử dụng:
 - MILVUS_URL
 - MILVUS_USER (user / database id tuỳ cấu hình serverless)
 - MILVUS_PASSWORD
 - (optional) MILVUS_TOKEN  -> override user:password
 - MILVUS_COLLECTION
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Iterable
from services.embedding.gemini_embedder import embed_text
try:
    from model.schemas import Hospital  # type: ignore
except Exception:  # pragma: no cover
    Hospital = Any  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()  # Load .env if exists

try:
    from pymilvus import (  # type: ignore
        MilvusClient,
        DataType,
        FieldSchema,
        CollectionSchema,
    )
except ImportError as e:  # pragma: no cover
    raise RuntimeError("pymilvus chưa được cài. Thêm vào requirements.txt") from e

_lock = threading.Lock()
_client: Optional[MilvusClient] = None
_MIGRATION_ATTEMPTED = False


def _get_uri() -> str:
    uri = os.getenv("MILVUS_URL")
    if not uri:
        raise RuntimeError("Thiếu MILVUS_URL")
    return uri


def _get_token() -> Optional[str]:
    # Zilliz serverless có thể dùng password làm token
    token = os.getenv("MILVUS_TOKEN")
    if token:
        return token
    pwd = os.getenv("MILVUS_PASSWORD")
    user = os.getenv("MILVUS_USER", os.getenv("MILVUS_USERNAME", "root"))
    if pwd:
        return f"{user}:{pwd}"
    return None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                token = _get_token() or ""
                _client = MilvusClient(uri=_get_uri(), token=token)
    return _client


def get_default_collection_name() -> str:
    name = os.getenv("MILVUS_COLLECTION")
    if not name:
        raise RuntimeError("Thiếu MILVUS_COLLECTION")
    return name


def _describe_collection(client: MilvusClient, coll_name: str) -> Optional[Dict[str, Any]]:
    try:
        if hasattr(client, "describe_collection"):
            return client.describe_collection(coll_name)  # type: ignore
    except Exception:
        return None
    return None


def ensure_collection(
    name: Optional[str] = None,
    dim: int = 768,
    metric_type: str = "COSINE",
    description: str = "Hospital embedding collection",
    force: bool = False,
) -> None:
    """Ensure collection exists with expected schema.

    Set env MILVUS_MIGRATE=1 to auto-drop & recreate if primary key field != 'id' or embedding dim mismatch.
    """
    client = get_client()
    coll_name = name or get_default_collection_name()
    exists = client.has_collection(coll_name)  # type: ignore[attr-defined]
    if exists and force:
        try:
            client.drop_collection(coll_name)
            exists = False
        except Exception:
            pass
    if exists:
        return

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="PK",
        ),
        FieldSchema(
            name="hospital_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Hospital logical id",
        ),
        FieldSchema(name="province", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="level", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(
        fields=fields, description=description, enable_dynamic_field=True
    )
    client.create_collection(collection_name=coll_name, schema=schema)


def insert_embeddings(
    records: Sequence[Dict[str, Any]],
    collection: Optional[str] = None,
) -> List[int]:
    global _MIGRATION_ATTEMPTED
    client = get_client()
    coll_name = collection or get_default_collection_name()
    data = [dict(r) for r in records]
    # Never send an 'id' field for auto_id collection
    for r in data:
        r.pop("id", None)
        r.pop("primary_key", None)
    try:
        res = client.insert(collection_name=coll_name, data=data)
    except Exception as e:
        msg = str(e)
        # If schema mismatch complaining about primary_key, drop & recreate minimal schema then reinsert.
        if "primary_key" in msg and not _MIGRATION_ATTEMPTED:
            _MIGRATION_ATTEMPTED = True
            emb_len = len(data[0].get("embedding", [])) if data else 768
            try:
                client.drop_collection(coll_name)
            except Exception:
                pass
            ensure_collection(dim=emb_len, force=True)
            res = client.insert(collection_name=coll_name, data=data)
        else:
            raise
    if isinstance(res, dict) and "ids" in res:
        return list(res["ids"])
    return list(res) if isinstance(res, (list, tuple)) else []  # type: ignore


def search_embeddings(
    embedding: List[float],
    k: int = 5,
    collection: Optional[str] = None,
    filter: Optional[str] = None,
    output_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    client = get_client()
    coll_name = collection or get_default_collection_name()
    res = client.search(
        collection_name=coll_name,
        data=[embedding],
        limit=k,
        filter=filter or "",
        output_fields=output_fields or ["hospital_id", "province", "level"],
    )
    # MilvusClient.search trả list các list hits
    if not res:
        return []
    hits = res[0]
    out: List[Dict[str, Any]] = []
    for h in hits:
        row = {"distance": h.get("distance")}
        fields = h.get("entity", {}) or {}
        row.update(fields)
        out.append(row)
    return out


def create_index(
    collection: Optional[str] = None,
    field: str = "embedding",
    metric_type: str = "COSINE",
    index_type: str = "AUTOINDEX",
    nlist: int = 1024,  # kept for backward compatibility (ignored by AUTOINDEX)
) -> None:
    client = get_client()
    coll_name = collection or get_default_collection_name()
    if not hasattr(client, "prepare_index_params"):
        raise RuntimeError("MilvusClient thiếu prepare_index_params (phiên bản pymilvus quá cũ)")
    prep = client.prepare_index_params()
    prep.add_index(
        field_name=field,
        index_type=index_type,
        metric_type=metric_type,
        params={"nlist": nlist} if index_type != "AUTOINDEX" else {},
    )
    # For MilvusClient the index_params already includes field info; don't pass field_name again
    client.create_index(collection_name=coll_name, index_params=prep)


def load_collection(collection: Optional[str] = None) -> None:
    client = get_client()
    coll_name = collection or get_default_collection_name()
    try:
        client.load_collection(collection_name=coll_name)
    except Exception as e:
        # If index not found, create a simple index then retry
        if "index not found" in str(e).lower():
            try:
                create_index(collection=coll_name)
                client.load_collection(collection_name=coll_name)
                return
            except Exception:
                raise
        raise


def compute_embedding(text: str) -> List[float]:
    """Gemini-only embedding (không fallback)."""
    vec = embed_text(text)
    if not vec:
        raise RuntimeError("Gemini embedding trả về rỗng")
    return vec


def index_hospital(hospital: Hospital) -> List[int]:  # type: ignore
    text = f"{getattr(hospital,'name','')} {getattr(hospital,'province','')} {getattr(hospital,'specialties','')} {getattr(hospital,'level','')}".strip()
    emb = compute_embedding(text)
    dim = len(emb)
    ensure_collection(dim=dim)
    # Defer loading until index created (optional)
    ids = insert_embeddings([
        {
            "hospital_id": getattr(hospital, 'id', None),
            "province": getattr(hospital, 'province', None),
            "level": getattr(hospital, 'level', None),
            "embedding": emb,
        }
    ])
    # Create index (idempotent) then load
    try:
        create_index()
    except Exception:
        pass
    try:
        load_collection()
    except Exception:
        pass
    return ids


def index_hospitals(hospitals: Iterable[Hospital], batch: int = 32) -> int:  # type: ignore
    iterator = iter(hospitals)
    try:
        first = next(iterator)
    except StopIteration:
        return 0
    first_emb = compute_embedding(f"{getattr(first,'name','')} {getattr(first,'province','')} {getattr(first,'specialties','')} {getattr(first,'level','')}".strip())
    dim = len(first_emb)
    ensure_collection(dim=dim)
    # Defer loading until caller performs search / index creation
    buf: List[Dict[str, Any]] = [
        {
            "hospital_id": getattr(first, 'id', None),
            "province": getattr(first, 'province', None),
            "level": getattr(first, 'level', None),
            "embedding": first_emb,
        }
    ]
    total = 0
    for h in iterator:
        text = f"{getattr(h,'name','')} {getattr(h,'province','')} {getattr(h,'specialties','')} {getattr(h,'level','')}".strip()
        emb = compute_embedding(text)
        if len(emb) != dim:
            raise ValueError("Embedding dimension không đồng nhất")
        buf.append({
            "hospital_id": getattr(h, 'id', None),
            "province": getattr(h, 'province', None),
            "level": getattr(h, 'level', None),
            "embedding": emb,
        })
        if len(buf) >= batch:
            insert_embeddings(buf)
            total += len(buf)
            buf.clear()
    if buf:
        insert_embeddings(buf)
        total += len(buf)
    # Build index once after all inserts and load collection
    try:
        create_index()
    except Exception:
        pass
    try:
        load_collection()
    except Exception:
        pass
    return total


def search_text(query: str, k: int = 5, filter: Optional[str] = None) -> List[Dict[str, Any]]:
    emb = compute_embedding(query)
    # Ensure collection loaded
    try:
        load_collection()
    except Exception:
        try:
            create_index()
            load_collection()
        except Exception:
            pass
    return search_embeddings(emb, k=k, filter=filter)


__all__ = [
    "get_client",
    "ensure_collection",
    "insert_embeddings",
    "search_embeddings",
    "create_index",
    "load_collection",
    "compute_embedding",
    "index_hospital",
    "index_hospitals",
    "search_text",
    # text chunk helpers
    "ensure_text_collection",
    "index_text_file",
]


# --------------------- TEXT CHUNK INGESTION ---------------------
def ensure_text_collection(
    name: Optional[str] = None,
    dim: int = 768,
    description: str = "Generic text chunk collection",
    force: bool = False,
) -> str:
    """Đảm bảo tồn tại collection để lưu text chunks.

    Schema:
      - id (auto, primary)
      - doc_id (VARCHAR)
      - chunk_index (INT64)
      - text (VARCHAR large)
      - embedding (FLOAT_VECTOR)
    """
    client = get_client()
    coll_name = name or os.getenv("MILVUS_TEXT_COLLECTION", "ihos_documents")
    exists = client.has_collection(coll_name)  # type: ignore[attr-defined]
    if exists and force:
        try:
            client.drop_collection(coll_name)
            exists = False
        except Exception:
            pass
    if exists:
        return coll_name
    from pymilvus import FieldSchema, DataType, CollectionSchema  # type: ignore

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description=description, enable_dynamic_field=True)
    client.create_collection(collection_name=coll_name, schema=schema)
    return coll_name


def _split_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Tách văn bản dài thành các chunk theo số "từ" (word) gần giống token.

    overlap: số từ cuối của chunk trước sẽ được lặp lại ở chunk sau để giữ ngữ cảnh.
    """
    if chunk_size <= 0:
        chunk_size = 800
    if overlap < 0:
        overlap = 0
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end == n:
            break
        # next start with overlap
        start = end - overlap if overlap > 0 else end
        if start < 0:
            start = 0
    return chunks


def index_text_file(
    doc_id: str,
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    collection: Optional[str] = None,
) -> dict[str, Any]:
    """Chunk + embed toàn bộ text và insert vào Milvus.

    Trả về dict gồm doc_id, collection, chunk_count, inserted_ids.
    """
    raw_chunks = _split_text(text, chunk_size=chunk_size, overlap=overlap)
    if not raw_chunks:
        return {"doc_id": doc_id, "collection": collection or os.getenv("MILVUS_TEXT_COLLECTION", "ihos_documents"), "chunk_count": 0, "inserted_ids": []}
    from services.embedding.gemini_embedder import get_embeddings  # local import tránh vòng lặp

    embeddings = get_embeddings(raw_chunks)
    if not embeddings or len(embeddings) != len(raw_chunks):
        raise RuntimeError("Embedding lỗi: số vector không khớp số chunk")
    dim = len(embeddings[0])
    coll_name = ensure_text_collection(name=collection, dim=dim)
    # build records
    records = []
    for idx, (chunk, emb) in enumerate(zip(raw_chunks, embeddings)):
        records.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk,
            "embedding": emb,
        })
    ids = insert_embeddings(records, collection=coll_name)
    # tạo index (idempotent) & load
    try:
        create_index(collection=coll_name)
    except Exception:
        pass
    try:
        load_collection(collection=coll_name)
    except Exception:
        pass
    return {
        "doc_id": doc_id,
        "collection": coll_name,
        "chunk_count": len(records),
        "inserted_ids": ids,
    }
