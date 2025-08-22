"""Gemini embedding helper using langchain-google-genai.

Đòi hỏi biến môi trường:
  - GOOGLE_API_KEY
  - (optional) GEMINI_EMBED_MODEL (mặc định: text-embedding-004)

Expose hàm:
  get_embeddings(texts: list[str]) -> list[list[float]]
  embed_text(text: str) -> list[float]

Sử dụng batch qua embed_documents để giảm số request.
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

_embedding_instance: GoogleGenerativeAIEmbeddings | None = None
_cached_dim: Optional[int] = None

api_key = os.getenv("GOOGLE_API_KEY") or ""
model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")


def _get_instance() -> GoogleGenerativeAIEmbeddings:
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=SecretStr(api_key)
        )
    return _embedding_instance


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed nhiều documents. Cache dimension ở lần đầu."""
    global _cached_dim
    if not texts:
        return []
    emb = _get_instance().embed_documents(texts)
    if emb and _cached_dim is None:
        _cached_dim = len(emb[0])
    return emb


def embed_text(text: str) -> List[float]:
    global _cached_dim
    vec = _get_instance().embed_query(text)
    if vec and _cached_dim is None:
        _cached_dim = len(vec)
    return vec


def get_embedding_dimension(default: int = 768) -> int:
    """Trả về dimension đã cache hoặc default nếu chưa embed lần nào."""
    return _cached_dim or default


__all__ = ["get_embeddings", "embed_text", "get_embedding_dimension"]
