from __future__ import annotations

from typing import List, Dict, Any, Optional
import datetime

from services.mongo.mongo_repo import find_one, update_one, insert_one

CONVERSATION_COLLECTION = "conversations"


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def get_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return last `limit` messages for session_id.

    Each message is a dict with keys: role, content, ts (ISO string).
    """
    try:
        doc = find_one(CONVERSATION_COLLECTION, {"session_id": session_id})
    except Exception as e:
        # Print stack for tracing
        import traceback

        print(f"[conversation_service.get_history] error fetching session={session_id}: {e}")
        traceback.print_exc()
        return []
    if not doc:
        return []
    msgs = doc.get("messages", []) or []
    if limit is None or limit <= 0:
        return msgs
    return msgs[-limit:]


def append_messages(session_id: str, messages: List[Dict[str, Any]]) -> None:
    """Append messages to the conversation document (upsert).

    Each message should contain at least 'role' and 'content'. A timestamp 'ts' will be added
    when missing.
    """
    now = _now_iso()
    for m in messages:
        if "ts" not in m:
            m["ts"] = now
    update = {"$push": {"messages": {"$each": messages}}, "$set": {"updated_at": now, "session_id": session_id}}
    # upsert using update_one from mongo_repo
    try:
        update_one(CONVERSATION_COLLECTION, {"session_id": session_id}, update, upsert=True)
    except Exception as e:
        import traceback

        print(f"[conversation_service.append_messages] error appending for session={session_id}: {e}")
        traceback.print_exc()
        # swallow to avoid breaking main flow


def clear_history(session_id: str) -> None:
    """Remove conversation doc for session_id. (optional helper)
    """
    # We don't import delete_one to keep API minimal here; using update to unset messages
    update = {"$set": {"messages": [], "updated_at": _now_iso()}}
    update_one(CONVERSATION_COLLECTION, {"session_id": session_id}, update, upsert=True)


__all__ = ["get_history", "append_messages", "clear_history"]
