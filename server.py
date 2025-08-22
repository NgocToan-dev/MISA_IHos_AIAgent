"""Entry point chạy FastAPI app bằng python server.py.

Ví dụ:
    python server.py --port 8000 --reload

Tuỳ chọn:
    --host 0.0.0.0 (mặc định 127.0.0.1)
    --port 8000
    --reload (bật auto reload dev)
    --workers 4 (chỉ dùng khi không bật --reload)
"""
from __future__ import annotations
import argparse
import os
from dotenv import load_dotenv
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Run LangGraph Agent FastAPI server")
    parser.add_argument("--host", default=os.getenv("SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", 8000)))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--workers", type=int, default=1, help="Số worker (bỏ qua nếu --reload)")
    return parser.parse_args()


def main():
    # Load biến môi trường
    load_dotenv()
    args = parse_args()
    # Import app trễ để chắc env đã load
    from api.main import app  # noqa

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
