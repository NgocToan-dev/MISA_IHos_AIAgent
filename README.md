## Base LangGraph Agent Skeleton

Thư mục khởi tạo kiến trúc gồm:

```
api/        # FastAPI app expose endpoint gọi inference
inference/  # Logic chạy agent (hàm tiện ích để invoke graph)
agent/      # Xây dựng LangGraph (state graph, nodes, compile)
state/      # Định nghĩa State dùng xuyên suốt các node
tools/      # Các tool (ví dụ: máy tính, echo) -> có thể mở rộng
model/      # Model wrapper (LLM / local / mock)
tests/      # (tùy chọn) test sau này
requirements.txt
```

### Luồng
FastAPI `/invoke` -> `inference/run.py:invoke_agent` -> build/khởi tạo graph (lazy singleton) -> truyền `query` vào state -> graph chạy qua các node (router -> tool/model) -> trả JSON kết quả + trace.

### Chạy nhanh
```bash
pip install -r requirements.txt
python server.py --port 8000 --reload
```

Gửi thử:
```bash
curl -X POST http://localhost:8000/invoke -H "Content-Type: application/json" -d '{"query":"calc: 2+3*4"}'
```

Hoặc:
```bash
curl -X POST http://localhost:8000/invoke -H "Content-Type: application/json" -d '{"query":"hello world"}'
```

### Mở rộng
- Thêm tool: tạo file trong `tools/`, implement hàm `run_tool(state)` hoặc class, đăng ký ở `agent/registry.py`.
- Thêm model thật (OpenAI, v.v.): thay `model/local_llm.py` bằng wrapper tương ứng.

### Ghi chú
- Code chú thích tiếng Việt để dễ chỉnh sửa.
- State đơn giản, có thể đổi sang Pydantic nếu cần validation mạnh hơn.
