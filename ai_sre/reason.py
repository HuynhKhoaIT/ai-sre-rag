"""Reasoning layer — grounding OpenAI vào tài liệu đã retrieve.

diagnose(): gọi OpenAI Chat Completions, nhét tài liệu retrieved vào context và
yêu cầu model trích dẫn nguồn theo nhãn [n] (OpenAI không có citations native như
Claude nên ta trích dẫn qua prompt). OpenAI tự cache prefix (không cần tham số).

Theo sơ đồ GIAI ĐOẠN 2, sau HYBRID RETRIEVE có cổng quyết định
"Có incident KHỚP trong data?" → hai nhánh chẩn đoán:
  - CÓ  (known)   : sự cố đã có trong lịch sử → chỉ rõ INC-xxxx, nguyên nhân đã
                    xác định, bước khắc phục cụ thể (trích postmortem/runbook).
  - KHÔNG (unknown): chưa có trong lịch sử → mô tả tổng quát + liệt kê KHU VỰC
                    CẦN KIỂM TRA (checklist).
"""

from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

MODEL_DIAGNOSE = os.getenv("OPENAI_MODEL", "gpt-4o")

# Ngưỡng điểm để coi một postmortem là "incident KHỚP".
# Điểm là RRF (+ recency), khá nhỏ: xuất hiện rank-0 ở 1 list ≈ 1/60 ≈ 0.017,
# ở CẢ vector + BM25 thì cao hơn. 0.015 = "cùng service + cùng triệu chứng" là khớp.
MATCH_SCORE_THRESHOLD = float(os.getenv("MATCH_SCORE_THRESHOLD", "0.015"))

# Nhánh CÓ — sự cố đã có trong lịch sử.
SYSTEM_DIAGNOSE_KNOWN = (
    "Bạn là AI-SRE hỗ trợ kỹ sư trực xử lý sự cố production. "
    "Các tài liệu cung cấp CÓ chứa một sự cố lịch sử KHỚP với triệu chứng hiện tại. "
    "CHỈ suy luận dựa trên bằng chứng trong tài liệu (postmortem, runbook, deploy, telemetry). "
    "Luôn TRÍCH DẪN nguồn theo nhãn [n] cho mỗi kết luận. "
    "TUYỆT ĐỐI không bịa bước khắc phục. Bạn chỉ ĐỀ XUẤT; con người quyết định thực thi.\n\n"
    "Trả lời theo cấu trúc:\n"
    "1. Sự cố khớp: nêu rõ mã incident (INC-xxxx) và vì sao khớp\n"
    "2. Nguyên nhân đã xác định (trích từ postmortem)\n"
    "3. Bước khắc phục cụ thể (ưu tiên bước có trong runbook)\n"
    "4. Bằng chứng & nguồn tham chiếu (theo nhãn [n])"
)

# Nhánh KHÔNG — chưa có trong lịch sử. Model TỰ sinh checklist "khu vực cần kiểm tra"
# dựa trên triệu chứng + bằng chứng (KHÔNG dùng list tĩnh).
SYSTEM_DIAGNOSE_UNKNOWN = (
    "Bạn là AI-SRE hỗ trợ kỹ sư trực xử lý sự cố production. "
    "Các tài liệu cung cấp KHÔNG chứa sự cố lịch sử nào khớp rõ với triệu chứng hiện tại. "
    "Không được bịa ra một sự cố lịch sử. Thay vào đó: mô tả tổng quát vấn đề dựa trên "
    "triệu chứng + bất kỳ manh mối nào trong tài liệu (deploy gần đây, telemetry), rồi "
    "TỰ suy ra và liệt kê các KHU VỰC CẦN KIỂM TRA phù hợp với chính triệu chứng này "
    "(vd: kết nối database, auth/token, logic code/hồi quy, phụ thuộc ngoài, config/biến "
    "môi trường, deploy gần đây… — nhưng hãy chọn lọc và ưu tiên theo bằng chứng, đừng "
    "liệt kê máy móc). "
    "Trích dẫn nguồn theo nhãn [n] khi có. Bạn chỉ ĐỀ XUẤT; con người quyết định.\n\n"
    "Trả lời theo cấu trúc:\n"
    "1. Mô tả tổng quát vấn đề (kèm độ tin cậy: cao/trung bình/thấp)\n"
    "2. Khu vực cần kiểm tra (checklist ưu tiên, giải thích ngắn vì sao)\n"
    "3. Bằng chứng liên quan (theo nhãn [n], nếu có)"
)

_INSTRUCTION = (
    "\n\nDựa trên các tài liệu ở trên, hãy chẩn đoán sự cố. "
    "Trích dẫn tài liệu cụ thể theo nhãn [n] cho từng nhận định."
)


def match_known_incident(docs: List[dict], threshold: float = MATCH_SCORE_THRESHOLD) -> Optional[dict]:
    """Cổng 'Có incident KHỚP trong data?' — trả postmortem khớp nhất nếu đạt ngưỡng.

    Khớp = có chunk postmortem (cùng triệu chứng, đã qua filter service/time ở retrieve)
    với score ≥ ngưỡng. None = nhánh KHÔNG (chưa có trong lịch sử).
    """
    best = None
    for d in docs:
        if d.get("source_type") == "postmortem" and d.get("score", 0.0) >= threshold:
            if best is None or d["score"] > best["score"]:
                best = d
    return best


def _client() -> OpenAI:
    # Local (Ollama/LM Studio/vLLM…) qua endpoint OpenAI-compatible: set OPENAI_BASE_URL.
    # Ollama: OPENAI_BASE_URL=http://localhost:11434/v1 (api_key tùy ý, không cần thật).
    # Không set OPENAI_BASE_URL → dùng OpenAI cloud như cũ (đọc OPENAI_API_KEY từ môi trường).
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "ollama"))
    return OpenAI()  # đọc OPENAI_API_KEY từ môi trường


def format_alert(alert: dict) -> str:
    lines = [
        "## Alert hiện tại",
        f"- Service: {alert.get('service', '?')}",
        f"- Môi trường: {alert.get('environment', 'prod')}",
        f"- Thời điểm: {alert.get('fired_at', '?')}",
        f"- Triệu chứng: {alert.get('symptom', '')}",
    ]
    if alert.get("context"):
        lines.append(f"- Ngữ cảnh thêm: {alert['context']}")
    return "\n".join(lines)


def _context_from_docs(docs: List[dict]) -> str:
    """Ghép tài liệu retrieved thành context có nhãn [n] để model trích dẫn."""
    lines = []
    for i, d in enumerate(docs, 1):
        title = d.get("incident_id") or d.get("source_url") or d["id"]
        lines.append(f"[{i}] ({d['source_type']}:{title})\n{d['content']}")
    return "\n\n".join(lines)


def diagnose(alert_text: str, docs: List[dict], known: bool = True):
    system = SYSTEM_DIAGNOSE_KNOWN if known else SYSTEM_DIAGNOSE_UNKNOWN
    user = f"## Tài liệu liên quan\n{_context_from_docs(docs)}\n\n{alert_text}{_INSTRUCTION}"
    return _client().chat.completions.create(
        model=MODEL_DIAGNOSE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


def diagnose_stream(alert_text: str, docs: List[dict], known: bool = True, usage_sink: Optional[dict] = None):
    """Như diagnose() nhưng STREAM: yield từng đoạn text ngay khi model sinh ra.

    Dùng cho UI để render dần, không phải đợi hết. Nếu truyền usage_sink (dict),
    token usage (khi server trả về ở chunk cuối) được ghi vào usage_sink['usage'].
    """
    system = SYSTEM_DIAGNOSE_KNOWN if known else SYSTEM_DIAGNOSE_UNKNOWN
    user = f"## Tài liệu liên quan\n{_context_from_docs(docs)}\n\n{alert_text}{_INSTRUCTION}"
    stream = _client().chat.completions.create(
        model=MODEL_DIAGNOSE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        if usage_sink is not None and getattr(chunk, "usage", None):
            usage_sink["usage"] = chunk.usage


def diagnose_and_print(alert_text: str, docs: List[dict], known: bool = True) -> None:
    resp = diagnose(alert_text, docs, known=known)
    print(resp.choices[0].message.content)
    u = resp.usage
    print(f"\n[usage] prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
