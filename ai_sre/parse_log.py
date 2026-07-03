"""GIAI ĐOẠN 2 · bước 'PARSE LOG → ALERT'.

Nhận một đoạn LOG THÔ (chuỗi text terminal in ra) và tách ra các field cần thiết:
    service / symptom / timestamp
→ dựng `alert` (dict) làm đầu vào cho bước HYBRID RETRIEVE.

Chiến lược: parse bằng REGEX trước (rẻ, offline, deterministic). Nếu không rút đủ
field và bật cờ `LOG_PARSE_LLM_FALLBACK=1`, thử fallback sang LLM (OpenAI).

Cổng quyết định "Parse được các field?" (trong drawio):
    parse_log(...) trả về dict  -> OK  -> alert.json -> retrieve
    parse_log(...) trả về None  -> ⛔ dừng: input không hợp lệ
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

# 2026-02-15T10:00:00Z | 2026-02-15 10:00:00 | 2026-02-15
_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?"
)

# [payment-service] hoặc service=payment-service / svc=payment-service
_SERVICE_BRACKET_RE = re.compile(r"\[([a-z0-9][a-z0-9._-]*)\]", re.I)
_SERVICE_KV_RE = re.compile(r"\b(?:service|svc|app|component)\s*[=:]\s*([a-z0-9][a-z0-9._-]*)", re.I)

_LEVEL_RE = re.compile(r"\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", re.I)
_LOG_LEVELS = {"trace", "debug", "info", "warn", "warning", "error", "fatal", "critical"}


def _find_service(text: str):
    """Bắt service, bỏ qua bracket là log-level (vd [ERROR]). Ưu tiên bracket, rồi service=."""
    for m in _SERVICE_BRACKET_RE.finditer(text):
        if m.group(1).lower() not in _LOG_LEVELS:
            return m
    return _SERVICE_KV_RE.search(text)


def _normalize_ts(ts: str) -> str:
    """Chuẩn hóa timestamp về ISO 8601 (ngày trần -> nửa đêm UTC)."""
    if len(ts) == 10:  # chỉ có ngày
        return f"{ts}T00:00:00Z"
    return ts


def parse_log_regex(raw: str) -> Optional[dict]:
    """Rút service/symptom/timestamp bằng regex. Thiếu service HOẶC symptom -> None."""
    text = raw.strip()
    if not text:
        return None

    ts_match = _TS_RE.search(text)
    timestamp = _normalize_ts(ts_match.group(0)) if ts_match else ""

    svc_match = _find_service(text)
    service = svc_match.group(1) if svc_match else ""

    # symptom = phần thông điệp còn lại sau khi bỏ timestamp / log level / service tag.
    symptom = text
    if ts_match:
        symptom = symptom.replace(ts_match.group(0), " ", 1)
    if svc_match:
        symptom = symptom.replace(svc_match.group(0), " ", 1)
    symptom = _LEVEL_RE.sub(" ", symptom, count=1)
    symptom = re.sub(r"\s+", " ", symptom).strip(" -:[]|\t")

    if not service or not symptom:
        return None

    return {
        "service": service,
        "environment": "prod",
        "fired_at": timestamp,
        "symptom": symptom,
    }


_LLM_SYSTEM = (
    "Bạn trích thông tin từ một dòng log production. Trả về DUY NHẤT một JSON object "
    'với các khóa: "service", "symptom", "timestamp" (ISO 8601, có thể rỗng nếu không rõ). '
    "Không thêm chữ nào ngoài JSON. Nếu không xác định được service hoặc triệu chứng, "
    'trả về {"service": "", "symptom": ""}.'
)


def parse_log_llm(raw: str) -> Optional[dict]:
    """Fallback dùng LLM khi regex không đủ (chỉ chạy nếu bật cờ + có OPENAI_API_KEY)."""
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": raw.strip()},
        ],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return None

    service = (data.get("service") or "").strip()
    symptom = (data.get("symptom") or "").strip()
    if not service or not symptom:
        return None

    ts = (data.get("timestamp") or "").strip()
    return {
        "service": service,
        "environment": "prod",
        "fired_at": _normalize_ts(ts) if len(ts) == 10 else ts,
        "symptom": symptom,
    }


def parse_log(raw: str) -> Optional[dict]:
    """Cổng 'PARSE LOG → ALERT'. Trả alert dict nếu tách được, None nếu không.

    None = nhánh '⛔ KẾT THÚC — input không hợp lệ' trong sơ đồ.
    """
    alert = parse_log_regex(raw)
    if alert:
        return alert

    if os.getenv("LOG_PARSE_LLM_FALLBACK", "").lower() in ("1", "true", "yes"):
        try:
            return parse_log_llm(raw)
        except Exception:  # noqa: BLE001 — fallback lỗi thì coi như parse thất bại
            return None
    return None
