"""CLI: ingest dữ liệu và diagnose sự cố.

    python -m ai_sre.cli ingest
    python -m ai_sre.cli diagnose --log "2026-02-15 ERROR [payment-service] OOMKilled: memory limit exceeded"
    python -m ai_sre.cli diagnose --log-file examples/sample_log.txt
    python -m ai_sre.cli diagnose --alert examples/sample_alert.json   # alert đã parse sẵn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Console Windows mặc định cp1252 -> ép UTF-8 để in được tiếng Việt + ký hiệu.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

load_dotenv()

from .embeddings import get_embedder  # noqa: E402
from .ingest import build_index  # noqa: E402
from .parse_log import parse_log  # noqa: E402
from .retrieve import retrieve  # noqa: E402
from .store import VectorStore  # noqa: E402
from . import reason  # noqa: E402


def cmd_ingest(a) -> None:
    n = build_index(a.data, a.index)
    print(f"✓ Đã index {n} chunk → {a.index}/")


def _load_alert(a) -> dict | None:
    """GIAI ĐOẠN 2 · PARSE LOG → ALERT (kèm cổng 'Parse được các field?').

    Trả alert dict, hoặc None nếu là log thô không parse được (nhánh ⛔ dừng).
    """
    if a.alert:  # alert đã parse sẵn (bỏ qua bước parse log)
        return json.loads(Path(a.alert).read_text(encoding="utf-8"))

    raw = Path(a.log_file).read_text(encoding="utf-8") if a.log_file else a.log
    print("=== 📥 Log thô (input) ===")
    print(raw.strip())

    alert = parse_log(raw)
    if alert is None:
        return None

    print("\n=== ⚠️ alert.json (đã parse) ===")
    print(json.dumps(alert, ensure_ascii=False, indent=2))
    if a.emit_alert:
        Path(a.emit_alert).write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ Đã ghi alert.json → {a.emit_alert}")
    return alert


def cmd_diagnose(a) -> None:
    if not (a.log or a.log_file or a.alert):
        raise SystemExit("Cần một trong: --log, --log-file, hoặc --alert")

    alert = _load_alert(a)
    if alert is None:
        # Cổng 'Parse được các field?' = KHÔNG → ⛔ KẾT THÚC
        print("\n⛔ KẾT THÚC — không parse được service/triệu chứng từ log. Input không hợp lệ.")
        raise SystemExit(2)

    store = VectorStore.load(a.index)
    embedder = get_embedder()

    docs = retrieve(
        store,
        embedder,
        query=alert["symptom"],
        service=alert.get("service"),
        window_start=alert.get("window_start"),
        k=a.k,
    )

    print(f"\n=== HYBRID RETRIEVE — {len(docs)} tài liệu liên quan ===")
    for d in docs:
        ref = d.get("incident_id") or d.get("source_url") or d["id"]
        print(f"  - [{d['source_type']:<10}] {ref:<28} score={d['score']:.3f}")

    # Cổng 'Có incident KHỚP trong data?'
    matched = reason.match_known_incident(docs, threshold=a.match_threshold)
    known = matched is not None
    if known:
        print(f"\n✅ CÓ incident KHỚP: {matched.get('incident_id') or matched['id']} (score={matched['score']:.3f})")
    else:
        print(f"\n❔ KHÔNG có incident khớp (ngưỡng={a.match_threshold}). Nhánh chẩn đoán tổng quát.")
        print("   🔍 Khu vực cần kiểm tra: (model tự đề xuất bên dưới)")

    alert_text = reason.format_alert(alert)
    branch = "sự cố ĐÃ CÓ trong lịch sử" if known else "CHƯA CÓ trong lịch sử"
    try:
        print(f"\n=== Chẩn đoán (OpenAI · {branch}) ===")
        reason.diagnose_and_print(alert_text, docs, known=known)
    except Exception as e:  # noqa: BLE001
        print(
            "\n[Bỏ qua reasoning — kiểm tra OPENAI_API_KEY.]\n"
            f"Lỗi: {type(e).__name__}: {e}"
        )

    print("\n👤 Dev duyệt → xử lý.")


def main() -> None:
    p = argparse.ArgumentParser(prog="ai_sre", description="AI-SRE RAG over telemetry & incidents")
    sub = p.add_subparsers(required=True)

    pi = sub.add_parser("ingest", help="Dựng index từ data/")
    pi.add_argument("--data", default="data")
    pi.add_argument("--index", default="index")
    pi.set_defaults(func=cmd_ingest)

    pd = sub.add_parser("diagnose", help="Chẩn đoán từ log thô hoặc alert đã parse")
    src = pd.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", help="Đoạn log thô (1 chuỗi text)")
    src.add_argument("--log-file", help="File chứa log thô")
    src.add_argument("--alert", help="File alert.json đã parse sẵn")
    pd.add_argument("--emit-alert", help="Ghi alert.json đã parse ra file này")
    pd.add_argument("--index", default="index")
    pd.add_argument("--k", type=int, default=8)
    pd.add_argument(
        "--match-threshold",
        type=float,
        default=reason.MATCH_SCORE_THRESHOLD,
        help="Ngưỡng score để coi là incident KHỚP (mặc định từ MATCH_SCORE_THRESHOLD)",
    )
    pd.set_defaults(func=cmd_diagnose)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
