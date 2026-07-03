"""UI Streamlit để test pipeline AI-SRE trực quan.

Chạy:
    streamlit run streamlit_app.py

Luồng bám GIAI ĐOẠN 2 của sơ đồ: log thô → parse → alert.json → hybrid retrieve
→ cổng "có incident KHỚP?" → chẩn đoán (known / unknown).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ai_sre.embeddings import get_embedder
from ai_sre.parse_log import parse_log
from ai_sre.retrieve import retrieve
from ai_sre.store import VectorStore
from ai_sre import reason

EXAMPLES_DIR = Path("examples")
INDEX_DIR = "index"

st.set_page_config(page_title="AI-SRE · Chẩn đoán sự cố", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Đang nạp index + embedder…")
def load_store_and_embedder(index_dir: str):
    """Nạp 1 lần, cache lại giữa các lần chạy (tránh load model mỗi lần bấm)."""
    store = VectorStore.load(index_dir)
    embedder = get_embedder()
    return store, embedder


def example_logs() -> dict[str, str]:
    if not EXAMPLES_DIR.exists():
        return {}
    return {p.name: p.read_text(encoding="utf-8").strip() for p in sorted(EXAMPLES_DIR.glob("*.txt"))}


# --- Sidebar: tham số ---------------------------------------------------------
st.sidebar.header("⚙️ Tham số")
provider = os.getenv("EMBEDDING_PROVIDER", "hash")
st.sidebar.caption(f"Embedding provider: **{provider}**")
k = st.sidebar.slider("Top-K tài liệu", min_value=3, max_value=20, value=8)
threshold = st.sidebar.number_input(
    "Ngưỡng khớp (match threshold)",
    min_value=0.0,
    max_value=0.2,
    value=float(reason.MATCH_SCORE_THRESHOLD),
    step=0.001,
    format="%.3f",
)
call_model = st.sidebar.checkbox("Gọi model chẩn đoán (OpenAI)", value=True)
st.sidebar.caption(f"Model: **{reason.MODEL_DIAGNOSE}** · cần OPENAI_API_KEY")

# --- Main: nhập log -----------------------------------------------------------
st.title("🩺 AI-SRE — Chẩn đoán sự cố từ log")

examples = example_logs()
default_text = examples.get("sample_log.txt", "")

col_a, col_b = st.columns([3, 1])
with col_b:
    picked = st.selectbox("Nạp ví dụ", ["(tự nhập)"] + list(examples.keys()))
    if picked != "(tự nhập)":
        default_text = examples[picked]

    uploaded = st.file_uploader("📤 Upload file log", type=["txt", "log"])
    if uploaded is not None:
        # File upload ưu tiên hơn ví dụ đã chọn.
        default_text = uploaded.getvalue().decode("utf-8", errors="replace").strip()

with col_a:
    raw = st.text_area("📥 Log thô (input)", value=default_text, height=120,
                       placeholder="2026-02-15 ERROR [payment-service] OOMKilled: memory limit exceeded")

run = st.button("🔍 Chẩn đoán", type="primary")

# --- Pipeline -----------------------------------------------------------------
if run:
    if not raw.strip():
        st.warning("Hãy nhập log hoặc chọn một ví dụ.")
        st.stop()

    # 1) PARSE LOG → ALERT (+ cổng parse được field?)
    alert = parse_log(raw)
    if alert is None:
        st.error("⛔ KẾT THÚC — không parse được service/triệu chứng từ log. Input không hợp lệ.")
        st.stop()

    st.subheader("⚠️ alert.json (đã parse)")
    st.json(alert)

    # 2) HYBRID RETRIEVE
    store, embedder = load_store_and_embedder(INDEX_DIR)
    docs = retrieve(
        store,
        embedder,
        query=alert["symptom"],
        service=alert.get("service"),
        window_start=alert.get("window_start"),
        k=k,
    )

    st.subheader(f"🔎 Hybrid retrieve — {len(docs)} tài liệu")
    if docs:
        st.dataframe(
            [
                {
                    "source_type": d["source_type"],
                    "ref": d.get("incident_id") or d.get("source_url") or d["id"],
                    "service": d.get("service", ""),
                    "score": round(d["score"], 4),
                }
                for d in docs
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Không có tài liệu nào qua được filter (service/time).")

    # 3) Cổng "có incident KHỚP?"
    matched = reason.match_known_incident(docs, threshold=threshold)
    known = matched is not None
    if known:
        ref = matched.get("incident_id") or matched["id"]
        st.success(f"✅ CÓ incident KHỚP: **{ref}** (score={matched['score']:.3f}) → nhánh *sự cố đã có trong lịch sử*")
    else:
        st.warning("❔ KHÔNG có incident khớp → nhánh *chẩn đoán tổng quát* (model tự sinh checklist)")

    # 4) Chẩn đoán 2 nhánh
    st.subheader("🧠 Chẩn đoán")
    if not call_model:
        st.info("Đã tắt gọi model. Bật checkbox ở sidebar để chạy phần chẩn đoán.")
    else:
        alert_text = reason.format_alert(alert)
        try:
            with st.spinner("Đang gọi model…"):
                resp = reason.diagnose(alert_text, docs, known=known)
            st.markdown(resp.choices[0].message.content)
            u = resp.usage
            st.caption(f"[usage] prompt={u.prompt_tokens} · completion={u.completion_tokens} · total={u.total_tokens}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Bỏ qua reasoning — kiểm tra OPENAI_API_KEY.\n\n{type(e).__name__}: {e}")

    st.divider()
    st.caption("👤 Dev duyệt → xử lý.")
