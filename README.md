# AI-SRE — RAG over Telemetry & Incident History

Grounding một LLM (OpenAI) vào hệ thống của bạn: logs, metrics, traces, runbooks,
deploy metadata và postmortems được **embed** và lưu trong **vector store**. Lúc có
sự cố, agent **truy hồi** các sự cố cũ + bước runbook liên quan nhất rồi **suy luận**
trên đó — biến lịch sử rải rác thành trí nhớ tổ chức truy vấn được.

Đây là **layer 3 (grounding)** đang làm việc thật: không có nó, model chỉ biết văn bản
chung chung; có nó, model biết về chính các sự cố của bạn.

Pipeline gồm **2 giai đoạn** (xem `docs/ai-sre-pipeline.drawio`):

```
GIAI ĐOẠN 1 · INGEST (chạy 1 lần / khi data đổi)
data/ ──► chunking ──► embeddings ──► FAISS + BM25 ──► index/
(postmortem, runbook,                (vector store)    (faiss + meta)
 deploy, telemetry)

GIAI ĐOẠN 2 · DIAGNOSE (mỗi lần có sự cố)
📥 log thô ──► parse log → alert.json ──► hybrid retrieve ──► [Có incident KHỚP?]
              (regex/LLM; parse fail → ⛔ dừng)   (RRF + filter)      │
                                                        ┌─────────────┴─────────────┐
                                                     CÓ │                            │ KHÔNG
                                          kết luận cụ thể                 mô tả tổng quát +
                                       (INC-xxxx + fix + [n])          🔍 checklist khu vực
                                                        └─────────────┬─────────────┘
                                                                 👤 Dev duyệt → xử lý
```

## Cấu trúc

| File | Vai trò |
|------|---------|
| `ai_sre/chunking.py`   | Normalize + chunk từng nguồn (postmortem theo section, runbook theo H2, …) |
| `ai_sre/embeddings.py` | Embedding provider cắm-rút: `hash` \| `openai` \| `sentence-transformers` \| `voyage` |
| `ai_sre/store.py`      | FAISS (cosine) + BM25 — vector store in-process |
| `ai_sre/parse_log.py`  | **Parse log thô → alert.json** (service/symptom/timestamp; regex, fallback LLM); cổng "parse được field?" |
| `ai_sre/retrieve.py`   | Hybrid search: vector + BM25 hợp nhất bằng RRF, filter service/time, recency boost |
| `ai_sre/reason.py`     | Cổng "có incident KHỚP?" + 2 nhánh diagnose (known/unknown) qua OpenAI, trích dẫn nhãn [n] |
| `ai_sre/ingest.py`     | Dựng index từ `data/` |
| `ai_sre/cli.py`        | `ingest` / `diagnose` (`--log` \| `--log-file` \| `--alert`) |

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # điền OPENAI_API_KEY nếu muốn chạy phần reasoning
```

> **Windows (PowerShell)** — xem mục [Chạy trên Windows](#chạy-trên-windows-powershell) bên dưới cho từng lệnh tương đương (`Copy-Item` thay `cp`, activate venv, chạy LLM local bằng Ollama…).

**Embedding mặc định `sentence-transformers`** — chạy **local, MIỄN PHÍ** (không gọi API),
chỉ tốn chi phí ở bước reasoning (OpenAI). Cài: `pip install sentence-transformers`
(lần đầu tự tải model ~130MB). Data lẫn tiếng Việt → cân nhắc model đa ngôn ngữ qua
`ST_MODEL` (vd `BAAI/bge-m3` hoặc `paraphrase-multilingual-MiniLM-L12-v2`).

Cần chạy nhanh không cài gì: đặt `EMBEDDING_PROVIDER=hash` (offline nhưng ngữ nghĩa thấp).
**Đổi provider phải chạy lại `ingest`** (vector khác embedder không tương thích).

## Chạy

```bash
# 1) Dựng index từ dữ liệu mẫu (GIAI ĐOẠN 1)
python -m ai_sre.cli ingest

# 2) Chẩn đoán từ LOG THÔ (GIAI ĐOẠN 2 — đầu vào theo sơ đồ)
python -m ai_sre.cli diagnose --log "2026-02-15 ERROR [payment-service] OOMKilled: memory limit exceeded"

# nhánh CÓ — sự cố đã có trong lịch sử (payment-service OOMKilled → khớp postmortem)
python -m ai_sre.cli diagnose --log-file examples/sample_log.txt

# nhánh KHÔNG — chưa có trong lịch sử (search-service TLS cert → model tự sinh checklist)
python -m ai_sre.cli diagnose --log-file examples/sample_log_unknown.txt

# hoặc dùng alert.json đã parse sẵn (bỏ qua bước parse log)
python -m ai_sre.cli diagnose --alert examples/sample_alert.json
```

### UI Streamlit (test trực quan)

```bash
streamlit run streamlit_app.py
```

Dán log thô hoặc chọn một file trong `examples/` từ dropdown → xem `alert.json`, bảng
tài liệu retrieve, cổng "có incident KHỚP?", và phần chẩn đoán (**stream** — chữ hiện dần
theo thời gian thực). Sidebar chỉnh Top-K, ngưỡng khớp, bật/tắt gọi model, và **toggle
Backend LLM**: `Local (Ollama)` ↔ `Cloud (OpenAI)` + ô nhập tên model — đổi ngay không cần
khởi động lại app. Bộ ví dụ sẵn có:

| File | Nhánh |
|------|-------|
| `sample_log.txt`, `log_order_crashloop.txt`, `log_kv_format.txt` | ✅ KNOWN (khớp postmortem) |
| `sample_log_unknown.txt`, `log_unknown_disk.txt` | ❔ UNKNOWN (model tự sinh checklist) |
| `log_invalid.txt` | ⛔ PARSE-FAIL (dừng) |

Luồng `diagnose`: **parse log → alert.json** (không parse được service/triệu chứng →
⛔ dừng, "input không hợp lệ") → **hybrid retrieve** → **cổng "có incident KHỚP?"**:
- **CÓ** (postmortem đạt ngưỡng `--match-threshold` / `MATCH_SCORE_THRESHOLD`): kết luận
  cụ thể — INC-xxxx + nguyên nhân + bước khắc phục, kèm trích dẫn nhãn [n].
- **KHÔNG**: mô tả tổng quát + in 🔍 **checklist khu vực cần kiểm tra**.

Không có `OPENAI_API_KEY`: bước retrieval + quyết định khớp + checklist vẫn in ra; chỉ
bỏ qua phần gọi model. Muốn parse log bằng LLM khi regex thất bại: đặt `LOG_PARSE_LLM_FALLBACK=1`.

## Chạy trên Windows (PowerShell)

Toàn bộ bước tương đương, chỉ khác cú pháp shell.

```powershell
# 1) Tạo & kích hoạt venv
python -m venv .venv
.venv\Scripts\Activate.ps1        # nếu bị chặn: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt

# 2) Tạo file .env (điền key nếu chạy Cloud)
Copy-Item .env.example .env

# 3) Dựng index + chẩn đoán (giống bản *nix)
python -m ai_sre.cli ingest
python -m ai_sre.cli diagnose --log-file examples/sample_log.txt

# 4) UI
streamlit run streamlit_app.py
```

### Chạy LLM local bằng Ollama (không cần OpenAI key)

Reasoning trỏ sang endpoint **OpenAI-compatible** của Ollama, không phải sửa code.

```powershell
# 1) Cài Ollama (https://ollama.com/download) rồi kéo model text
ollama pull qwen2.5:32b          # xem model đã có: ollama list
ollama serve                     # thường đã tự chạy nền sau khi cài

# 2a) Cách nhanh: bật app rồi chọn ở sidebar
#     Backend LLM = "Local (Ollama)", ô Model = qwen2.5:32b
streamlit run streamlit_app.py

# 2b) Hoặc cố định qua .env (áp cho cả CLI lẫn UI)
```

`.env` cho chế độ local:

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:32b
OPENAI_API_KEY=ollama            # local không cần key thật
```

Ghi chú:
- Chỉ dùng model **text** (`qwen2.5:32b`, `qwen2.5:14b`, `llama3.1`…) cho chẩn đoán; model `*vl` (vision) không phù hợp.
- Streamlit phải chạy **cùng máy với Ollama** (endpoint `localhost`). Deploy Streamlit Cloud thì `localhost` không tới máy bạn → local model chỉ dùng khi chạy `streamlit run` tại chỗ.
- Không set `OPENAI_BASE_URL` → mặc định dùng OpenAI cloud (`gpt-4o`) như cũ.
- Embedding local (miễn phí, offline): `EMBEDDING_PROVIDER=sentence-transformers` — độc lập với backend reasoning ở trên.

## Kịch bản mẫu

`sample_alert.json` mô phỏng `payment-service` bị **OOMKilled** (hết bộ nhớ, restart liên tục).
Retrieval kéo về các postmortem `payment-service` cùng triệu chứng OOMKilled, mỗi cái đã kèm
bước khắc phục từ runbook `RB-004` (Increase memory / Profile heap) — đủ để model đề xuất
nguyên nhân (memory leak) + cách xử lý, có trích dẫn nguồn theo nhãn [n].

## Từ PoC lên production

- **Vector DB**: đổi FAISS → **Weaviate** (metadata filter + hybrid native, scale). Giữ nguyên interface `store`.
- **Embeddings**: OpenAI (`text-embedding-3-small/large`) hoặc bge self-hosted cho dữ liệu nhạy cảm.
- **Ingestion thật**: cắm log-templating (Drain3), anomaly→text cho metrics, exporter trace; giữ telemetry theo cửa sổ trượt, postmortem/runbook giữ lâu.
- **An toàn**: scrub PII/secrets trước embed; agent chỉ *đề xuất*, con người duyệt trước khi chạy lệnh; đo retrieval hit-rate trên tập sự cố lịch sử.

> Model dùng cho reasoning: **OpenAI** qua Chat Completions (mặc định `gpt-4o`, đổi bằng `OPENAI_MODEL`),
> hoặc **LLM local** qua endpoint OpenAI-compatible (Ollama/LM Studio/vLLM) bằng `OPENAI_BASE_URL`
> — xem [Chạy LLM local bằng Ollama](#chạy-llm-local-bằng-ollama-không-cần-openai-key). Đáp án được
> **stream** (render dần). Trích dẫn nguồn thực hiện qua prompt (nhãn [n]).
