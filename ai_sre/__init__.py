"""AI-SRE RAG over telemetry & incident history.

Layer 3 (grounding): nhúng logs/metrics/traces/runbooks/postmortems/deploy vào
vector store, rồi lúc có sự cố truy hồi tài liệu liên quan và cho Claude suy luận.
"""

__version__ = "0.1.0"
