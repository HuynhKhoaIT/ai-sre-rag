"""Normalize + chunk từng nguồn dữ liệu thành các Chunk có metadata.

Nguyên tắc (structure cũ):
- Postmortem: tách theo section nghiệp vụ (symptom / root_cause / resolution / lessons),
  các chunk cùng incident_id để retrieve xong kéo theo nhau.
- Runbook: tách theo H2, prepend breadcrumb tiêu đề để không mất ngữ cảnh.
- Deploy: mỗi record -> 1 chunk (chủ yếu để filter theo service + thời gian).
- Telemetry (log/metric/trace): giả định đã được "chưng cất" thành mô tả anomaly
  dạng văn bản trước khi tới đây.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Chunk:
    content: str
    source_type: str  # postmortem | runbook | deploy | log | metric | trace
    service: str = ""
    environment: str = "prod"
    severity: str = ""
    incident_id: str = ""
    section: str = ""
    timestamp: str = ""  # ISO 8601
    source_url: str = ""

    def id(self) -> str:
        raw = f"{self.source_type}|{self.source_url}|{self.content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------

def _split_frontmatter(text: str) -> Tuple[dict, str]:
    """Tách khối frontmatter `--- ... ---` ở đầu file thành dict + phần thân."""
    if text.lstrip().startswith("---"):
        _, fm, body = text.split("---", 2)
        meta = {}
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta, body.strip()
    return {}, text


def _split_h2(body: str) -> List[Tuple[str, str]]:
    """Tách thân markdown theo heading `## `."""
    parts = re.split(r"^##\s+", body, flags=re.M)
    out: List[Tuple[str, str]] = []
    for p in parts:
        if not p.strip():
            continue
        head, _, rest = p.partition("\n")
        out.append((head.strip(), rest.strip()))
    return out


# ---------------------------------------------------------------------------

_SECTION_MAP = {
    "symptom": "symptom",
    "impact": "symptom",
    "triệu chứng": "symptom",
    "root cause": "root_cause",
    "nguyên nhân": "root_cause",
    "resolution": "resolution",
    "timeline": "resolution",
    "khắc phục": "resolution",
    "lessons": "lessons",
    "action items": "lessons",
    "bài học": "lessons",
}


def parse_postmortem(path) -> List[Chunk]:
    meta, body = _split_frontmatter(Path(path).read_text(encoding="utf-8"))
    chunks: List[Chunk] = []
    for title, sec_body in _split_h2(body):
        section = _SECTION_MAP.get(title.lower().strip(), "other")
        breadcrumb = f"[Postmortem {meta.get('incident_id', '')} · {meta.get('service', '')} · {title}]"
        chunks.append(
            Chunk(
                content=f"{breadcrumb}\n{sec_body}",
                source_type="postmortem",
                service=meta.get("service", ""),
                environment=meta.get("environment", "prod"),
                severity=meta.get("severity", ""),
                incident_id=meta.get("incident_id", ""),
                section=section,
                timestamp=meta.get("date", ""),
                source_url=meta.get("url", ""),
            )
        )
    return chunks


def parse_runbook(path) -> List[Chunk]:
    meta, body = _split_frontmatter(Path(path).read_text(encoding="utf-8"))
    title = meta.get("title", Path(path).stem)
    chunks: List[Chunk] = []
    for sec, sec_body in _split_h2(body):
        chunks.append(
            Chunk(
                content=f"[Runbook: {title} > {sec}]\n{sec_body}",
                source_type="runbook",
                service=meta.get("service", ""),
                environment=meta.get("environment", "prod"),
                section=sec.lower().strip(),
                timestamp=meta.get("date", ""),
                source_url=meta.get("url", ""),
            )
        )
    return chunks


def parse_deploys(path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        content = (
            f"Deploy {d['service']} version {d['version']} lúc {d['deployed_at']}. "
            f"Commits: {d.get('commit_range', '')}. Author: {d.get('author', '')}. "
            f"Rollback: {d.get('rollback', False)}."
        )
        chunks.append(
            Chunk(
                content=content,
                source_type="deploy",
                service=d["service"],
                environment=d.get("environment", "prod"),
                timestamp=d["deployed_at"],
                source_url=d.get("url", ""),
                section="deploy",
            )
        )
    return chunks


def parse_telemetry(path) -> List[Chunk]:
    """Anomaly đã được textify (log-template/metric->text/trace) ở tầng thu thập."""
    chunks: List[Chunk] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        chunks.append(
            Chunk(
                content=d["content"],
                source_type=d.get("source_type", "metric"),
                service=d.get("service", ""),
                environment=d.get("environment", "prod"),
                severity=d.get("severity", ""),
                timestamp=d.get("timestamp", ""),
                source_url=d.get("source_url", ""),
                section="telemetry",
            )
        )
    return chunks
