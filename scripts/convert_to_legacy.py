"""Chuyển data schema MỚI → structure CŨ mà pipeline RAG đang đọc.

Đọc (raw, do người dùng upload):
    data/incidents.jsonl, logs.jsonl, metrics.jsonl,
    data/deployments.jsonl, runbooks.json, services.json

Sinh ra (structure cũ):
    data/postmortems/<incident_id>.md   (frontmatter + ## Symptom/Root Cause/Resolution)
    data/runbooks/<id>_<slug>.md        (frontmatter + ## section)
    data/deploys.jsonl                  (đổi tên field: time->deployed_at, deploymentId->url)
    data/telemetry.jsonl                (join log + metric của mỗi incident -> 1 dòng textify)

Chạy:  python scripts/convert_to_legacy.py
Rồi:   python -m ai_sre.cli ingest
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

DATA = Path("data")


def load_jsonl(name):
    p = DATA / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main():
    incidents = load_jsonl("incidents.jsonl")
    logs = {r["incidentId"]: r for r in load_jsonl("logs.jsonl")}
    metrics = {r["incidentId"]: r for r in load_jsonl("metrics.jsonl")}
    deployments = load_jsonl("deployments.jsonl")
    runbooks = load_json("runbooks.json")
    services = {s["service"]: s for s in load_json("services.json")}
    rb_by_title = {rb.get("title", ""): rb for rb in runbooks}

    # --- postmortems/ (mỗi incident 1 file) ---
    pm_dir = DATA / "postmortems"
    if pm_dir.exists():
        shutil.rmtree(pm_dir)
    pm_dir.mkdir(parents=True)

    for inc in incidents:
        inc_id = inc["id"]
        service = inc.get("service", "")
        tech = services.get(service, {})
        tech_str = f" ({tech.get('language', '')}/{tech.get('framework', '')})" if tech else ""
        symptoms = ", ".join(inc.get("symptoms", []))

        log = logs.get(inc_id)
        log_line = f"Log ({log['level']}): {log['message']}" if log else "Log: (không có)"
        m = metrics.get(inc_id)
        metric_line = (
            f"Metrics: cpu={m['cpu']}% memory={m['memory']}% "
            f"latency={m['latency_ms']}ms error_rate={m['error_rate']}%"
            if m else "Metrics: (không có)"
        )

        rb = rb_by_title.get(inc.get("runbook", ""))
        if rb:
            steps = "; ".join(f"{i}. {s}" for i, s in enumerate(rb.get("steps", []), 1))
            resolution = f"Theo runbook {rb.get('id', '')} ({rb.get('title', '')}): {steps}"
        else:
            resolution = "(chưa có runbook khớp)"

        md = (
            f"---\n"
            f"incident_id: {inc_id}\n"
            f"service: {service}\n"
            f"environment: prod\n"
            f"severity: {inc.get('severity', '')}\n"
            f"date: {inc.get('startTime', '')}\n"
            f"url: {inc_id}\n"
            f"---\n\n"
            f"## Symptom\n"
            f"Service {service}{tech_str}. Triệu chứng: {symptoms}.\n"
            f"{log_line}\n{metric_line}\n\n"
            f"## Root Cause\n{inc.get('rootCause', '')}\n\n"
            f"## Resolution\n{resolution}\n"
        )
        (pm_dir / f"{inc_id}.md").write_text(md, encoding="utf-8")

    # --- runbooks/ ---
    rb_dir = DATA / "runbooks"
    if rb_dir.exists():
        shutil.rmtree(rb_dir)
    rb_dir.mkdir(parents=True)

    for rb in runbooks:
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(rb.get("steps", []), 1))
        md = (
            f"---\n"
            f"title: {rb.get('title', '')}\n"
            f"url: {rb.get('id', '')}\n"
            f"---\n\n"
            f"## Nhận biết\nNguyên nhân gốc: {rb.get('rootCause', '')}\n\n"
            f"## Khắc phục\n{steps}\n"
        )
        (rb_dir / f"{rb.get('id', 'RB')}_{slug(rb.get('title', ''))}.md").write_text(md, encoding="utf-8")

    # --- deploys.jsonl ---
    with (DATA / "deploys.jsonl").open("w", encoding="utf-8") as f:
        for d in deployments:
            row = {
                "service": d.get("service", ""),
                "version": d.get("version", ""),
                "deployed_at": d.get("time", ""),
                "environment": "prod",
                "url": d.get("deploymentId", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- telemetry.jsonl (join log + metric mỗi incident) ---
    with (DATA / "telemetry.jsonl").open("w", encoding="utf-8") as f:
        for inc in incidents:
            inc_id = inc["id"]
            log = logs.get(inc_id)
            m = metrics.get(inc_id)
            parts = []
            if log:
                parts.append(f"{log['level']}: {log['message']}")
            if m:
                parts.append(
                    f"cpu={m['cpu']}% mem={m['memory']}% "
                    f"latency={m['latency_ms']}ms error_rate={m['error_rate']}%"
                )
            row = {
                "source_type": "metric",
                "service": inc.get("service", ""),
                "severity": (log or {}).get("level", ""),
                "timestamp": inc.get("startTime", ""),
                "content": f"[{inc.get('service', '')}] " + " | ".join(parts),
                "source_url": inc_id,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✓ postmortems: {len(incidents)} file")
    print(f"✓ runbooks: {len(runbooks)} file")
    print(f"✓ deploys.jsonl: {len(deployments)} dòng")
    print(f"✓ telemetry.jsonl: {len(incidents)} dòng")


if __name__ == "__main__":
    main()
