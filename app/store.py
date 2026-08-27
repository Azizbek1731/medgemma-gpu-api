"""SQLite persistence for uploads, inference runs, and radiologist scoring.

The scoring table is the point of the whole exercise: a run you have not graded tells
you nothing about whether the model is worth wiring into a production pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id           TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    source_name  TEXT NOT NULL,
    path         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    report_json  TEXT NOT NULL DEFAULT '{}',
    manifest_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS runs (
    id             TEXT PRIMARY KEY,
    upload_id      TEXT NOT NULL,
    study_uid      TEXT NOT NULL DEFAULT '',
    series_uid     TEXT NOT NULL DEFAULT '',
    modality       TEXT NOT NULL DEFAULT '',
    body_part      TEXT NOT NULL DEFAULT '',
    frames_json    TEXT NOT NULL DEFAULT '[]',
    window_json    TEXT NOT NULL DEFAULT '{}',
    engine         TEXT NOT NULL DEFAULT '',
    model_id       TEXT NOT NULL DEFAULT '',
    template_key   TEXT NOT NULL DEFAULT '',
    system_prompt  TEXT NOT NULL DEFAULT '',
    user_prompt    TEXT NOT NULL DEFAULT '',
    context        TEXT NOT NULL DEFAULT '',
    output         TEXT NOT NULL DEFAULT '',
    error          TEXT NOT NULL DEFAULT '',
    elapsed_ms     INTEGER NOT NULL DEFAULT 0,
    image_count    INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    batch_id       TEXT NOT NULL DEFAULT '',
    detections_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (upload_id) REFERENCES uploads(id)
);

CREATE INDEX IF NOT EXISTS idx_runs_upload ON runs(upload_id);
CREATE INDEX IF NOT EXISTS idx_runs_study ON runs(study_uid);
CREATE INDEX IF NOT EXISTS idx_runs_batch ON runs(batch_id);

-- One row per "analyse the whole study" job.
CREATE TABLE IF NOT EXISTS batches (
    id             TEXT PRIMARY KEY,
    upload_id      TEXT NOT NULL,
    study_uid      TEXT NOT NULL DEFAULT '',
    label          TEXT NOT NULL DEFAULT '',
    mode           TEXT NOT NULL DEFAULT '',
    engine         TEXT NOT NULL DEFAULT '',
    model_id       TEXT NOT NULL DEFAULT '',
    template_key   TEXT NOT NULL DEFAULT '',
    planned_jobs   INTEGER NOT NULL DEFAULT 0,
    completed_jobs INTEGER NOT NULL DEFAULT 0,
    failed_jobs    INTEGER NOT NULL DEFAULT 0,
    synthesis      TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'running',  -- running | done | cancelled | error
    error          TEXT NOT NULL DEFAULT '',
    elapsed_ms     INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (upload_id) REFERENCES uploads(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    run_id                 TEXT PRIMARY KEY,
    ground_truth           TEXT NOT NULL DEFAULT '',
    agreement              INTEGER,          -- 1..5 Likert
    missed_findings        TEXT NOT NULL DEFAULT '',
    hallucinations         TEXT NOT NULL DEFAULT '',
    clinically_significant_error INTEGER NOT NULL DEFAULT 0,
    would_change_management INTEGER NOT NULL DEFAULT 0,
    usable_as_draft        INTEGER NOT NULL DEFAULT 0,
    reference_label        TEXT NOT NULL DEFAULT '',  -- 'normal' | 'abnormal' | ''
    notes                  TEXT NOT NULL DEFAULT '',
    rated_at               TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.executescript(SCHEMA)
            _migrate(_conn)
            _conn.commit()
        return _conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
    if "batch_id" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN batch_id TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_batch ON runs(batch_id)")
    if "detections_json" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN detections_json TEXT NOT NULL DEFAULT '[]'")


def _exec(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    conn = connect()
    with _lock:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def _query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = connect()
    with _lock:
        return conn.execute(sql, params).fetchall()


# --------------------------------------------------------------------------------------
# uploads
# --------------------------------------------------------------------------------------


def create_upload(
    label: str,
    source_name: str,
    path: Path,
    report: dict,
    manifest: list,
    upload_id: str = "",
) -> str:
    """Insert an upload row. Pass ``upload_id`` to match a directory created beforehand."""
    upload_id = upload_id or new_id("up")
    _exec(
        "INSERT INTO uploads (id, label, source_name, path, created_at, report_json, manifest_json)"
        " VALUES (?,?,?,?,?,?,?)",
        (
            upload_id,
            label,
            source_name,
            str(path),
            _now(),
            json.dumps(report, ensure_ascii=False),
            json.dumps(manifest, ensure_ascii=False),
        ),
    )
    return upload_id


def get_upload(upload_id: str) -> dict | None:
    rows = _query("SELECT * FROM uploads WHERE id = ?", (upload_id,))
    if not rows:
        return None
    row = dict(rows[0])
    row["report"] = json.loads(row.pop("report_json") or "{}")
    row["manifest"] = json.loads(row.pop("manifest_json") or "[]")
    return row


def list_uploads() -> list[dict]:
    rows = _query(
        "SELECT u.id, u.label, u.source_name, u.created_at, u.report_json,"
        " (SELECT COUNT(*) FROM runs r WHERE r.upload_id = u.id) AS run_count"
        " FROM uploads u ORDER BY u.created_at DESC"
    )
    out = []
    for row in rows:
        d = dict(row)
        report = json.loads(d.pop("report_json") or "{}")
        d["dicom_files"] = report.get("dicom_files", 0)
        d["study_count"] = report.get("study_count", 0)
        out.append(d)
    return out


def delete_upload(upload_id: str) -> None:
    _exec(
        "DELETE FROM evaluations WHERE run_id IN (SELECT id FROM runs WHERE upload_id = ?)",
        (upload_id,),
    )
    _exec("DELETE FROM runs WHERE upload_id = ?", (upload_id,))
    _exec("DELETE FROM batches WHERE upload_id = ?", (upload_id,))
    _exec("DELETE FROM uploads WHERE id = ?", (upload_id,))


# --------------------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------------------


def create_run(**kwargs: Any) -> str:
    run_id = new_id("run")
    _exec(
        """INSERT INTO runs (id, upload_id, study_uid, series_uid, modality, body_part,
              frames_json, window_json, engine, model_id, template_key, system_prompt,
              user_prompt, context, output, error, elapsed_ms, image_count, created_at,
              batch_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            kwargs.get("upload_id", ""),
            kwargs.get("study_uid", ""),
            kwargs.get("series_uid", ""),
            kwargs.get("modality", ""),
            kwargs.get("body_part", ""),
            json.dumps(kwargs.get("frames", []), ensure_ascii=False),
            json.dumps(kwargs.get("window", {}), ensure_ascii=False),
            kwargs.get("engine", ""),
            kwargs.get("model_id", ""),
            kwargs.get("template_key", ""),
            kwargs.get("system_prompt", ""),
            kwargs.get("user_prompt", ""),
            kwargs.get("context", ""),
            kwargs.get("output", ""),
            kwargs.get("error", ""),
            int(kwargs.get("elapsed_ms", 0)),
            int(kwargs.get("image_count", 0)),
            _now(),
            kwargs.get("batch_id", ""),
        ),
    )
    return run_id


def finish_run(
    run_id: str,
    output: str,
    elapsed_ms: int,
    error: str = "",
    detections: list | None = None,
) -> None:
    _exec(
        "UPDATE runs SET output = ?, elapsed_ms = ?, error = ?, detections_json = ? WHERE id = ?",
        (
            output,
            int(elapsed_ms),
            error,
            json.dumps(detections or [], ensure_ascii=False),
            run_id,
        ),
    )


def get_run(run_id: str) -> dict | None:
    rows = _query(
        "SELECT r.*, e.ground_truth, e.agreement, e.missed_findings, e.hallucinations,"
        " e.clinically_significant_error, e.would_change_management, e.usable_as_draft,"
        " e.reference_label, e.notes, e.rated_at"
        " FROM runs r LEFT JOIN evaluations e ON e.run_id = r.id WHERE r.id = ?",
        (run_id,),
    )
    return _row_to_run(rows[0]) if rows else None


def list_runs(upload_id: str = "", limit: int = 500) -> list[dict]:
    sql = (
        "SELECT r.*, e.ground_truth, e.agreement, e.missed_findings, e.hallucinations,"
        " e.clinically_significant_error, e.would_change_management, e.usable_as_draft,"
        " e.reference_label, e.notes, e.rated_at"
        " FROM runs r LEFT JOIN evaluations e ON e.run_id = r.id"
    )
    params: tuple = ()
    if upload_id:
        sql += " WHERE r.upload_id = ?"
        params = (upload_id,)
    sql += " ORDER BY r.created_at DESC LIMIT ?"
    params = params + (limit,)
    return [_row_to_run(r) for r in _query(sql, params)]


def _row_to_run(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["frames"] = json.loads(d.pop("frames_json") or "[]")
    d["window"] = json.loads(d.pop("window_json") or "{}")
    d["detections"] = json.loads(d.pop("detections_json", None) or "[]")
    d["rated"] = d.get("rated_at") is not None
    return d


def delete_run(run_id: str) -> None:
    _exec("DELETE FROM evaluations WHERE run_id = ?", (run_id,))
    _exec("DELETE FROM runs WHERE id = ?", (run_id,))


# --------------------------------------------------------------------------------------
# batches (whole-study analysis)
# --------------------------------------------------------------------------------------


def create_batch(**kwargs: Any) -> str:
    batch_id = new_id("batch")
    _exec(
        """INSERT INTO batches (id, upload_id, study_uid, label, mode, engine, model_id,
              template_key, planned_jobs, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            batch_id,
            kwargs.get("upload_id", ""),
            kwargs.get("study_uid", ""),
            kwargs.get("label", ""),
            kwargs.get("mode", ""),
            kwargs.get("engine", ""),
            kwargs.get("model_id", ""),
            kwargs.get("template_key", ""),
            int(kwargs.get("planned_jobs", 0)),
            _now(),
        ),
    )
    return batch_id


def bump_batch(batch_id: str, *, completed: int = 0, failed: int = 0) -> None:
    _exec(
        "UPDATE batches SET completed_jobs = completed_jobs + ?, failed_jobs = failed_jobs + ?"
        " WHERE id = ?",
        (completed, failed, batch_id),
    )


def finish_batch(
    batch_id: str, status: str, synthesis: str = "", elapsed_ms: int = 0, error: str = ""
) -> None:
    _exec(
        "UPDATE batches SET status = ?, synthesis = ?, elapsed_ms = ?, error = ? WHERE id = ?",
        (status, synthesis, int(elapsed_ms), error, batch_id),
    )


def get_batch(batch_id: str) -> dict | None:
    rows = _query("SELECT * FROM batches WHERE id = ?", (batch_id,))
    if not rows:
        return None
    batch = dict(rows[0])
    batch["runs"] = [
        _row_to_run(r)
        for r in _query(
            "SELECT r.*, e.ground_truth, e.agreement, e.missed_findings, e.hallucinations,"
            " e.clinically_significant_error, e.would_change_management, e.usable_as_draft,"
            " e.reference_label, e.notes, e.rated_at"
            " FROM runs r LEFT JOIN evaluations e ON e.run_id = r.id"
            " WHERE r.batch_id = ? ORDER BY r.created_at",
            (batch_id,),
        )
    ]
    return batch


def list_batches(upload_id: str = "", limit: int = 100) -> list[dict]:
    sql = "SELECT * FROM batches"
    params: tuple = ()
    if upload_id:
        sql += " WHERE upload_id = ?"
        params = (upload_id,)
    sql += " ORDER BY created_at DESC LIMIT ?"
    return [dict(r) for r in _query(sql, params + (limit,))]


def delete_batch(batch_id: str) -> None:
    _exec(
        "DELETE FROM evaluations WHERE run_id IN (SELECT id FROM runs WHERE batch_id = ?)",
        (batch_id,),
    )
    _exec("DELETE FROM runs WHERE batch_id = ?", (batch_id,))
    _exec("DELETE FROM batches WHERE id = ?", (batch_id,))


# --------------------------------------------------------------------------------------
# evaluations
# --------------------------------------------------------------------------------------


def save_evaluation(run_id: str, data: dict) -> None:
    _exec(
        """INSERT INTO evaluations (run_id, ground_truth, agreement, missed_findings,
              hallucinations, clinically_significant_error, would_change_management,
              usable_as_draft, reference_label, notes, rated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(run_id) DO UPDATE SET
              ground_truth = excluded.ground_truth,
              agreement = excluded.agreement,
              missed_findings = excluded.missed_findings,
              hallucinations = excluded.hallucinations,
              clinically_significant_error = excluded.clinically_significant_error,
              would_change_management = excluded.would_change_management,
              usable_as_draft = excluded.usable_as_draft,
              reference_label = excluded.reference_label,
              notes = excluded.notes,
              rated_at = excluded.rated_at""",
        (
            run_id,
            data.get("ground_truth", ""),
            data.get("agreement"),
            data.get("missed_findings", ""),
            data.get("hallucinations", ""),
            1 if data.get("clinically_significant_error") else 0,
            1 if data.get("would_change_management") else 0,
            1 if data.get("usable_as_draft") else 0,
            data.get("reference_label", ""),
            data.get("notes", ""),
            _now(),
        ),
    )


# --------------------------------------------------------------------------------------
# metrics + export
# --------------------------------------------------------------------------------------


def _model_verdict(run: dict) -> str:
    """Pull a normal/abnormal call out of a run's output when the template forced one."""
    raw = (run.get("output") or "").strip()
    if not raw:
        return ""
    text = raw
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if "{" in p:
                text = p.lstrip("json").strip()
                break
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            v = obj.get("verdict")
            if isinstance(v, str) and v.lower() in ("normal", "abnormal"):
                return v.lower()
            if "critical_finding_present" in obj:
                return "abnormal" if obj["critical_finding_present"] else "normal"
    lowered = raw.lower()
    if "no acute" in lowered or "unremarkable" in lowered or "within normal limits" in lowered:
        return "normal"
    return ""


def metrics(upload_id: str = "") -> dict:
    runs = [r for r in list_runs(upload_id, limit=5000) if r.get("rated")]
    total = len(runs)
    if total == 0:
        return {"rated_runs": 0}

    agreements = [r["agreement"] for r in runs if r.get("agreement")]
    result: dict[str, Any] = {
        "rated_runs": total,
        "mean_agreement": round(sum(agreements) / len(agreements), 2) if agreements else None,
        "usable_as_draft_pct": round(
            100 * sum(1 for r in runs if r.get("usable_as_draft")) / total, 1
        ),
        "clinically_significant_error_pct": round(
            100 * sum(1 for r in runs if r.get("clinically_significant_error")) / total, 1
        ),
        "would_change_management_pct": round(
            100 * sum(1 for r in runs if r.get("would_change_management")) / total, 1
        ),
    }

    # Binary confusion matrix, where a reference label was recorded.
    tp = tn = fp = fn = 0
    for r in runs:
        ref = (r.get("reference_label") or "").lower()
        pred = _model_verdict(r)
        if ref not in ("normal", "abnormal") or pred not in ("normal", "abnormal"):
            continue
        if ref == "abnormal" and pred == "abnormal":
            tp += 1
        elif ref == "normal" and pred == "normal":
            tn += 1
        elif ref == "normal" and pred == "abnormal":
            fp += 1
        elif ref == "abnormal" and pred == "normal":
            fn += 1

    scored = tp + tn + fp + fn
    if scored:
        result["binary"] = {
            "n": scored,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": round(tp / (tp + fn), 3) if (tp + fn) else None,
            "specificity": round(tn / (tn + fp), 3) if (tn + fp) else None,
            "ppv": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "npv": round(tn / (tn + fn), 3) if (tn + fn) else None,
            "accuracy": round((tp + tn) / scored, 3),
        }

    by_template: dict[str, dict] = {}
    for r in runs:
        key = r.get("template_key") or "?"
        b = by_template.setdefault(key, {"n": 0, "agreement_sum": 0, "agreement_n": 0, "errors": 0})
        b["n"] += 1
        if r.get("agreement"):
            b["agreement_sum"] += r["agreement"]
            b["agreement_n"] += 1
        if r.get("clinically_significant_error"):
            b["errors"] += 1
    for key, b in by_template.items():
        b["mean_agreement"] = (
            round(b["agreement_sum"] / b["agreement_n"], 2) if b["agreement_n"] else None
        )
        b["error_pct"] = round(100 * b["errors"] / b["n"], 1)
    result["by_template"] = by_template

    by_modality: dict[str, dict] = {}
    for r in runs:
        key = r.get("modality") or "?"
        b = by_modality.setdefault(key, {"n": 0, "errors": 0})
        b["n"] += 1
        if r.get("clinically_significant_error"):
            b["errors"] += 1
    for key, b in by_modality.items():
        b["error_pct"] = round(100 * b["errors"] / b["n"], 1)
    result["by_modality"] = by_modality

    return result


EXPORT_COLUMNS = [
    "id", "created_at", "upload_id", "batch_id", "study_uid", "series_uid", "modality", "body_part",
    "engine", "model_id", "template_key", "image_count", "elapsed_ms",
    "context", "user_prompt", "output", "error",
    "reference_label", "model_verdict", "agreement", "usable_as_draft",
    "clinically_significant_error", "would_change_management",
    "missed_findings", "hallucinations", "ground_truth", "notes", "rated_at",
]


def export_rows(upload_id: str = "") -> list[dict]:
    rows = []
    for r in list_runs(upload_id, limit=10_000):
        r = dict(r)
        r["model_verdict"] = _model_verdict(r)
        rows.append({c: r.get(c, "") for c in EXPORT_COLUMNS})
    return rows


def export_csv(upload_id: str = "") -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in export_rows(upload_id):
        writer.writerow(row)
    return buf.getvalue()
