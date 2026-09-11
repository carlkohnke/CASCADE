"""Import and export GUI parameter sweeps as human-readable CSV files."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PROVENANCE_FIELDS = [
    "run_id",
    "run_name",
    "run_status",
    "created_at",
    "started_at",
    "finished_at",
    "error",
]


def rebuild_combined_sweep_csv(jobs: Iterable[Any], output_path: str | Path) -> Path:
    """Atomically rebuild one long-form CSV from a GUI sweep batch."""
    metric_fields = list(_tissuesim_csv_fieldnames())
    path = Path(output_path).expanduser().resolve()
    batch_jobs = [
        job
        for job in jobs
        if str(getattr(job, "combined_csv_path", "") or "") == str(path)
    ]
    sweep_fields: list[str] = []
    rows: list[dict[str, Any]] = []
    extra_fields: list[str] = []

    for job in batch_jobs:
        provenance = {
            "run_id": getattr(job, "id", ""),
            "run_name": getattr(job, "name", ""),
            "run_status": getattr(job, "status", ""),
            "created_at": getattr(job, "created_at", ""),
            "started_at": getattr(job, "started_at", ""),
            "finished_at": getattr(job, "finished_at", ""),
            "error": getattr(job, "error", ""),
        }
        for parameter in getattr(job, "sweep_parameters", []) or []:
            field = str(parameter.get("column") or "").strip()
            if not field:
                continue
            if field not in sweep_fields:
                sweep_fields.append(field)
            provenance[field] = parameter.get("value")

        summaries, read_error = _read_job_summary(job)
        if read_error and not provenance["error"]:
            provenance["error"] = read_error
        if not summaries:
            rows.append(provenance)
            continue
        for summary in summaries:
            row = dict(provenance)
            row.update(summary)
            rows.append(row)
            for field in summary:
                if (
                    field not in metric_fields
                    and field not in PROVENANCE_FIELDS
                    and field not in sweep_fields
                    and field not in extra_fields
                ):
                    extra_fields.append(field)

    fields = PROVENANCE_FIELDS + sweep_fields + metric_fields
    fields.extend(field for field in extra_fields if field not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


@lru_cache(maxsize=1)
def _tissuesim_csv_fieldnames() -> tuple[str, ...]:
    # Keep the GUI light until the user actually queues a combined sweep.
    from cascade.exporting.schema import CSV_FIELDNAMES

    return tuple(CSV_FIELDNAMES)


def _read_job_summary(job: Any) -> tuple[list[dict[str, str]], str | None]:
    if getattr(job, "status", "") != "Completed":
        return [], None
    summary_path = Path(str(getattr(job, "output_dir", ""))) / "summary.csv"
    if not summary_path.is_file():
        return [], f"Completed run has no readable summary CSV: {summary_path}"
    try:
        with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle)), None
    except Exception as exc:
        return [], f"Could not read run summary CSV: {exc}"
