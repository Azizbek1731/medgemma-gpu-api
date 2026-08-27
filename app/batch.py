"""Whole-study batch analysis.

One ZIP is usually one exam — an MRI study is a dozen series of twenty to forty slices
each, and MedGemma sees at most a handful of images per request. So a study is analysed
as a *pipeline*: sample or walk each series, run one request per batch of slices, then
feed every per-series observation into a final text-only synthesis pass that writes the
consolidated report.

The sampling strategy matters. Slices are drawn stratified across the series rather than
from the ends, because the first and last slices of an MR series are usually the least
informative part of the coverage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# Rough cost model, fitted to MedGemma 4B (4-bit MLX) on an M-series Mac:
# a request costs a fixed overhead plus a per-image term.
SECONDS_BASE = 4.0
SECONDS_PER_IMAGE = 3.0
SECONDS_SYNTHESIS = 15.0

MODES = ("sample", "full")


@dataclass
class BatchJob:
    index: int  # 1-based, across the whole batch
    series_uid: str
    series_label: str
    modality: str
    body_part: str
    frames: list[int]
    part: int = 1
    parts: int = 1

    @property
    def label(self) -> str:
        suffix = f" ({self.part}/{self.parts})" if self.parts > 1 else ""
        return f"{self.series_label}{suffix}"


@dataclass
class BatchPlan:
    jobs: list[BatchJob] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    total_frames: int = 0

    @property
    def planned(self) -> int:
        return len(self.jobs)

    def estimate_seconds(self, synthesis: bool) -> float:
        total = sum(SECONDS_BASE + SECONDS_PER_IMAGE * len(j.frames) for j in self.jobs)
        if synthesis and len(self.jobs) > 1:
            total += SECONDS_SYNTHESIS
        return round(total, 1)

    def to_json(self, synthesis: bool = True) -> dict:
        return {
            "planned": self.planned,
            "total_frames": self.total_frames,
            "estimate_seconds": self.estimate_seconds(synthesis),
            "skipped": self.skipped,
            "jobs": [
                {
                    "index": j.index,
                    "series_uid": j.series_uid,
                    "label": j.label,
                    "modality": j.modality,
                    "frames": j.frames,
                    "part": j.part,
                    "parts": j.parts,
                }
                for j in self.jobs
            ],
        }


def stratified_indices(frame_count: int, k: int) -> list[int]:
    """Pick ``k`` slices spread evenly through a series, avoiding the extreme edges.

    Uses bin midpoints, so 4 samples from 40 slices gives 5, 15, 25, 35 rather than
    0, 13, 26, 39 — the ends of an MR series are usually the least useful slices.
    """
    if frame_count <= 0:
        return []
    if k >= frame_count:
        return list(range(frame_count))
    if k <= 1:
        return [frame_count // 2]
    out: list[int] = []
    for i in range(k):
        idx = int((i + 0.5) * frame_count / k)
        out.append(min(frame_count - 1, max(0, idx)))
    # Guard against duplicates on tiny series.
    seen: list[int] = []
    for idx in out:
        if idx not in seen:
            seen.append(idx)
    return seen


def _chunk(items: list[int], size: int) -> list[list[int]]:
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_plan(
    study: dict,
    *,
    series_uids: list[str] | None = None,
    mode: str = "sample",
    per_series: int = 6,
    max_images: int = 8,
) -> BatchPlan:
    """Turn a study manifest into an ordered list of inference jobs."""
    if mode not in MODES:
        raise ValueError(f"Noma'lum rejim: {mode}")

    plan = BatchPlan()
    wanted = set(series_uids) if series_uids else None
    index = 0

    for series in study.get("series", []):
        uid = series["series_instance_uid"]
        if wanted is not None and uid not in wanted:
            continue

        label = (
            series.get("series_description")
            or series.get("protocol_name")
            or f"Series {series.get('series_number') or '?'}"
        )

        if not series.get("is_image", True):
            plan.skipped.append({"label": label, "reason": "tasvirsiz obyekt (SR/PR/RTSTRUCT)"})
            continue

        frame_count = int(series.get("frame_count") or 0)
        if frame_count <= 0:
            plan.skipped.append({"label": label, "reason": "kadr topilmadi"})
            continue

        if mode == "full":
            frames = list(range(frame_count))
        else:
            frames = stratified_indices(frame_count, max(1, per_series))

        chunks = _chunk(frames, max_images)
        for part, chunk in enumerate(chunks, start=1):
            index += 1
            plan.jobs.append(
                BatchJob(
                    index=index,
                    series_uid=uid,
                    series_label=label,
                    modality=series.get("modality", ""),
                    body_part=series.get("body_part", ""),
                    frames=chunk,
                    part=part,
                    parts=len(chunks),
                )
            )
        plan.total_frames += len(frames)

    return plan


def format_series_reports(results: list[dict]) -> str:
    """Render per-series outputs into the block the synthesis prompt consumes."""
    blocks = []
    for r in results:
        text = (r.get("output") or "").strip()
        if not text:
            continue
        header = f"[{r['label']}"
        if r.get("modality"):
            header += f" · {r['modality']}"
        header += f" · slices {', '.join(str(f + 1) for f in r.get('frames', []))}]"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------------------

_cancelled: set[str] = set()
_cancel_lock = threading.Lock()


def request_cancel(batch_id: str) -> None:
    with _cancel_lock:
        _cancelled.add(batch_id)


def is_cancelled(batch_id: str) -> bool:
    with _cancel_lock:
        return batch_id in _cancelled


def clear_cancel(batch_id: str) -> None:
    with _cancel_lock:
        _cancelled.discard(batch_id)
