"""FastAPI application: ingest DICOM, render frames, run the model, score the output."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import pydicom
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from PIL import Image

from . import batch as batch_mod
from . import deid, detections as det_mod, dicom_io, prompts, rendering, store
from .auth import ApiKeyMiddleware
from .retention import start_background_purge
from .config import UPLOAD_DIR, ensure_dirs, is_remote, settings
from .engines import (
    ENGINE_CLASSES,
    EngineError,
    GenerationRequest,
    generation_slot,
    get_engine,
    list_engines,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("aviradiology-ai")


# DIQQAT: /docs va /openapi.json ATAYLAB o'chirilgan. Ular endpoint ro'yxatini
# va sxemalarni oshkor qiladi — ochiq domendagi tibbiy servis uchun keraksiz
# hujum yuzasi. Kerak bo'lsa MG_ENABLE_DOCS=1 bilan yoqiladi (kalit baribir talab
# qilinadi, chunki middleware barcha yo'llarni yopadi).
_docs = "/docs" if os.getenv("MG_ENABLE_DOCS", "").strip() in {"1", "true", "yes"} else None

app = FastAPI(
    title="AviRadiology AI GPU API",
    version="1.0.0",
    docs_url=_docs,
    redoc_url=None,
    openapi_url="/openapi.json" if _docs else None,
)

# Autentifikatsiya — BARCHA yo'llar uchun (PUBLIC_PATHS dan tashqari).
app.add_middleware(ApiKeyMiddleware)


@app.get("/healthz")
def healthz() -> dict:
    """
    Monitoring uchun. Autentifikatsiyasiz ochiq, shuning uchun HECH QANDAY
    maxfiy ma'lumot bermaydi — na model nomi, na yuklamalar soni.
    """
    return {"status": "ok"}


@app.on_event("startup")
def _startup() -> None:
    ensure_dirs()
    store.connect()
    start_background_purge()   # PHI saqlash muddati (MG_RETENTION_HOURS)
    log.info("Data directory: %s", UPLOAD_DIR.parent)
    log.info("Engine preference: %s", settings.engine)


# --------------------------------------------------------------------------------------
# manifest helpers
# --------------------------------------------------------------------------------------


def _upload_or_404(upload_id: str) -> dict:
    up = store.get_upload(upload_id)
    if up is None:
        raise HTTPException(404, f"Upload topilmadi: {upload_id}")
    return up


def _find_series(manifest: list[dict], study_uid: str, series_uid: str) -> tuple[dict, dict]:
    for study in manifest:
        if study_uid and study["study_instance_uid"] != study_uid:
            continue
        for series in study["series"]:
            if series["series_instance_uid"] == series_uid:
                return study, series
    raise HTTPException(404, f"Series topilmadi: {series_uid}")


def flatten_frames(series: dict) -> list[tuple[int, int]]:
    """Series -> flat list of ``(instance_index, frame_index)`` for linear scrolling."""
    out: list[tuple[int, int]] = []
    for i, inst in enumerate(series["instances"]):
        if not inst.get("is_image", True):
            continue
        for f in range(max(1, int(inst.get("frames", 1) or 1))):
            out.append((i, f))
    return out


def _frame_path(root: Path, series: dict, instance_index: int) -> Path:
    try:
        rel = series["instances"][instance_index]["path"]
    except (IndexError, KeyError) as exc:
        raise HTTPException(404, "Instance topilmadi") from exc
    path = root / rel
    if not path.exists():
        raise HTTPException(404, f"Fayl yo'q: {rel}")
    return path


# --------------------------------------------------------------------------------------
# upload + browse
# --------------------------------------------------------------------------------------


@app.post("/api/upload")
async def upload_archive(file: UploadFile = File(...), label: str = Query("")) -> dict:
    ensure_dirs()
    upload_id = store.new_id("up")
    dest = UPLOAD_DIR / upload_id
    dest.mkdir(parents=True, exist_ok=True)

    # DIQQAT: fayl nomi kiruvchi fayl turiga qarab tanlanadi.
    # Ilgari hamma yuklama "_archive.zip" nomi bilan saqlanardi va quyidagi
    # `.zip` tekshiruvi DOIM rost bo'lardi — bitta .dcm fayl yuklash tarmog'i
    # hech qachon bajarilmay, har safar "Arxivni ochib bo'lmadi" (400) berardi.
    incoming = (file.filename or "").lower()
    is_zip = incoming.endswith(".zip") or not incoming.endswith(".dcm")
    archive = dest / ("_archive.zip" if is_zip else "_instance.dcm")
    size = 0
    limit = settings.max_upload_mb * 1024 * 1024
    try:
        with open(archive, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Fayl {settings.max_upload_mb} MB dan katta.")
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    extracted = dest / "files"
    report = dicom_io.IngestReport()
    try:
        if is_zip:
            dicom_io.extract_archive(archive, extracted, settings.max_files_per_zip)
        else:
            extracted.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archive), extracted / (file.filename or "instance.dcm"))
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(400, f"Arxivni ochib bo'lmadi: {exc}") from exc
    finally:
        archive.unlink(missing_ok=True)

    # index_directory buzuq DICOM'da xato berishi mumkin (masalan
    # ImagePositionPatient VM=1). Ilgari bu xato ushlanmasdi: bazada yozuv
    # yaratilmasdi, lekin OCHILGAN BEMOR FAYLLARI diskda qolardi. Tozalash
    # esa faqat bazadagi yozuvlar bo'ylab yuradi — ya'ni bunday papka
    # hech qachon o'chmasdi.
    try:
        studies = dicom_io.index_directory(extracted, report)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(dest, ignore_errors=True)
        log.exception("Arxivni indekslash muvaffaqiyatsiz: %s", upload_id)
        raise HTTPException(400, f"Arxivni o'qib bo'lmadi: {exc}") from exc

    if not studies:
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(
            400,
            f"DICOM fayl topilmadi. Ko'rilgan fayllar: {report.total_files}, "
            f"o'qib bo'lmagan: {report.unreadable_files}.",
        )

    manifest = dicom_io.studies_to_json(studies)
    report_dict = {
        "total_files": report.total_files,
        "dicom_files": report.dicom_files,
        "skipped_files": report.skipped_files,
        "unreadable_files": report.unreadable_files,
        "non_image_instances": report.non_image_instances,
        "errors": report.errors[:20],
        "study_count": len(studies),
    }

    # Reuse the pre-generated id so the on-disk directory and the row match.
    store.create_upload(
        label=label or (file.filename or "upload"),
        source_name=file.filename or "",
        path=extracted,
        report=report_dict,
        manifest=manifest,
        upload_id=upload_id,
    )

    return {"upload_id": upload_id, "report": report_dict, "studies": manifest}


@app.get("/api/uploads")
def api_list_uploads() -> list[dict]:
    return store.list_uploads()


@app.get("/api/uploads/{upload_id}")
def api_get_upload(upload_id: str) -> dict:
    up = _upload_or_404(upload_id)
    return {
        "upload_id": up["id"],
        "label": up["label"],
        "created_at": up["created_at"],
        "report": up["report"],
        "studies": up["manifest"],
    }


@app.delete("/api/uploads/{upload_id}")
def api_delete_upload(upload_id: str) -> dict:
    up = _upload_or_404(upload_id)
    shutil.rmtree(Path(up["path"]).parent, ignore_errors=True)
    store.delete_upload(upload_id)
    return {"deleted": upload_id}


@app.get("/api/uploads/{upload_id}/series")
def api_series_detail(
    upload_id: str, study_uid: str = Query(""), series_uid: str = Query(...)
) -> dict:
    up = _upload_or_404(upload_id)
    study, series = _find_series(up["manifest"], study_uid, series_uid)
    frames = flatten_frames(series)
    return {
        "study": {k: v for k, v in study.items() if k != "series"},
        "series": {k: v for k, v in series.items() if k != "instances"},
        "frame_count": len(frames),
        "presets": rendering.suggest_presets(series.get("modality", "")),
        "instances": [
            {
                "index": i,
                "instance_number": inst.get("instance_number"),
                "frames": inst.get("frames", 1),
                "rows": inst.get("rows"),
                "columns": inst.get("columns"),
                "slice_location": inst.get("slice_location"),
                "is_image": inst.get("is_image", True),
            }
            for i, inst in enumerate(series["instances"])
        ],
    }


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def _render_frame_of(
    root: Path,
    series: dict,
    frame: int,
    wc: float | None,
    ww: float | None,
    preset: str | None,
    invert: bool | None,
    px: int,
) -> rendering.RenderResult:
    """Render one frame of an already-resolved series (no manifest lookup)."""
    frames = flatten_frames(series)
    if not frames:
        raise HTTPException(400, "Bu seriyada ko'rsatiladigan tasvir yo'q.")
    frame = max(0, min(frame, len(frames) - 1))
    inst_idx, frame_idx = frames[frame]
    path = _frame_path(root, series, inst_idx)
    try:
        return rendering.render_frame(
            path,
            frame=frame_idx,
            window_center=wc,
            window_width=ww,
            preset=preset,
            invert=invert,
            max_px=px,
        )
    except Exception as exc:  # noqa: BLE001
        # Faqat fayl NOMI, to'liq yo'l emas: PACS eksportlarida yo'lda bemor
        # ismi yoki MRN bo'lishi mumkin, ilova logi esa PHI saqlash muddati
        # qamrovidan tashqarida turadi.
        log.exception("Kadrni render qilib bo'lmadi (%s)", path.name)
        raise HTTPException(500, f"Tasvirni ochib bo'lmadi: {exc}") from exc


def _render(
    upload_id: str,
    study_uid: str,
    series_uid: str,
    frame: int,
    wc: float | None,
    ww: float | None,
    preset: str | None,
    invert: bool | None,
    px: int,
) -> tuple[rendering.RenderResult, dict, dict]:
    up = _upload_or_404(upload_id)
    study, series = _find_series(up["manifest"], study_uid, series_uid)
    result = _render_frame_of(Path(up["path"]), series, frame, wc, ww, preset, invert, px)
    return result, study, series


@app.get("/api/uploads/{upload_id}/frame.png")
def api_frame(
    upload_id: str,
    series_uid: str = Query(...),
    study_uid: str = Query(""),
    frame: int = Query(0),
    wc: float | None = Query(None),
    ww: float | None = Query(None),
    preset: str | None = Query(None),
    invert: bool | None = Query(None),
    px: int = Query(0),
) -> Response:
    result, _, _ = _render(
        upload_id, study_uid, series_uid, frame, wc, ww, preset, invert,
        px or settings.viewer_image_px,
    )
    buf = io.BytesIO()
    result.image.save(buf, format="PNG", optimize=True)
    headers = {
        "Cache-Control": "no-store",
        "X-Window-Center": str(result.window_center),
        "X-Window-Width": str(result.window_width),
        "X-Window-Source": result.window_source,
        "X-Value-Min": f"{result.value_min:.1f}",
        "X-Value-Max": f"{result.value_max:.1f}",
        "X-Units": result.units or "",
        "X-Inverted": "1" if result.inverted else "0",
    }
    return Response(buf.getvalue(), media_type="image/png", headers=headers)


@app.get("/api/uploads/{upload_id}/frame-info")
def api_frame_info(
    upload_id: str,
    series_uid: str = Query(...),
    study_uid: str = Query(""),
    frame: int = Query(0),
) -> dict:
    """Header-level detail plus the PHI audit for one frame."""
    up = _upload_or_404(upload_id)
    study, series = _find_series(up["manifest"], study_uid, series_uid)
    frames = flatten_frames(series)
    if not frames:
        raise HTTPException(400, "Tasvir yo'q.")
    frame = max(0, min(frame, len(frames) - 1))
    inst_idx, frame_idx = frames[frame]
    path = _frame_path(Path(up["path"]), series, inst_idx)

    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    findings = deid.audit_dataset(ds)
    burned, burned_reason = deid.burned_in_risk(ds)
    dwc, dww = rendering.dicom_window(ds)

    return {
        "instance_index": inst_idx,
        "frame_index": frame_idx,
        "file": series["instances"][inst_idx]["path"],
        "modality": str(getattr(ds, "Modality", "") or ""),
        "photometric": str(getattr(ds, "PhotometricInterpretation", "") or ""),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
        "slice_thickness": series.get("slice_thickness"),
        "slice_location": series["instances"][inst_idx].get("slice_location"),
        "dicom_window": {"center": dwc, "width": dww},
        "transfer_syntax": series["instances"][inst_idx].get("transfer_syntax", ""),
        "phi": deid.summarize_findings(findings),
        "burned_in": {"risk": burned, "reason": burned_reason},
        "clinical_context": deid.safe_clinical_context(
            modality=series.get("modality", ""),
            body_part=series.get("body_part", ""),
            patient_age=study.get("patient_age", ""),
            patient_sex=study.get("patient_sex", ""),
            view_position=series.get("view_position", ""),
            laterality=series.get("laterality", ""),
            series_description=series.get("series_description", ""),
            study_description=study.get("study_description", ""),
            contrast=series.get("contrast", ""),
        ),
    }


# --------------------------------------------------------------------------------------
# engines + templates
# --------------------------------------------------------------------------------------


@app.get("/api/engines")
def api_engines() -> dict:
    from .engines import current_engine, resolve_auto

    # Before anything is loaded, report the capability of the model that *would* run,
    # otherwise a fresh page load always shows "localization unsupported".
    active = current_engine()
    if active is not None:
        model_id = active.model_id
    else:
        model_id = settings.model_id
        if not model_id:
            key = settings.engine if settings.engine != "auto" else resolve_auto()
            cls = ENGINE_CLASSES.get(key)
            model_id = cls.default_model if cls else ""
    return {
        "engines": list_engines(),
        "auto": resolve_auto(),
        "configured": settings.engine,
        "local_only": settings.local_only,
        "max_images": settings.max_images_per_request,
        "localization": {
            "supported": det_mod.model_supports_localization(model_id),
            "warning": det_mod.localization_warning(model_id),
            "modalities": list(det_mod.LOCALIZATION_MODALITIES),
            "recommended_model": "google/medgemma-1.5-4b-it",
        },
    }


@app.get("/api/templates")
def api_templates(modality: str = Query("")) -> list[dict]:
    return prompts.list_templates(modality)


# --------------------------------------------------------------------------------------
# inferens
# --------------------------------------------------------------------------------------


def _sse(event: str, data: Any) -> str:
    """Bitta SSE blokini yasaydi: `event:` + `data:` + bo'sh qator."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/infer")
def api_infer(payload: dict = Body(...)) -> StreamingResponse:
    upload_id = payload.get("upload_id", "")
    study_uid = payload.get("study_uid", "")
    series_uid = payload.get("series_uid", "")
    frame_list: list[int] = payload.get("frames") or [0]
    template_key = payload.get("template", "findings_only")
    variables: dict = payload.get("variables") or {}
    engine_key = payload.get("engine", "")
    model_id = payload.get("model_id", "")
    window = payload.get("window") or {}
    include_context = bool(payload.get("include_context", True))
    extra_context = str(payload.get("extra_context", "") or "")
    max_new_tokens = int(payload.get("max_new_tokens") or settings.max_new_tokens)
    temperature = float(payload.get("temperature", settings.temperature))

    up = _upload_or_404(upload_id)
    study, series = _find_series(up["manifest"], study_uid, series_uid)

    if settings.local_only and is_remote(engine_key):
        raise HTTPException(403, "MG_LOCAL_ONLY yoqilgan — tarmoq engine'lari o'chirilgan.")

    frame_list = frame_list[: settings.max_images_per_request]
    if not frame_list:
        raise HTTPException(400, "Kamida bitta kadr tanlang.")

    # --- render every selected frame with identical window settings ---
    images: list[Image.Image] = []
    frame_meta: list[dict] = []
    for f in frame_list:
        result, _, _ = _render(
            upload_id, study_uid, series_uid, int(f),
            window.get("wc"), window.get("ww"), window.get("preset"), window.get("invert"),
            settings.inference_image_px,
        )
        img = result.image
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
        frame_meta.append(
            {
                "frame": int(f),
                "wc": result.window_center,
                "ww": result.window_width,
                "source": result.window_source,
                "size": list(img.size),
            }
        )

    # --- build the prompt ---
    context = ""
    if include_context:
        context = deid.safe_clinical_context(
            modality=series.get("modality", ""),
            body_part=series.get("body_part", ""),
            patient_age=study.get("patient_age", ""),
            patient_sex=study.get("patient_sex", ""),
            view_position=series.get("view_position", ""),
            laterality=series.get("laterality", ""),
            series_description=series.get("series_description", ""),
            study_description=study.get("study_description", ""),
            contrast=series.get("contrast", ""),
            extra=extra_context,
        )
        if len(images) > 1:
            context += f"\nImages provided: {len(images)} (in order)."
    elif extra_context:
        context = extra_context

    try:
        system_prompt, user_prompt = prompts.build_prompt(template_key, context, **variables)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        engine = get_engine(engine_key, model_id)
    except EngineError as exc:
        raise HTTPException(400, str(exc)) from exc

    run_id = store.create_run(
        upload_id=upload_id,
        study_uid=study.get("study_instance_uid", ""),
        series_uid=series_uid,
        modality=series.get("modality", ""),
        body_part=series.get("body_part", ""),
        frames=frame_meta,
        window=window,
        engine=engine.key,
        model_id=engine.model_id,
        template_key=template_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context=context,
        image_count=len(images),
    )

    req = GenerationRequest(
        system=system_prompt,
        user_text=user_prompt,
        images=images,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    def generator() -> Iterator[str]:
        started = time.perf_counter()
        collected: list[str] = []
        yield _sse(
            "start",
            {
                "run_id": run_id,
                "engine": engine.key,
                "model_id": engine.model_id,
                "images": len(images),
                "frames": frame_meta,
                "system": system_prompt,
                "prompt": user_prompt,
                "remote": is_remote(engine.key),
                # Shifokor "klinik kontekst" maydoniga bemor ismini yoki MRN
                # ni yopishtirishi mumkin — u to'g'ridan-to'g'ri promptga,
                # ya'ni modelga ketadi. deid.scan_free_text buni topadi,
                # lekin ilgari HECH QAYERDA chaqirilmasdi.
                "context_warnings": deid.scan_free_text(extra_context),
            },
        )
        try:
            with generation_slot() as slot:
                if slot.waited:
                    yield _sse("status", {"message": "Navbat tugadi, boshlanmoqda…"})
                if not engine.loaded:
                    yield _sse("status", {"message": f"Model yuklanmoqda: {engine.model_id} ..."})
                    engine.load()
                    yield _sse("status", {"message": f"Model tayyor ({engine.device_name()})."})

                for delta in engine.stream(req):
                    collected.append(delta)
                    yield _sse("delta", {"text": delta})

            elapsed = int((time.perf_counter() - started) * 1000)
            output = "".join(collected)
            parsed = det_mod.parse_detailed(output, image_count=len(images))
            boxes = det_mod.to_json(parsed.detections)
            store.finish_run(run_id, output, elapsed, detections=boxes)

            notes = []
            if parsed.rejected:
                notes.append(parsed.rejected_summary)
            if boxes or parsed.rejected:
                caveat = det_mod.localization_warning(
                    engine.model_id, series.get("modality", "")
                )
                if caveat:
                    notes.append(caveat)

            yield _sse(
                "done",
                {
                    "run_id": run_id,
                    "elapsed_ms": elapsed,
                    "chars": len(output),
                    "detections": boxes,
                    "rejected": parsed.rejected,
                    "frames": frame_meta,
                    "localization_warning": " ".join(notes),
                },
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = int((time.perf_counter() - started) * 1000)
            message = str(exc)
            log.exception("Inference failed")
            store.finish_run(run_id, "".join(collected), elapsed, error=message)
            yield _sse("error", {"run_id": run_id, "message": message})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------------------
# batch: analyse the whole study
# --------------------------------------------------------------------------------------


def _study_context(study: dict, series_list: list[dict]) -> str:
    """Study-level context — the parts common to every series."""
    modalities = sorted({s.get("modality", "") for s in series_list if s.get("modality")})
    body_parts = sorted({s.get("body_part", "") for s in series_list if s.get("body_part")})
    return deid.safe_clinical_context(
        modality="/".join(modalities),
        body_part=body_parts[0] if body_parts else "",
        patient_age=study.get("patient_age", ""),
        patient_sex=study.get("patient_sex", ""),
        study_description=study.get("study_description", ""),
    )


def _resolve_batch_request(payload: dict) -> tuple[dict, dict, list[dict], batch_mod.BatchPlan]:
    upload_id = payload.get("upload_id", "")
    study_uid = payload.get("study_uid", "")
    up = _upload_or_404(upload_id)

    study = None
    for st in up["manifest"]:
        if not study_uid or st["study_instance_uid"] == study_uid:
            study = st
            break
    if study is None:
        raise HTTPException(404, f"Tekshiruv topilmadi: {study_uid}")

    series_uids = payload.get("series_uids") or None
    try:
        plan = batch_mod.build_plan(
            study,
            series_uids=series_uids,
            mode=payload.get("mode", "sample"),
            per_series=int(payload.get("per_series") or 6),
            max_images=min(
                int(payload.get("max_images") or settings.max_images_per_request),
                settings.max_images_per_request,
            ),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    chosen = [
        s
        for s in study["series"]
        if not series_uids or s["series_instance_uid"] in set(series_uids)
    ]
    return up, study, chosen, plan


@app.post("/api/batch/plan")
def api_batch_plan(payload: dict = Body(...)) -> dict:
    """Preview what a whole-study run would do, without running anything."""
    _, study, chosen, plan = _resolve_batch_request(payload)
    return {
        "study": {k: v for k, v in study.items() if k != "series"},
        "series": [
            {
                "series_instance_uid": s["series_instance_uid"],
                "modality": s.get("modality", ""),
                "label": s.get("series_description") or s.get("protocol_name") or "",
                "frame_count": s.get("frame_count", 0),
                "is_image": s.get("is_image", True),
            }
            for s in study["series"]
        ],
        "selected": len(chosen),
        **plan.to_json(bool(payload.get("synthesis", True))),
    }


@app.post("/api/batch")
def api_batch(payload: dict = Body(...)) -> StreamingResponse:
    up, study, chosen, plan = _resolve_batch_request(payload)

    engine_key = payload.get("engine", "")
    model_id = payload.get("model_id", "")
    if settings.local_only and is_remote(engine_key):
        raise HTTPException(403, "MG_LOCAL_ONLY yoqilgan — tarmoq engine'lari o'chirilgan.")
    if not plan.jobs:
        raise HTTPException(400, "Tahlil qilinadigan seriya topilmadi.")

    template_key = payload.get("template", "series_review")
    do_synthesis = bool(payload.get("synthesis", True)) and len(plan.jobs) > 1
    window = payload.get("window") or {}
    per_window: dict = payload.get("series_window") or {}
    max_new_tokens = min(int(payload.get("max_new_tokens") or 700),
                         settings.max_new_tokens)   # mijoz GPU\'ni cheksiz band qila olmasin
    temperature = float(payload.get("temperature", settings.temperature))
    extra_context = str(payload.get("extra_context", "") or "")

    try:
        engine = get_engine(engine_key, model_id)
    except EngineError as exc:
        raise HTTPException(400, str(exc)) from exc

    root = Path(up["path"])
    series_by_uid = {s["series_instance_uid"]: s for s in study["series"]}
    study_context = _study_context(study, chosen)
    if extra_context:
        study_context = (study_context + "\n" + extra_context).strip()

    batch_id = store.create_batch(
        upload_id=up["id"],
        study_uid=study.get("study_instance_uid", ""),
        label=study.get("study_description") or up["label"],
        mode=payload.get("mode", "sample"),
        engine=engine.key,
        model_id=engine.model_id,
        template_key=template_key,
        planned_jobs=len(plan.jobs),
    )

    def generator() -> Iterator[str]:
        started = time.perf_counter()
        completed = failed = 0
        results: list[dict] = []
        status = "done"
        batch_mod.clear_cancel(batch_id)

        yield _sse(
            "plan",
            {
                "batch_id": batch_id,
                "engine": engine.key,
                "model_id": engine.model_id,
                "remote": is_remote(engine.key),
                "synthesis": do_synthesis,
                **plan.to_json(do_synthesis),
            },
        )

        try:
            if not engine.loaded:
                yield _sse("status", {"message": f"Model yuklanmoqda: {engine.model_id} ..."})
                with generation_slot():
                    engine.load()
                yield _sse("status", {"message": f"Model tayyor ({engine.device_name()})."})

            for job in plan.jobs:
                if batch_mod.is_cancelled(batch_id):
                    status = "cancelled"
                    break

                series = series_by_uid.get(job.series_uid)
                if series is None:
                    failed += 1
                    continue

                win = {**window, **(per_window.get(job.series_uid) or {})}
                yield _sse(
                    "job_start",
                    {
                        "index": job.index,
                        "total": len(plan.jobs),
                        "label": job.label,
                        "modality": job.modality,
                        "frames": job.frames,
                    },
                )

                # --- render this job's slices ---
                images: list[Image.Image] = []
                frame_meta: list[dict] = []
                render_error = ""
                for f in job.frames:
                    try:
                        r = _render_frame_of(
                            root, series, f, win.get("wc"), win.get("ww"),
                            win.get("preset"), win.get("invert"), settings.inference_image_px,
                        )
                    except HTTPException as exc:
                        render_error = str(exc.detail)
                        break
                    img = r.image if r.image.mode == "RGB" else r.image.convert("RGB")
                    images.append(img)
                    frame_meta.append(
                        {"frame": f, "wc": r.window_center, "ww": r.window_width,
                         "source": r.window_source, "size": list(img.size)}
                    )

                if render_error or not images:
                    failed += 1
                    store.bump_batch(batch_id, failed=1)
                    yield _sse(
                        "job_error",
                        {"index": job.index, "message": render_error or "Kadr render qilinmadi"},
                    )
                    continue

                context = (
                    f"{study_context}\nSeries: {job.series_label}"
                    f"\nImages provided: {len(images)} "
                    f"(slice numbers {', '.join(str(f + 1) for f in job.frames)})"
                ).strip()
                system_prompt, user_prompt = prompts.build_prompt(template_key, context)

                run_id = store.create_run(
                    upload_id=up["id"],
                    study_uid=study.get("study_instance_uid", ""),
                    series_uid=job.series_uid,
                    modality=job.modality,
                    body_part=job.body_part,
                    frames=frame_meta,
                    window=win,
                    engine=engine.key,
                    model_id=engine.model_id,
                    template_key=template_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    context=context,
                    image_count=len(images),
                    batch_id=batch_id,
                )

                job_started = time.perf_counter()
                collected: list[str] = []
                try:
                    # Per job, not per batch: an interactive run can slip in between jobs
                    # instead of waiting out a 20-minute study.
                    with generation_slot():
                        for delta in engine.stream(
                            GenerationRequest(
                                system=system_prompt,
                                user_text=user_prompt,
                                images=images,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                            )
                        ):
                            collected.append(delta)
                            yield _sse("delta", {"index": job.index, "text": delta})
                            if batch_mod.is_cancelled(batch_id):
                                break
                except Exception as exc:  # noqa: BLE001
                    log.exception("Batch job %s failed", job.index)
                    failed += 1
                    store.finish_run(
                        run_id, "".join(collected),
                        int((time.perf_counter() - job_started) * 1000), error=str(exc),
                    )
                    store.bump_batch(batch_id, failed=1)
                    yield _sse("job_error", {"index": job.index, "message": str(exc)})
                    continue

                elapsed = int((time.perf_counter() - job_started) * 1000)
                text = "".join(collected)
                boxes = det_mod.to_json(det_mod.parse(text, image_count=len(images)))
                store.finish_run(run_id, text, elapsed, detections=boxes)
                store.bump_batch(batch_id, completed=1)
                completed += 1
                results.append(
                    {
                        "label": job.label,
                        "modality": job.modality,
                        "frames": job.frames,
                        "output": text,
                        "run_id": run_id,
                    }
                )
                yield _sse(
                    "job_done",
                    {"index": job.index, "run_id": run_id, "elapsed_ms": elapsed,
                     "chars": len(text), "detections": boxes,
                     "series_uid": job.series_uid, "frames": job.frames},
                )

            # --- study-level synthesis ---
            synthesis = ""
            if status != "cancelled" and do_synthesis and results:
                yield _sse("synthesis_start", {"sources": len(results)})
                sys_p, user_p = prompts.build_prompt(
                    "study_synthesis",
                    study_context,
                    series_reports=batch_mod.format_series_reports(results),
                )
                syn_run = store.create_run(
                    upload_id=up["id"],
                    study_uid=study.get("study_instance_uid", ""),
                    modality="/".join(sorted({r["modality"] for r in results if r["modality"]})),
                    engine=engine.key,
                    model_id=engine.model_id,
                    template_key="study_synthesis",
                    system_prompt=sys_p,
                    user_prompt=user_p,
                    context=study_context,
                    batch_id=batch_id,
                )
                syn_started = time.perf_counter()
                chunks: list[str] = []
                try:
                    with generation_slot():
                        for delta in engine.stream(
                            GenerationRequest(
                                system=sys_p, user_text=user_p, images=[],
                                max_new_tokens=max(max_new_tokens, 1200),
                                temperature=temperature,
                            )
                        ):
                            chunks.append(delta)
                            yield _sse("synthesis_delta", {"text": delta})
                    synthesis = "".join(chunks)
                    store.finish_run(
                        syn_run, synthesis, int((time.perf_counter() - syn_started) * 1000)
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("Synthesis failed")
                    store.finish_run(syn_run, "".join(chunks), 0, error=str(exc))
                    yield _sse("job_error", {"index": 0, "message": f"Sintez xatosi: {exc}"})

            total_ms = int((time.perf_counter() - started) * 1000)
            store.finish_batch(batch_id, status, synthesis, total_ms)
            yield _sse(
                "done",
                {"batch_id": batch_id, "status": status, "elapsed_ms": total_ms,
                 "completed": completed, "failed": failed},
            )
        except GeneratorExit:  # client navigated away / aborted
            store.finish_batch(batch_id, "cancelled", "", int((time.perf_counter() - started) * 1000))
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Batch failed")
            store.finish_batch(
                batch_id, "error", "", int((time.perf_counter() - started) * 1000), str(exc)
            )
            yield _sse("error", {"batch_id": batch_id, "message": str(exc)})
        finally:
            batch_mod.clear_cancel(batch_id)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/batch/{batch_id}/cancel")
def api_batch_cancel(batch_id: str) -> dict:
    batch_mod.request_cancel(batch_id)
    return {"cancelled": batch_id}


@app.get("/api/batches")
def api_batches(upload_id: str = Query("")) -> list[dict]:
    return store.list_batches(upload_id)


@app.get("/api/batches/{batch_id}")
def api_batch_detail(batch_id: str) -> dict:
    b = store.get_batch(batch_id)
    if b is None:
        raise HTTPException(404, "Batch topilmadi")

    # Runs store only the series UID; resolve readable labels from the upload manifest.
    up = store.get_upload(b["upload_id"])
    labels: dict[str, str] = {}
    if up:
        for study in up["manifest"]:
            for series in study["series"]:
                labels[series["series_instance_uid"]] = (
                    series.get("series_description")
                    or series.get("protocol_name")
                    or f"Series {series.get('series_number') or '?'}"
                )
    for run in b["runs"]:
        run["series_label"] = labels.get(run.get("series_uid", ""), "")
    return b


@app.delete("/api/batches/{batch_id}")
def api_batch_delete(batch_id: str) -> dict:
    store.delete_batch(batch_id)
    return {"deleted": batch_id}


# --------------------------------------------------------------------------------------
# runs, scoring, export
# --------------------------------------------------------------------------------------


@app.get("/api/runs")
def api_runs(upload_id: str = Query(""), limit: int = Query(200)) -> list[dict]:
    return store.list_runs(upload_id, limit)


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run topilmadi")
    return run


@app.delete("/api/runs/{run_id}")
def api_delete_run(run_id: str) -> dict:
    store.delete_run(run_id)
    return {"deleted": run_id}


@app.post("/api/runs/{run_id}/evaluate")
def api_evaluate(run_id: str, payload: dict = Body(...)) -> dict:
    if store.get_run(run_id) is None:
        raise HTTPException(404, "Run topilmadi")
    store.save_evaluation(run_id, payload)
    return {"saved": run_id, "metrics": store.metrics()}


@app.get("/api/metrics")
def api_metrics(upload_id: str = Query("")) -> dict:
    return store.metrics(upload_id)


@app.get("/api/export.csv")
def api_export_csv(upload_id: str = Query("")) -> Response:
    csv_text = store.export_csv(upload_id)
    return Response(
        csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="medgemma-runs.csv"'},
    )


@app.get("/api/export.json")
def api_export_json(upload_id: str = Query("")) -> Response:
    data = {"metrics": store.metrics(upload_id), "runs": store.export_rows(upload_id)}
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="medgemma-runs.json"'},
    )


# --------------------------------------------------------------------------------------
# static
# --------------------------------------------------------------------------------------
