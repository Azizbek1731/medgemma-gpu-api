"""DICOM pixel pipeline: Modality LUT -> VOI LUT / window -> presentation -> PNG.

Getting this wrong is the fastest way to get worthless model output, so the order of
operations follows PS3.4 Annex N:

1. Stored values -> real-world values (Rescale Slope/Intercept, i.e. HU for CT)
2. Real-world values -> displayed range (Window Center/Width or a VOI LUT Sequence)
3. Presentation transform (MONOCHROME1 means "0 = white", so invert)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

log = logging.getLogger(__name__)

try:  # pydicom >= 3.0
    from pydicom.pixels import apply_modality_lut, apply_voi_lut, convert_color_space
except ImportError:  # pydicom 2.x
    from pydicom.pixel_data_handlers.util import (  # type: ignore[no-redef]
        apply_modality_lut,
        apply_voi_lut,
        convert_color_space,
    )


@dataclass(frozen=True)
class WindowPreset:
    key: str
    label: str
    center: float
    width: float
    modality: tuple[str, ...] = ()


# Conventional radiology windows. CT values are in Hounsfield units.
PRESETS: list[WindowPreset] = [
    WindowPreset("ct_soft", "Soft tissue (50/400)", 50, 400, ("CT",)),
    WindowPreset("ct_lung", "Lung (-600/1500)", -600, 1500, ("CT",)),
    WindowPreset("ct_mediastinum", "Mediastinum (50/350)", 50, 350, ("CT",)),
    WindowPreset("ct_bone", "Bone (400/1800)", 400, 1800, ("CT",)),
    WindowPreset("ct_brain", "Brain (40/80)", 40, 80, ("CT",)),
    WindowPreset("ct_stroke", "Stroke / narrow (32/8)", 32, 8, ("CT",)),
    WindowPreset("ct_post_fossa", "Posterior fossa (40/40)", 40, 40, ("CT",)),
    WindowPreset("ct_liver", "Liver (60/160)", 60, 160, ("CT",)),
    WindowPreset("ct_angio", "CTA / vascular (200/700)", 200, 700, ("CT",)),
    WindowPreset("ct_spine_soft", "Spine soft tissue (50/250)", 50, 250, ("CT",)),
]

PRESETS_BY_KEY = {p.key: p for p in PRESETS}

# Modalities whose stored values have no standard scale -> percentile auto-window.
AUTO_WINDOW_MODALITIES = {"MR", "PT", "NM", "US", "XA", "RF", "OT"}


@dataclass
class RenderResult:
    image: Image.Image
    window_center: float | None
    window_width: float | None
    window_source: str  # "manual" | "preset" | "dicom" | "voi_lut" | "auto" | "rgb"
    value_min: float
    value_max: float
    modality: str
    photometric: str
    inverted: bool
    units: str
    frames: int


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _as_float(value) -> float | None:
    try:
        if isinstance(value, pydicom.multival.MultiValue):
            value = value[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return None


def dicom_window(ds: pydicom.Dataset) -> tuple[float | None, float | None]:
    """The window the modality itself recorded, if any."""
    wc = _as_float(getattr(ds, "WindowCenter", None))
    ww = _as_float(getattr(ds, "WindowWidth", None))
    if ww is not None and ww <= 0:
        ww = None
    return wc, ww


def auto_window(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> tuple[float, float]:
    """Percentile window, ignoring the air/background mode that dominates most scans."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    # Drop the single most common value (usually padding / air) before taking percentiles.
    if finite.size > 10_000:
        finite = finite[:: max(1, finite.size // 200_000)]
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return (lo + hi) / 2.0, hi - lo


def suggest_presets(modality: str) -> list[dict]:
    mod = (modality or "").upper()
    items = [p for p in PRESETS if not p.modality or mod in p.modality]
    return [{"key": p.key, "label": p.label, "center": p.center, "width": p.width} for p in items]


def _read_frame(ds: pydicom.Dataset, path: Path, frame: int) -> np.ndarray:
    """Decode a single frame, preferring pydicom 3's lazy per-frame decoder."""
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    frame = max(0, min(frame, n_frames - 1))
    if n_frames > 1:
        try:
            from pydicom.pixels import pixel_array as _pixel_array

            return np.asarray(_pixel_array(str(path), index=frame))
        except Exception:  # noqa: BLE001 - fall back to full decode
            pass
    arr = ds.pixel_array
    if arr.ndim >= 3 and n_frames > 1:
        arr = arr[frame]
    return np.asarray(arr)


def _to_uint8_rgb(ds: pydicom.Dataset, arr: np.ndarray) -> np.ndarray:
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric.startswith("YBR"):
        try:
            arr = convert_color_space(arr, photometric, "RGB")
        except Exception as exc:  # noqa: BLE001
            log.warning("Colour conversion %s->RGB failed: %s", photometric, exc)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        hi = float(arr.max()) or 1.0
        arr = np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8)
    return arr


def _resize(img: Image.Image, max_px: int) -> Image.Image:
    if max_px <= 0:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_px:
        return img
    scale = max_px / longest
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------


def render_frame(
    path: Path | str,
    frame: int = 0,
    window_center: float | None = None,
    window_width: float | None = None,
    preset: str | None = None,
    invert: bool | None = None,
    max_px: int = 1024,
) -> RenderResult:
    """Render one DICOM frame to a display-ready :class:`PIL.Image`."""
    path = Path(path)
    ds = pydicom.dcmread(path, force=True)
    modality = str(getattr(ds, "Modality", "") or "").upper()
    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)

    raw = _read_frame(ds, path, frame)

    # --- colour images (US, secondary capture, some XA) bypass windowing entirely ---
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    if samples > 1 or raw.ndim == 3:
        rgb = _to_uint8_rgb(ds, raw)
        img = _resize(Image.fromarray(rgb, mode="RGB"), max_px)
        return RenderResult(
            image=img,
            window_center=None,
            window_width=None,
            window_source="rgb",
            value_min=0.0,
            value_max=255.0,
            modality=modality,
            photometric=photometric,
            inverted=False,
            units="RGB",
            frames=n_frames,
        )

    if photometric == "PALETTE COLOR":
        try:
            from pydicom.pixels import apply_color_lut

            rgb = apply_color_lut(raw, ds)
            rgb = _to_uint8_rgb(ds, rgb)
            img = _resize(Image.fromarray(rgb, mode="RGB"), max_px)
            return RenderResult(
                image=img, window_center=None, window_width=None, window_source="rgb",
                value_min=0.0, value_max=255.0, modality=modality, photometric=photometric,
                inverted=False, units="RGB", frames=n_frames,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Palette LUT failed: %s", exc)

    # --- step 1: Modality LUT (stored value -> HU / real world value) ---
    try:
        values = apply_modality_lut(raw, ds).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        log.warning("Modality LUT failed (%s); using raw values", exc)
        values = raw.astype(np.float32)

    units = "HU" if modality == "CT" else str(getattr(ds, "RescaleType", "") or "")

    # --- step 2: decide the window ---
    source = "auto"
    wc, ww = window_center, window_width
    if wc is not None and ww is not None and ww > 0:
        source = "manual"
    elif preset and preset in PRESETS_BY_KEY:
        p = PRESETS_BY_KEY[preset]
        wc, ww, source = p.center, p.width, "preset"
    elif preset == "dicom" or (preset in (None, "", "auto") and modality not in AUTO_WINDOW_MODALITIES):
        dwc, dww = dicom_window(ds)
        if dwc is not None and dww is not None:
            wc, ww, source = dwc, dww, "dicom"

    if wc is None or ww is None or ww <= 0:
        # A VOI LUT Sequence (rare but authoritative) beats a percentile guess.
        if "VOILUTSequence" in ds:
            try:
                voi = apply_voi_lut(raw, ds).astype(np.float32)
                lo, hi = float(voi.min()), float(voi.max())
                normalized = (voi - lo) / (hi - lo if hi > lo else 1.0)
                out = _finish(ds, normalized, modality, photometric, invert, max_px)
                return RenderResult(
                    image=out.image, window_center=None, window_width=None,
                    window_source="voi_lut", value_min=lo, value_max=hi, modality=modality,
                    photometric=photometric, inverted=out.inverted, units=units, frames=n_frames,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("VOI LUT sequence failed: %s", exc)
        wc, ww = auto_window(values)
        source = "auto"

    # --- step 3: apply the linear window ---
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    normalized = np.clip((values - lo) / (hi - lo if hi > lo else 1.0), 0.0, 1.0)

    out = _finish(ds, normalized, modality, photometric, invert, max_px)
    return RenderResult(
        image=out.image,
        window_center=float(wc),
        window_width=float(ww),
        window_source=source,
        value_min=float(values.min()),
        value_max=float(values.max()),
        modality=modality,
        photometric=photometric,
        inverted=out.inverted,
        units=units,
        frames=n_frames,
    )


@dataclass
class _Finished:
    image: Image.Image
    inverted: bool


def _finish(
    ds: pydicom.Dataset,
    normalized: np.ndarray,
    modality: str,
    photometric: str,
    invert: bool | None,
    max_px: int,
) -> _Finished:
    """Presentation transform + 8-bit conversion + resize."""
    # MONOCHROME1 stores "low value = white". Flip it unless the caller overrides.
    auto_invert = photometric == "MONOCHROME1"
    inverted = auto_invert if invert is None else bool(invert)
    if auto_invert != inverted:
        pass  # caller explicitly disagreed with the header; honour the caller
    if inverted:
        normalized = 1.0 - normalized

    img8 = (normalized * 255.0).round().astype(np.uint8)
    img = Image.fromarray(img8, mode="L")

    # Non-square pixels (common in MR localisers / US) need aspect correction.
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is not None and len(spacing) == 2:
        try:
            row_mm, col_mm = float(spacing[0]), float(spacing[1])
            if row_mm > 0 and col_mm > 0:
                ratio = row_mm / col_mm
                if abs(ratio - 1.0) > 0.02:
                    w, h = img.size
                    img = img.resize((w, max(1, round(h * ratio))), Image.LANCZOS)
        except (TypeError, ValueError):
            pass

    return _Finished(image=_resize(img, max_px), inverted=inverted)
