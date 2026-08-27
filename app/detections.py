"""Parse bounding boxes out of model output so findings can be drawn on the image.

There is no single documented box format for MedGemma, and instruction-tuned models are
inconsistent under the best conditions, so this parser is deliberately permissive: it
accepts fenced or bare JSON, several key spellings, PaliGemma-style ``<locNNNN>`` tokens,
and coordinates expressed 0-1000, 0-1, or in pixels.

Everything is normalized to floats in 0..1 as ``(y0, x0, y1, x1)``.

Note on trust: localization is a MedGemma 1.5 capability (region-of-interest IoU 38.0 vs
3.1 for MedGemma 1 4B). Boxes from an older checkpoint parse fine and mean almost
nothing — :func:`model_supports_localization` exists so the UI can say so.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

# `<loc0123>` tokens, PaliGemma / Gemma detection convention.
_LOC_TOKEN_RE = re.compile(r"<loc(\d{1,4})>")
_LOC_GROUP_RE = re.compile(
    r"(?:<loc(\d{1,4})>){4}\s*([A-Za-z][\w \-/']*)"
)

_BOX_KEYS = ("box_2d", "box", "bbox", "bounding_box", "boundingbox", "box2d", "coordinates")
_LABEL_KEYS = ("label", "name", "finding", "class", "category", "description")
_CONF_KEYS = ("confidence", "score", "probability", "prob")
_INDEX_KEYS = ("image_index", "image", "image_number", "slice", "slice_number", "frame")
_LIST_KEYS = ("detections", "boxes", "findings", "objects", "regions", "results")


@dataclass
class Detection:
    label: str
    y0: float
    x0: float
    y1: float
    x1: float
    confidence: float | None = None
    image_index: int = 0  # 0-based position within the images sent for this run

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def area(self) -> float:
        return max(0.0, self.y1 - self.y0) * max(0.0, self.x1 - self.x0)


# --------------------------------------------------------------------------------------
# coordinate normalization
# --------------------------------------------------------------------------------------


class RejectedBox(Exception):
    """A box was syntactically parseable but geometrically useless."""

    def __init__(self, reason: str, raw: object):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


def _normalize_coords(values: list[float]) -> tuple[float, float, float, float] | None:
    """Map four raw numbers onto 0..1 ``(y0, x0, y1, x1)``.

    Raises :class:`RejectedBox` when the numbers parse but describe nothing usable —
    a zero-area point or a box covering the whole image. Those are worth reporting
    rather than silently dropping: they are the signature failure of a checkpoint that
    was never trained to localize.
    """
    if len(values) != 4:
        return None
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if any(n != n for n in nums):  # NaN
        return None

    peak = max(abs(n) for n in nums)
    if peak <= 1.5:
        scale = 1.0                       # already 0-1
    elif peak <= 1000.0:
        scale = 1000.0                    # Gemma convention
    else:
        scale = float(peak)               # pixels; normalize against the largest value
    nums = [n / scale for n in nums]

    y0, x0, y1, x1 = nums
    # Some models emit xyxy instead of yxyx. If the yxyx reading is degenerate but the
    # xyxy reading is not, take the other interpretation.
    if (y1 <= y0 or x1 <= x0) and nums[2] > nums[0] and nums[3] > nums[1]:
        x0, y0, x1, y1 = nums

    y0, y1 = sorted((y0, y1))
    x0, x1 = sorted((x0, x1))
    y0, x0 = max(0.0, y0), max(0.0, x0)
    y1, x1 = min(1.0, y1), min(1.0, x1)
    if y1 - y0 <= 0.001 or x1 - x0 <= 0.001:
        raise RejectedBox("nol o'lchamli ramka (nuqta)", list(values))
    if (y1 - y0) * (x1 - x0) > 0.92:
        raise RejectedBox("butun rasmni qamragan ramka", list(values))
    return y0, x0, y1, x1


def _coerce_confidence(value: object) -> float | None:
    try:
        c = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if c > 1.0:
        c = c / 100.0 if c <= 100.0 else 1.0
    return max(0.0, min(1.0, c))


def _coerce_index(value: object) -> int:
    try:
        i = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    # Prompts ask for 1-based image numbers; store 0-based.
    return max(0, i - 1) if i > 0 else 0


# --------------------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------------------


def _json_candidates(text: str) -> list[object]:
    """Yield every JSON object/array embedded in ``text``, fenced or bare."""
    out: list[object] = []
    for fenced in re.findall(r"```(?:json)?\s*(.+?)```", text, re.S):
        try:
            out.append(json.loads(fenced.strip()))
        except json.JSONDecodeError:
            pass

    # Brace/bracket matching, so prose around the JSON does not break parsing.
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closer and depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        out.append(json.loads(text[start : i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return out


def _try_coords(
    raw: object, rejects: list[dict] | None
) -> tuple[float, float, float, float] | None:
    try:
        return _normalize_coords(raw)  # type: ignore[arg-type]
    except RejectedBox as rb:
        if rejects is not None:
            rejects.append({"reason": rb.reason, "raw": rb.raw})
        return None


def _detections_from_obj(
    obj: object, inherited_index: int = 0, rejects: list[dict] | None = None
) -> list[Detection]:
    found: list[Detection] = []

    if isinstance(obj, list):
        for item in obj:
            found.extend(_detections_from_obj(item, inherited_index, rejects))
        return found

    if not isinstance(obj, dict):
        return found

    # Nested containers, e.g. {"detections": [...]} or {"image_index": 2, "boxes": [...]}
    index = inherited_index
    for key in _INDEX_KEYS:
        if key in obj:
            index = _coerce_index(obj[key])
            break

    for key in _LIST_KEYS:
        value = obj.get(key)
        if isinstance(value, list):
            found.extend(_detections_from_obj(value, index, rejects))

    raw_box = None
    for key in _BOX_KEYS:
        if key in obj:
            raw_box = obj[key]
            break
    if raw_box is None:
        return found

    # A box key may itself hold a list of boxes.
    if isinstance(raw_box, list) and raw_box and isinstance(raw_box[0], (list, dict)):
        for sub in raw_box:
            if isinstance(sub, dict):
                found.extend(_detections_from_obj(sub, index, rejects))
            else:
                coords = _try_coords(sub, rejects)
                if coords:
                    found.append(_make(obj, coords, index))
        return found

    if isinstance(raw_box, dict):  # {"ymin": .., "xmin": .., ...}
        keys = {k.lower(): v for k, v in raw_box.items()}
        ordered = [keys.get(k) for k in ("ymin", "xmin", "ymax", "xmax")]
        if all(v is not None for v in ordered):
            coords = _try_coords(ordered, rejects)
            if coords:
                found.append(_make(obj, coords, index))
        return found

    if isinstance(raw_box, str):
        raw_box = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", raw_box)]

    coords = _try_coords(raw_box, rejects) if isinstance(raw_box, list) else None
    if coords:
        found.append(_make(obj, coords, index))
    return found


def _make(obj: dict, coords: tuple[float, float, float, float], index: int) -> Detection:
    label = ""
    for key in _LABEL_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            label = value.strip()
            break
    confidence = None
    for key in _CONF_KEYS:
        if key in obj:
            confidence = _coerce_confidence(obj[key])
            break
    y0, x0, y1, x1 = coords
    return Detection(
        label=label or "finding",
        y0=y0, x0=x0, y1=y1, x1=x1,
        confidence=confidence,
        image_index=index,
    )


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


@dataclass
class ParseResult:
    detections: list[Detection]
    rejected: list[dict]  # [{reason, raw}] — parseable but unusable geometry

    @property
    def rejected_summary(self) -> str:
        if not self.rejected:
            return ""
        reasons = sorted({r["reason"] for r in self.rejected})
        return (
            f"Model {len(self.rejected)} ta ramka qaytardi, lekin ular yaroqsiz: "
            + ", ".join(reasons)
            + "."
        )


def parse_detailed(text: str, image_count: int = 1) -> ParseResult:
    """Extract boxes, and report boxes that parsed but describe nothing usable."""
    if not text or not text.strip():
        return ParseResult([], [])

    rejects: list[dict] = []
    found: list[Detection] = []
    for obj in _json_candidates(text):
        found.extend(_detections_from_obj(obj, 0, rejects))

    # PaliGemma-style `<loc0123><loc0456><loc0789><loc0012> pneumothorax`
    if not found and "<loc" in text:
        for match in _LOC_GROUP_RE.finditer(text):
            nums = [float(n) for n in _LOC_TOKEN_RE.findall(match.group(0))]
            coords = _try_coords([n / 1024 * 1000 for n in nums[:4]], rejects)
            if coords:
                y0, x0, y1, x1 = coords
                found.append(
                    Detection(label=(match.group(2) or "finding").strip(),
                              y0=y0, x0=x0, y1=y1, x1=x1)
                )

    unique: list[Detection] = []
    for d in found:
        if d.image_index >= max(1, image_count):
            d.image_index = 0
        key = (round(d.y0, 3), round(d.x0, 3), round(d.y1, 3), round(d.x1, 3), d.image_index)
        if any(
            (round(u.y0, 3), round(u.x0, 3), round(u.y1, 3), round(u.x1, 3), u.image_index) == key
            for u in unique
        ):
            continue
        unique.append(d)

    # The same JSON is often matched twice (fenced and bare), so dedupe the rejects too.
    seen_rejects: list[dict] = []
    for r in rejects:
        if r not in seen_rejects:
            seen_rejects.append(r)
    return ParseResult(unique, seen_rejects)


def parse(text: str, image_count: int = 1) -> list[Detection]:
    """Extract every usable bounding box from a model response."""
    return parse_detailed(text, image_count).detections


def to_json(detections: list[Detection]) -> list[dict]:
    return [d.to_json() for d in detections]


# --------------------------------------------------------------------------------------
# model capability
# --------------------------------------------------------------------------------------

# Region-of-interest IoU on Chest ImaGenome, from the MedGemma 1.5 model card:
#   Gemma 3 4B 5.7 · MedGemma 1 4B 3.1 · MedGemma 1 27B 16.0 · MedGemma 1.5 4B 38.0
LOCALIZATION_MODELS = ("medgemma-1.5",)

# The model card scopes localization narrowly: "Bounding box-based localization of
# anatomical features and findings in chest X-rays", benchmarked only on Chest
# ImaGenome. Nothing claims CT, MRI, ultrasound, or anything else — so boxes on those
# modalities are out-of-distribution however good the checkpoint is.
LOCALIZATION_MODALITIES = ("CR", "DX", "DR", "RG")


def model_supports_localization(model_id: str) -> bool:
    return any(tag in (model_id or "").lower() for tag in LOCALIZATION_MODELS)


def modality_supports_localization(modality: str) -> bool:
    return (modality or "").upper() in LOCALIZATION_MODALITIES


def localization_warning(model_id: str, modality: str = "") -> str:
    """Every reason the boxes about to be drawn might not be trustworthy."""
    notes: list[str] = []
    if not model_supports_localization(model_id):
        notes.append(
            "Bu model bounding box uchun o'rgatilmagan — ramkalar ishonchsiz bo'ladi "
            "(MedGemma 1 4B: IoU 3.1, MedGemma 1.5 4B: IoU 38.0). "
            "Ishonchli lokalizatsiya uchun medgemma-1.5 kerak."
        )
    if modality and not modality_supports_localization(modality):
        notes.append(
            f"MedGemma lokalizatsiyasi faqat ko'krak rentgeni (CR/DX) uchun o'rgatilgan "
            f"va faqat shunda o'lchangan (Chest ImaGenome). {modality.upper()} uchun "
            "ramkalar tasdiqlanmagan — natijani tekshirmasdan ishonmang."
        )
    return " ".join(notes)
