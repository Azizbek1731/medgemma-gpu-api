"""DICOM ingest: safe ZIP extraction, file discovery, and Study/Series/Instance indexing.

The indexer only reads headers (``stop_before_pixels=True``) so a multi-thousand-slice
CT archive indexes in seconds; pixel data is pulled lazily by :mod:`app.rendering`.
"""

from __future__ import annotations

import logging
import math
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pydicom
from pydicom.errors import InvalidDicomError

log = logging.getLogger(__name__)

# SOP classes that carry no displayable pixel data.
NON_IMAGE_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.88.11": "Basic Text SR",
    "1.2.840.10008.5.1.4.1.1.88.22": "Enhanced SR",
    "1.2.840.10008.5.1.4.1.1.88.33": "Comprehensive SR",
    "1.2.840.10008.5.1.4.1.1.88.34": "Comprehensive 3D SR",
    "1.2.840.10008.5.1.4.1.1.11.1": "Grayscale Softcopy Presentation State",
    "1.2.840.10008.5.1.4.1.1.11.2": "Color Softcopy Presentation State",
    "1.2.840.10008.5.1.4.1.1.481.3": "RT Structure Set",
    "1.2.840.10008.5.1.4.1.1.481.5": "RT Plan",
    "1.2.840.10008.5.1.4.1.1.66.4": "Segmentation",
    "1.2.840.10008.5.1.4.1.1.104.1": "Encapsulated PDF",
    "1.2.840.10008.5.1.4.1.1.9.1.1": "12-lead ECG Waveform",
    "1.2.840.10008.5.1.4.1.1.66": "Raw Data",
}

_SKIP_NAMES = {"dicomdir", ".ds_store", "thumbs.db"}


# --------------------------------------------------------------------------------------
# dataclasses
# --------------------------------------------------------------------------------------


@dataclass
class InstanceMeta:
    path: str
    sop_instance_uid: str
    sop_class_uid: str = ""
    instance_number: int | None = None
    frames: int = 1
    rows: int = 0
    columns: int = 0
    sort_key: float = 0.0
    slice_location: float | None = None
    image_position: list[float] | None = None
    photometric: str = ""
    bits_stored: int = 0
    transfer_syntax: str = ""
    is_image: bool = True
    error: str = ""


@dataclass
class SeriesMeta:
    series_instance_uid: str
    series_number: int | None = None
    modality: str = ""
    series_description: str = ""
    body_part: str = ""
    protocol_name: str = ""
    laterality: str = ""
    view_position: str = ""
    contrast: str = ""
    slice_thickness: float | None = None
    instances: list[InstanceMeta] = field(default_factory=list)

    @property
    def is_image(self) -> bool:
        return any(i.is_image for i in self.instances)

    @property
    def frame_count(self) -> int:
        return sum(max(1, i.frames) for i in self.instances if i.is_image)


@dataclass
class StudyMeta:
    study_instance_uid: str
    study_date: str = ""
    study_time: str = ""
    study_description: str = ""
    accession_number: str = ""
    referring_physician: str = ""
    patient_id: str = ""
    patient_name: str = ""
    patient_birth_date: str = ""
    patient_sex: str = ""
    patient_age: str = ""
    institution: str = ""
    manufacturer: str = ""
    series: list[SeriesMeta] = field(default_factory=list)

    @property
    def modalities(self) -> list[str]:
        return sorted({s.modality for s in self.series if s.modality})


@dataclass
class IngestReport:
    total_files: int = 0
    dicom_files: int = 0
    skipped_files: int = 0
    unreadable_files: int = 0
    non_image_instances: int = 0
    nested_zips: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# ZIP extraction
# --------------------------------------------------------------------------------------


def _safe_extract(zf: zipfile.ZipFile, dest: Path, max_files: int) -> int:
    """Extract ``zf`` into ``dest``, refusing path traversal ("zip slip") entries."""
    dest = dest.resolve()
    count = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        count += 1
        if count > max_files:
            raise ValueError(f"ZIP contains more than {max_files} files; aborting.")
        target = (dest / info.filename).resolve()
        if not str(target).startswith(str(dest) + "/") and target != dest:
            log.warning("Skipping unsafe zip entry: %s", info.filename)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
    return count


def extract_archive(archive: Path, dest: Path, max_files: int = 100_000) -> int:
    """Extract a ZIP (recursively, for nested ZIPs) into ``dest``. Returns file count."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        total = _safe_extract(zf, dest, max_files)

    # Recurse one level into nested archives (common with PACS exports).
    nested = [p for p in dest.rglob("*.zip") if p.is_file()]
    for n in nested:
        sub = n.with_suffix("")
        try:
            with zipfile.ZipFile(n) as zf:
                total += _safe_extract(zf, sub, max_files)
            n.unlink()
        except Exception as exc:  # pragma: no cover - corrupt nested zip
            log.warning("Nested zip failed (%s): %s", n.name, exc)
    return total


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------


def looks_like_dicom(path: Path) -> bool:
    """Cheap magic-number test; also accepts headerless (Part 10-less) files."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(132)
    except OSError:
        return False
    if len(head) >= 132 and head[128:132] == b"DICM":
        return True
    # Some exports drop the 128-byte preamble. Look for a plausible group-0002/0008 tag.
    return len(head) >= 8 and head[:2] in (b"\x02\x00", b"\x08\x00")


def iter_candidate_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.lower() in _SKIP_NAMES:
            continue
        if p.name.startswith("._"):  # macOS AppleDouble sidecars
            continue
        yield p


# --------------------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------------------


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _s(ds: pydicom.Dataset, key: str, default: str = "") -> str:
    v = getattr(ds, key, None)
    if v is None:
        return default
    if isinstance(v, pydicom.multival.MultiValue):
        v = "\\".join(str(x) for x in v)
    return str(v).strip()


def _slice_sort_key(ds: pydicom.Dataset) -> tuple[float, list[float] | None, float | None]:
    """Order slices along the true scan axis when geometry is available."""
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    position = [float(x) for x in ipp] if ipp is not None and len(ipp) == 3 else None
    sloc = _f(getattr(ds, "SliceLocation", None))

    if position and iop is not None and len(iop) == 6:
        r = [float(x) for x in iop[:3]]
        c = [float(x) for x in iop[3:]]
        normal = [
            r[1] * c[2] - r[2] * c[1],
            r[2] * c[0] - r[0] * c[2],
            r[0] * c[1] - r[1] * c[0],
        ]
        norm = math.sqrt(sum(v * v for v in normal))
        if norm > 0:
            projection = sum(p * n for p, n in zip(position, normal)) / norm
            return projection, position, sloc

    if sloc is not None:
        return sloc, position, sloc

    inum = _i(getattr(ds, "InstanceNumber", None))
    return float(inum if inum is not None else 0), position, sloc


def index_directory(root: Path, report: IngestReport | None = None) -> list[StudyMeta]:
    """Walk ``root`` and build the Patient/Study/Series/Instance tree."""
    rep = report or IngestReport()
    studies: dict[str, StudyMeta] = {}
    series_index: dict[tuple[str, str], SeriesMeta] = {}

    for path in iter_candidate_files(root):
        rep.total_files += 1
        if not looks_like_dicom(path):
            rep.skipped_files += 1
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except (InvalidDicomError, OSError, Exception) as exc:  # noqa: BLE001
            rep.unreadable_files += 1
            if len(rep.errors) < 50:
                rep.errors.append(f"{path.name}: {exc}")
            continue

        sop_class = _s(ds, "SOPClassUID")
        study_uid = _s(ds, "StudyInstanceUID") or "NO_STUDY_UID"
        series_uid = _s(ds, "SeriesInstanceUID") or "NO_SERIES_UID"

        study = studies.get(study_uid)
        if study is None:
            study = StudyMeta(
                study_instance_uid=study_uid,
                study_date=_s(ds, "StudyDate"),
                study_time=_s(ds, "StudyTime"),
                study_description=_s(ds, "StudyDescription"),
                accession_number=_s(ds, "AccessionNumber"),
                referring_physician=_s(ds, "ReferringPhysicianName"),
                patient_id=_s(ds, "PatientID"),
                patient_name=_s(ds, "PatientName"),
                patient_birth_date=_s(ds, "PatientBirthDate"),
                patient_sex=_s(ds, "PatientSex"),
                patient_age=_s(ds, "PatientAge"),
                institution=_s(ds, "InstitutionName"),
                manufacturer=" ".join(
                    x for x in (_s(ds, "Manufacturer"), _s(ds, "ManufacturerModelName")) if x
                ),
            )
            studies[study_uid] = study

        sk = (study_uid, series_uid)
        series = series_index.get(sk)
        if series is None:
            series = SeriesMeta(
                series_instance_uid=series_uid,
                series_number=_i(getattr(ds, "SeriesNumber", None)),
                modality=_s(ds, "Modality"),
                series_description=_s(ds, "SeriesDescription"),
                body_part=_s(ds, "BodyPartExamined"),
                protocol_name=_s(ds, "ProtocolName"),
                laterality=_s(ds, "ImageLaterality") or _s(ds, "Laterality"),
                view_position=_s(ds, "ViewPosition"),
                contrast=_s(ds, "ContrastBolusAgent"),
                slice_thickness=_f(getattr(ds, "SliceThickness", None)),
            )
            series_index[sk] = series
            study.series.append(series)

        is_image = sop_class not in NON_IMAGE_SOP_CLASSES and hasattr(ds, "Rows")
        if not is_image:
            rep.non_image_instances += 1

        sort_key, position, sloc = _slice_sort_key(ds)
        try:
            tsyntax = str(ds.file_meta.TransferSyntaxUID)
        except Exception:  # noqa: BLE001
            tsyntax = ""

        series.instances.append(
            InstanceMeta(
                path=str(path.relative_to(root)),
                sop_instance_uid=_s(ds, "SOPInstanceUID"),
                sop_class_uid=sop_class,
                instance_number=_i(getattr(ds, "InstanceNumber", None)),
                frames=_i(getattr(ds, "NumberOfFrames", None)) or 1,
                rows=_i(getattr(ds, "Rows", None)) or 0,
                columns=_i(getattr(ds, "Columns", None)) or 0,
                sort_key=sort_key,
                slice_location=sloc,
                image_position=position,
                photometric=_s(ds, "PhotometricInterpretation"),
                bits_stored=_i(getattr(ds, "BitsStored", None)) or 0,
                transfer_syntax=tsyntax,
                is_image=is_image,
            )
        )
        rep.dicom_files += 1

    for study in studies.values():
        for series in study.series:
            series.instances.sort(key=lambda i: (i.sort_key, i.instance_number or 0))
        study.series.sort(key=lambda s: (s.series_number if s.series_number is not None else 9999))

    ordered = sorted(
        studies.values(), key=lambda s: (s.study_date, s.study_time), reverse=True
    )
    return ordered


def studies_to_json(studies: list[StudyMeta]) -> list[dict]:
    out = []
    for st in studies:
        d = asdict(st)
        d["modalities"] = st.modalities
        for s_dict, s_obj in zip(d["series"], st.series):
            s_dict["frame_count"] = s_obj.frame_count
            s_dict["is_image"] = s_obj.is_image
            s_dict["instance_count"] = len(s_obj.instances)
        out.append(d)
    return out
