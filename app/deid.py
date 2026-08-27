"""PHI handling.

Two separate jobs, deliberately kept apart:

* **Audit** — tell the radiologist exactly what identifiers are sitting in the archive.
  Nothing is hidden or silently rewritten; you cannot reason about disclosure risk you
  cannot see.
* **Egress control** — build the prompt-side clinical context from a strict allow-list,
  and (optionally) write de-identified DICOM copies following the PS3.15 Basic
  Application Level Confidentiality Profile.

Note that the model only ever receives a rendered PNG plus text, so header tags do not
leave this machine by themselves. The residual risks are burned-in pixel annotations and
free-text the user types into the prompt box — both are surfaced explicitly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import pydicom

# --------------------------------------------------------------------------------------
# PS3.15 Table E.1-1 — the identifiers that matter in practice.
# --------------------------------------------------------------------------------------

# Removed entirely.
REMOVE_TAGS: tuple[str, ...] = (
    "PatientName", "PatientID", "OtherPatientIDs", "OtherPatientIDsSequence",
    "OtherPatientNames", "PatientBirthDate", "PatientBirthTime", "PatientBirthName",
    "PatientMotherBirthName", "PatientAddress", "PatientTelephoneNumbers",
    "PatientTelecomInformation", "PatientInsurancePlanCodeSequence",
    "PatientReligiousPreference", "CountryOfResidence", "RegionOfResidence",
    "MilitaryRank", "BranchOfService", "MedicalRecordLocator", "IssuerOfPatientID",
    "ReferringPhysicianName", "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers", "ReferringPhysicianIdentificationSequence",
    "PerformingPhysicianName", "PerformingPhysicianIdentificationSequence",
    "PhysiciansOfRecord", "PhysiciansOfRecordIdentificationSequence",
    "PhysiciansReadingStudyName", "PhysiciansReadingStudyIdentificationSequence",
    "NameOfPhysiciansReadingStudy", "OperatorsName", "OperatorIdentificationSequence",
    "RequestingPhysician", "RequestingService", "ScheduledPerformingPhysicianName",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "InstitutionCodeSequence", "StationName",
    "AccessionNumber", "StudyID", "RequestAttributesSequence",
    "AdmissionID", "IssuerOfAdmissionID", "AdmittingDiagnosesDescription",
    "AdmittingDiagnosesCodeSequence", "PatientComments", "AdditionalPatientHistory",
    "OccupationalExposureOrRadiationDose", "Occupation",
    "ResponsiblePerson", "ResponsibleOrganization",
    "PersonName", "ContentCreatorName", "VerifyingObserverName",
    "VerifyingObserverIdentificationCodeSequence", "VerifyingOrganization",
    "ReviewerName", "AuthorObserverSequence", "ParticipantSequence",
    "OrderCallbackPhoneNumber", "OrderEnteredBy", "OrderEntererLocation",
    "PerformedProcedureStepID", "PerformedLocation", "ScheduledStudyLocation",
    "CurrentPatientLocation", "DeviceSerialNumber", "PlateID", "DetectorID",
    "GantryID", "CassetteID", "StorageMediaFileSetID",
)

# Zeroed (kept as an empty value so downstream tooling that requires the tag still works).
BLANK_TAGS: tuple[str, ...] = (
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "AcquisitionDateTime",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
    "PerformedProcedureStepStartDate", "PerformedProcedureStepStartTime",
    "ScheduledProcedureStepStartDate", "ScheduledProcedureStepStartTime",
    "InstanceCreationDate", "InstanceCreationTime", "DateOfLastCalibration",
)

# Kept — clinically necessary, low re-identification risk on their own.
KEEP_TAGS: tuple[str, ...] = (
    "Modality", "BodyPartExamined", "PatientSex", "PatientAge", "ViewPosition",
    "ImageLaterality", "Laterality", "StudyDescription", "SeriesDescription",
    "ProtocolName", "SliceThickness", "KVP", "ExposureTime", "XRayTubeCurrent",
    "ContrastBolusAgent", "MagneticFieldStrength", "ScanningSequence",
    "SequenceVariant", "RepetitionTime", "EchoTime", "PatientPosition",
)

# Modalities where demographics are routinely burned into the pixels themselves.
BURN_IN_RISK_MODALITIES = {"US", "SC", "XA", "RF", "OT", "NM", "DX_SECONDARY"}

_DATE_RE = re.compile(r"\b(19|20)\d{2}[-/.]?\d{2}[-/.]?\d{2}\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{6,}\b")
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3}[\s-]?\d{2,4}")


@dataclass
class PHIFinding:
    tag: str
    keyword: str
    value: str
    severity: str  # "direct" | "quasi" | "indirect"


DIRECT = {
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames",
    "MedicalRecordLocator", "AccessionNumber",
}
QUASI = {
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "PatientAge",
    "InstitutionName", "InstitutionAddress", "ReferringPhysicianName",
    "PerformingPhysicianName", "StationName", "DeviceSerialNumber", "StudyID",
}


def _severity(keyword: str) -> str:
    if keyword in DIRECT:
        return "direct"
    if keyword in QUASI:
        return "quasi"
    return "indirect"


def audit_dataset(ds: pydicom.Dataset) -> list[PHIFinding]:
    """List every identifier actually present (non-empty) in ``ds``."""
    findings: list[PHIFinding] = []
    watched = set(REMOVE_TAGS) | set(BLANK_TAGS)
    for keyword in sorted(watched):
        if keyword not in ds:
            continue
        elem = ds[keyword]
        value = elem.value
        if value in (None, "", []):
            continue
        text = str(value)
        if len(text) > 120:
            text = text[:117] + "..."
        findings.append(
            PHIFinding(
                tag=str(elem.tag),
                keyword=keyword,
                value=text,
                severity=_severity(keyword),
            )
        )
    return findings


def burned_in_risk(ds: pydicom.Dataset) -> tuple[bool, str]:
    """Does this image probably carry demographics rendered into the pixels?"""
    flag = str(getattr(ds, "BurnedInAnnotation", "") or "").upper()
    modality = str(getattr(ds, "Modality", "") or "").upper()
    if flag == "YES":
        return True, "BurnedInAnnotation = YES (header declares burned-in text)"
    if flag == "NO":
        return False, "BurnedInAnnotation = NO"
    if modality in BURN_IN_RISK_MODALITIES:
        return True, f"{modality} images commonly carry burned-in demographics; header is silent"
    if str(getattr(ds, "ConversionType", "") or ""):
        return True, "Secondary capture (ConversionType present) — check for burned-in text"
    return False, "No burned-in annotation indicated"


def pseudonymize(value: str, salt: str = "medgemma-lab") -> str:
    """Stable pseudonym so the same patient stays linkable across studies."""
    if not value:
        return ""
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return f"ANON-{digest[:12].upper()}"


def deidentify_dataset(
    ds: pydicom.Dataset,
    *,
    pseudonym_id: bool = True,
    keep_study_year: bool = False,
) -> pydicom.Dataset:
    """Apply the removal / blanking profile in place and return ``ds``."""
    original_id = str(getattr(ds, "PatientID", "") or "")

    for keyword in REMOVE_TAGS:
        if keyword in ds:
            del ds[keyword]

    for keyword in BLANK_TAGS:
        if keyword in ds:
            elem = ds[keyword]
            if keep_study_year and keyword.endswith("Date") and elem.value:
                elem.value = f"{str(elem.value)[:4]}0101"
            else:
                elem.value = ""

    if pseudonym_id:
        ds.PatientID = pseudonymize(original_id)
        ds.PatientName = ds.PatientID

    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "AviRadiology AI / PS3.15 Basic Profile (subset)"

    # Private tags can carry anything a vendor felt like storing.
    ds.remove_private_tags()
    return ds


def safe_clinical_context(
    *,
    modality: str = "",
    body_part: str = "",
    patient_age: str = "",
    patient_sex: str = "",
    view_position: str = "",
    laterality: str = "",
    series_description: str = "",
    study_description: str = "",
    contrast: str = "",
    extra: str = "",
) -> str:
    """Build the prompt's clinical context from an allow-list. No free-form passthrough."""
    parts: list[str] = []
    age = _normalize_age(patient_age)
    sex = {"M": "male", "F": "female"}.get((patient_sex or "").upper().strip(), "")
    if age or sex:
        parts.append(f"Patient: {' '.join(x for x in (age, sex) if x)}".strip())
    if modality:
        desc = modality
        if contrast:
            desc += " with contrast"
        parts.append(f"Modality: {desc}")
    if body_part:
        parts.append(f"Body part: {body_part.title()}")
    if view_position:
        parts.append(f"View: {view_position}")
    if laterality:
        parts.append(f"Laterality: {laterality}")
    for label, value in (("Study", study_description), ("Series", series_description)):
        if value and not _looks_identifying(value):
            parts.append(f"{label}: {value}")
    if extra:
        parts.append(extra.strip())
    return "\n".join(parts)


def _normalize_age(raw: str) -> str:
    """DICOM AS values look like '045Y' / '018M'."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    m = re.fullmatch(r"(\d{1,3})([DWMY])", raw.upper())
    if not m:
        return raw
    n, unit = int(m.group(1)), m.group(2)
    word = {"D": "day", "W": "week", "M": "month", "Y": "year"}[unit]
    return f"{n}-{word}-old"


def _looks_identifying(text: str) -> bool:
    return bool(_DATE_RE.search(text) or _LONG_DIGITS_RE.search(text))


def scan_free_text(text: str) -> list[str]:
    """Warn about identifiers a user may have pasted into the prompt box."""
    warnings: list[str] = []
    if _DATE_RE.search(text):
        warnings.append("Matnda to'liq sana bor (tug'ilgan sana bo'lishi mumkin).")
    if _LONG_DIGITS_RE.search(text):
        warnings.append("Matnda 6+ raqamli identifikator bor (MRN / passport bo'lishi mumkin).")
    if _PHONE_RE.search(text):
        warnings.append("Matnda telefon raqamiga o'xshash ketma-ketlik bor.")
    return warnings


def summarize_findings(findings: Iterable[PHIFinding]) -> dict:
    findings = list(findings)
    return {
        "total": len(findings),
        "direct": sum(1 for f in findings if f.severity == "direct"),
        "quasi": sum(1 for f in findings if f.severity == "quasi"),
        "indirect": sum(1 for f in findings if f.severity == "indirect"),
        "findings": [f.__dict__ for f in findings],
    }
