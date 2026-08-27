"""Radiology prompt templates.

MedGemma is instruction-tuned but not a reporting product: the prompt does most of the
work. These templates are the ones worth benchmarking against — a free-text "describe
this" run tells you far less about clinical usefulness than a structured report or a
forced binary call you can actually score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RADIOLOGIST_SYSTEM = "You are an expert radiologist."
CAUTIOUS_SYSTEM = (
    "You are an expert radiologist. Report only what is visible in the image. "
    "If a finding is uncertain, say so explicitly rather than committing. "
    "Do not invent prior studies, clinical history, or measurements you cannot make."
)


@dataclass
class PromptTemplate:
    key: str
    label: str
    label_uz: str
    description: str
    system: str
    template: str
    modalities: tuple[str, ...] = ()          # empty = any
    output: str = "text"                       # "text" | "json"
    multi_image: bool = False
    scoreable: bool = False                    # produces a machine-comparable answer
    variables: tuple[str, ...] = field(default_factory=tuple)
    internal: bool = False                     # used by batch runs, hidden from the picker


TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        key="cxr_report",
        label="Chest X-ray — full report",
        label_uz="Ko'krak qafasi rentgeni — to'liq xulosa",
        description="FINDINGS + IMPRESSION, the format MedGemma was tuned to produce.",
        system=RADIOLOGIST_SYSTEM,
        template=(
            "{context}\n\n"
            "Write a radiology report for this chest X-ray.\n"
            "Use exactly two sections:\n"
            "FINDINGS: systematic description (airway, lungs, pleura, heart and mediastinum, "
            "bones and soft tissues, lines and devices).\n"
            "IMPRESSION: numbered list of the clinically significant conclusions."
        ),
        modalities=("CR", "DX", "DR"),
    ),
    PromptTemplate(
        key="findings_only",
        label="Findings only",
        label_uz="Faqat topilmalar",
        description="Description without an impression — useful to isolate perception from reasoning.",
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "Describe the radiological findings in this image. Do not give an impression, "
            "differential, or management advice — findings only."
        ),
    ),
    PromptTemplate(
        key="normal_abnormal",
        label="Normal vs abnormal (binary)",
        label_uz="Norma / patologiya (binar)",
        description="Forced binary call with confidence — the cleanest thing to compute sensitivity and specificity on.",
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "Is this study normal or abnormal?\n"
            "Answer in strict JSON with no other text:\n"
            '{{"verdict": "normal" | "abnormal", "confidence": 0.0-1.0, '
            '"key_finding": "<the single most important finding, or null if normal>", '
            '"reasoning": "<two sentences maximum>"}}'
        ),
        output="json",
        scoreable=True,
    ),
    PromptTemplate(
        key="structured_report",
        label="Structured report (JSON)",
        label_uz="Strukturali xulosa (JSON)",
        description="Machine-comparable findings list — the template to use for batch accuracy runs.",
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "Analyse this study and reply with strict JSON only, no markdown fence:\n"
            "{{\n"
            '  "study_quality": {{"adequate": true|false, "limitations": "<text or null>"}},\n'
            '  "findings": [\n'
            '    {{"finding": "<short name>", "location": "<anatomic location>", '
            '"severity": "mild"|"moderate"|"severe", "confidence": 0.0-1.0}}\n'
            "  ],\n"
            '  "impression": ["<numbered conclusion>", "..."],\n'
            '  "urgency": "routine"|"urgent"|"critical",\n'
            '  "recommended_followup": "<text or null>"\n'
            "}}"
        ),
        output="json",
        scoreable=True,
    ),
    PromptTemplate(
        key="differential",
        label="Differential diagnosis",
        label_uz="Qiyosiy tashxis",
        description="Ranked differential with the imaging feature supporting each entry.",
        system=RADIOLOGIST_SYSTEM,
        template=(
            "{context}\n\n"
            "Give a ranked differential diagnosis for the imaging findings. For each entry give: "
            "diagnosis, the specific imaging feature that supports it, and the feature that would "
            "argue against it. Maximum five entries, most likely first."
        ),
    ),
    PromptTemplate(
        key="question",
        label="Specific question (VQA)",
        label_uz="Aniq savol (VQA)",
        description="Ask one targeted question — how MedGemma is usually benchmarked.",
        system=CAUTIOUS_SYSTEM,
        template="{context}\n\n{question}",
        variables=("question",),
    ),
    PromptTemplate(
        key="localize",
        label="Localize abnormality (bounding box)",
        label_uz="Patologiyani belgilash (bounding box)",
        description=(
            "Topilmani rasmda ramka bilan belgilaydi. Ishonchli natija uchun "
            "MedGemma 1.5 kerak (IoU 38.0 vs 3.1)."
        ),
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "Identify every abnormality you can see and give a bounding box for each.\n"
            "Reply with strict JSON only, no markdown fence and no commentary:\n"
            '{{"detections": [{{"label": "<finding>", "box_2d": [ymin, xmin, ymax, xmax], '
            '"image_index": <1-based number of the image>, "confidence": 0.0-1.0}}]}}\n'
            "Box coordinates are normalized to 0-1000, in the order ymin, xmin, ymax, xmax. "
            "Make each box tight around the finding — do not return a box covering the whole "
            "image. Return an empty list if there is no abnormality."
        ),
        output="json",
        scoreable=True,
        multi_image=True,
    ),
    PromptTemplate(
        key="report_with_boxes",
        label="Report + localization",
        label_uz="Xulosa + patologiyani belgilash",
        description=(
            "Avval matnli xulosa, so'ng har bir topilma uchun ramka. "
            "Radiologik ish uchun eng amaliy variant."
        ),
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "Analyse this study in two parts.\n\n"
            "PART 1 — the report, as plain text:\n"
            "FINDINGS: systematic description.\n"
            "IMPRESSION: numbered conclusions.\n\n"
            "PART 2 — localization. After the report, on its own line, output the marker "
            "DETECTIONS: followed by strict JSON on the next line:\n"
            '{{"detections": [{{"label": "<finding>", "box_2d": [ymin, xmin, ymax, xmax], '
            '"image_index": <1-based image number>, "confidence": 0.0-1.0}}]}}\n'
            "Include one entry for every abnormality named in the IMPRESSION that is visible "
            "on the image. Coordinates are normalized 0-1000 as ymin, xmin, ymax, xmax, and "
            "each box must be tight around the finding. Use an empty list if the study is normal."
        ),
        multi_image=True,
        scoreable=True,
    ),
    PromptTemplate(
        key="compare_prior",
        label="Compare with prior study",
        label_uz="Oldingi tekshiruv bilan solishtirish",
        description="Two or more images in temporal order; asks specifically for interval change.",
        system=RADIOLOGIST_SYSTEM,
        template=(
            "{context}\n\n"
            "The images are given in chronological order — the first is the earliest (prior) and "
            "the last is the current study.\n"
            "Report the interval change: what is new, what has progressed, what has improved, and "
            "what is unchanged. State explicitly if the studies are not comparable."
        ),
        multi_image=True,
    ),
    PromptTemplate(
        key="ct_slices",
        label="CT/MRI — review selected slices",
        label_uz="KT/MRT — tanlangan kesmalarni ko'rib chiqish",
        description="Several contiguous slices from one series, reviewed as a single volume.",
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "The images are consecutive slices from a single acquisition, in anatomical order.\n"
            "Review them as one volume. Describe any abnormality, state which slice number it is "
            "best seen on, and give its anatomic location. Note if the selected slices are "
            "insufficient to answer confidently."
        ),
        modalities=("CT", "MR", "PT"),
        multi_image=True,
    ),
    PromptTemplate(
        key="critical_triage",
        label="Critical finding triage",
        label_uz="Kritik topilma triaji",
        description="Screens only for time-critical findings — the realistic first integration point.",
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "Screen this study for time-critical findings only (for example: pneumothorax, "
            "free intraperitoneal air, intracranial haemorrhage, aortic dissection, "
            "malpositioned tube or line, large pulmonary embolism, bowel obstruction).\n"
            "Reply with strict JSON only:\n"
            '{{"critical_finding_present": true|false, "findings": ["<name>"], '
            '"confidence": 0.0-1.0, "rationale": "<two sentences maximum>"}}'
        ),
        output="json",
        scoreable=True,
    ),
    PromptTemplate(
        key="second_read",
        label="Second read against my report",
        label_uz="Mening xulosamga qarshi ikkinchi o'qish",
        description="Give the model your own report; it looks for disagreement and omissions.",
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "A radiologist reported this study as follows:\n"
            "---\n{my_report}\n---\n\n"
            "Acting as an independent second reader, state:\n"
            "1. Findings in the report that you can confirm on the image.\n"
            "2. Findings in the report that you cannot confirm.\n"
            "3. Findings visible on the image that the report does not mention.\n"
            "4. Whether any disagreement would change patient management."
        ),
        variables=("my_report",),
    ),
    # ---------------------------------------------------------------- batch templates
    PromptTemplate(
        key="series_review",
        label="Series review (batch)",
        label_uz="Seriyani ko'rib chiqish (batch)",
        description=(
            "Butun tekshiruvni tahlil qilishda har bir seriya uchun ishlatiladi. "
            "Seriya nomi va kesma raqamlari bilan javob beradi."
        ),
        system=CAUTIOUS_SYSTEM,
        template=(
            "{context}\n\n"
            "These images are representative slices from a single series of a larger study, "
            "given in anatomical order.\n"
            "Report only what this series shows:\n"
            "1. What anatomy and which plane/sequence this series covers.\n"
            "2. Any abnormality, with the slice number it is best seen on.\n"
            "3. Image quality problems (motion, artefact, incomplete coverage).\n"
            "Be brief — this is one part of a study that will be summarised later. "
            "Say 'No abnormality identified on these slices' when that is the case, and say "
            "so explicitly if the selected slices are insufficient."
        ),
        multi_image=True,
        internal=True,
    ),
    PromptTemplate(
        key="study_synthesis",
        label="Study synthesis (batch)",
        label_uz="Tekshiruv bo'yicha umumiy xulosa (batch)",
        description="Barcha seriya natijalarini bitta xulosaga birlashtiradi (matn, rasmsiz).",
        system=(
            "You are an expert radiologist writing the final report for a study, working from "
            "the per-series observations of a first reader. Do not invent findings that are "
            "not in those observations."
        ),
        template=(
            "{context}\n\n"
            "Below are the per-series observations from a first pass over this study:\n"
            "---\n{series_reports}\n---\n\n"
            "Write the consolidated report:\n"
            "FINDINGS: merge the observations across series. Where several series show the same "
            "abnormality, describe it once and name the series that show it best. Note any "
            "disagreement between series.\n"
            "IMPRESSION: numbered list of the clinically significant conclusions.\n"
            "LIMITATIONS: coverage gaps, image quality problems, or anything the selected slices "
            "could not answer.\n\n"
            "If the observations contain no abnormality, say so plainly rather than hedging."
        ),
        variables=("series_reports",),
        internal=True,
    ),
    PromptTemplate(
        key="custom",
        label="Custom prompt",
        label_uz="Erkin prompt",
        description="Raw prompt, sent as written.",
        system=RADIOLOGIST_SYSTEM,
        template="{question}",
        variables=("question",),
    ),
]

TEMPLATES_BY_KEY = {t.key: t for t in TEMPLATES}


def list_templates(modality: str = "", include_internal: bool = False) -> list[dict]:
    mod = (modality or "").upper()
    out = []
    for t in TEMPLATES:
        if t.internal and not include_internal:
            continue
        if t.modalities and mod and mod not in t.modalities:
            continue
        out.append(
            {
                "key": t.key,
                "label": t.label,
                "label_uz": t.label_uz,
                "description": t.description,
                "output": t.output,
                "multi_image": t.multi_image,
                "scoreable": t.scoreable,
                "internal": t.internal,
                "variables": list(t.variables),
                "modalities": list(t.modalities),
                "system": t.system,
                "template": t.template,
            }
        )
    return out


def build_prompt(key: str, context: str = "", **variables) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for template ``key``."""
    tpl = TEMPLATES_BY_KEY.get(key)
    if tpl is None:
        raise KeyError(f"Unknown prompt template: {key}")
    values = {"context": context.strip(), **{k: (v or "").strip() for k, v in variables.items()}}
    for name in tpl.variables:
        values.setdefault(name, "")
    try:
        text = tpl.template.format(**values)
    except KeyError as exc:  # missing variable -> surface it clearly
        raise KeyError(f"Template '{key}' needs variable {exc}") from exc
    return tpl.system, text.strip()
