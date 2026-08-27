#!/usr/bin/env python3
"""
AviRadiology AI GPU API — uchidan-uchiga tekshiruv.

Yangi mashinada servis TO'G'RI ishlayotganini tasdiqlaydi: autentifikatsiya,
DICOM pipeline, GPU inferens va SSE oqimi.

Ishlatish:
    python tools/smoke_test.py                      # lokal, .env dan kalit
    MG_URL=https://ai.example.uz MG_API_KEY=... python tools/smoke_test.py

Sun'iy DICOM ishlatiladi — HECH QANDAY bemor ma'lumoti kerak emas.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

BASE = os.getenv("MG_URL", "http://127.0.0.1:8077").rstrip("/")
KEY = os.getenv("MG_API_KEY", "")

if not KEY:
    # .env dan o'qishga urinamiz
    try:
        for line in open(os.path.join(os.path.dirname(__file__), "..", ".env")):
            if line.startswith("MG_API_KEY="):
                KEY = line.split("=", 1)[1].strip()
                break
    except OSError:
        pass

OK, FAIL = "\033[32m✓\033[0m", "\033[31m✗\033[0m"
_failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _failures
    print(f"  {OK if cond else FAIL} {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        _failures += 1


def req(path: str, *, method="GET", data=None, headers=None, auth=True, timeout=120):
    h = dict(headers or {})
    if auth and KEY:
        h["X-API-Key"] = KEY
    r = urllib.request.Request(f"{BASE}{path}", data=data, method=method, headers=h)
    return urllib.request.urlopen(r, timeout=timeout)


def synthetic_dicom_zip() -> bytes:
    """Sun'iy 1 kadrli DICOM (CR) yasab, ZIP qilib qaytaradi."""
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    px = np.zeros((256, 256), dtype=np.uint16)
    px[60:200, 80:180] = 2000
    px[110:150, 110:150] = 3500          # "topilma"

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = "TEST^SYNTHETIC"
    ds.PatientID = "SMOKE-001"
    ds.PatientSex = "M"
    ds.PatientAge = "055Y"
    ds.Modality = "CR"
    ds.StudyDescription = "SMOKE TEST"
    ds.SeriesDescription = "CHEST PA"
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = "PA"
    ds.StudyDate = "20260101"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = px.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = px.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    dcm = io.BytesIO()
    pydicom.dcmwrite(dcm, ds, enforce_file_format=True)

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("CR_0001.dcm", dcm.getvalue())
    return zbuf.getvalue()


def multipart(filename: str, payload: bytes) -> tuple[bytes, str]:
    b = "----smoke" + str(int(time.time()))
    body = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: application/zip\r\n\r\n"
    ).encode() + payload + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def main() -> int:
    print(f"\nAviRadiology AI GPU API — tekshiruv: {BASE}\n")

    # 1. Servis javob beradimi (autentifikatsiyasiz ochiq yo'l)
    print("1 · Ulanish")
    try:
        with req("/healthz", auth=False, timeout=15) as r:
            check("/healthz javob berdi", r.status == 200)
    except Exception as e:  # noqa: BLE001
        check("/healthz javob berdi", False, str(e))
        print("\n  Servis ishlamayapti — qolgan tekshiruvlar o'tkazilmadi.\n")
        return 1

    # 2. Autentifikatsiya — kalitsiz KIRIB BO'LMASLIGI shart
    print("\n2 · Autentifikatsiya")
    try:
        req("/api/engines", auth=False, timeout=15)
        check("kalitsiz so'rov bloklandi", False, "401 kutilgandi, o'tib ketdi!")
    except urllib.error.HTTPError as e:
        check("kalitsiz so'rov bloklandi", e.code in (401, 503), f"HTTP {e.code}")
    try:
        r = urllib.request.Request(f"{BASE}/api/engines", headers={"X-API-Key": "notarealkey"})
        urllib.request.urlopen(r, timeout=15)
        check("noto'g'ri kalit bloklandi", False, "401 kutilgandi")
    except urllib.error.HTTPError as e:
        check("noto'g'ri kalit bloklandi", e.code in (401, 503), f"HTTP {e.code}")

    if not KEY:
        print("\n  MG_API_KEY berilmadi — qolgan tekshiruvlar o'tkazilmadi.\n")
        return 1

    # 3. Engine va GPU
    print("\n3 · Model va GPU")
    with req("/api/engines", timeout=30) as r:
        eng = json.load(r)
    engines = eng.get("engines", [])
    avail = [e for e in engines if e.get("available")]
    check("to'g'ri kalit ishladi", True)
    check("mavjud engine bor", bool(avail), ", ".join(e["key"] for e in avail) or "yo'q")
    gpu = next((e for e in avail if e["key"] == "transformers"), None)
    if gpu:
        check("GPU (transformers) tayyor", True, gpu["default_model"])
    else:
        # Sababni KO'RSATAMIZ — operator nima qilishini bilishi kerak
        # ("torch o'rnatilmagan", "CUDA topilmadi", "model yuklab bo'lmadi"...).
        tr = next((e for e in engines if e["key"] == "transformers"), None)
        check("GPU (transformers) tayyor", False,
              (tr or {}).get("reason") or "transformers engine ro'yxatda yo'q")
        print("      → Servis MOCK rejimida ishlaydi: javob SOXTA, klinik "
              "ma'noga ega emas.")
    check("local_only yoqilgan (PHI himoyasi)", eng.get("local_only") is True)

    # 4. DICOM pipeline
    print("\n4 · DICOM pipeline")
    try:
        body, ctype = multipart("smoke.zip", synthetic_dicom_zip())
    except ImportError as e:
        check("sun'iy DICOM yasaldi", False, f"{e} (pydicom/numpy kerak)")
        return 1
    check("sun'iy DICOM yasaldi", True)

    with req("/api/upload?label=smoke", method="POST", data=body,
             headers={"Content-Type": ctype}, timeout=180) as r:
        up = json.load(r)
    uid = up["upload_id"]
    check("arxiv yuklandi", up["report"]["dicom_files"] == 1, f"upload_id={uid}")

    study = up["studies"][0]
    series = study["series"][0]
    q = (f"series_uid={series['series_instance_uid']}"
         f"&study_uid={study['study_instance_uid']}&frame=0")

    with req(f"/api/uploads/{uid}/frame.png?{q}&px=512", timeout=60) as r:
        png = r.read()
        wl = r.headers.get("X-Window-Center")
    check("kadr render qilindi", png[:8] == b"\x89PNG\r\n\x1a\n", f"{len(png)//1024} KB")
    check("X-Window-* sarlavhalari bor", wl is not None, f"center={wl}")

    with req(f"/api/uploads/{uid}/frame-info?{q}", timeout=60) as r:
        info = json.load(r)
    check("PHI audit ishladi", "phi" in info, f"{info.get('phi', {}).get('total')} teg topildi")
    check("klinik kontekst PHI'siz", "SMOKE-001" not in (info.get("clinical_context") or ""))

    # 5. Inferens (SSE)
    print("\n5 · AI inferens (SSE oqimi)")
    engine_key = gpu["key"] if gpu else "mock"
    payload = json.dumps({
        "upload_id": uid,
        "study_uid": study["study_instance_uid"],
        "series_uid": series["series_instance_uid"],
        "frames": [0],
        "template": "findings_only",
        "engine": engine_key,
        "model_id": (gpu or {}).get("default_model", ""),
        "window": {"preset": None, "wc": None, "ww": None, "invert": None},
        "temperature": 0.0,
    }).encode()

    t0 = time.time()
    events, text = [], []
    try:
        with req("/api/infer", method="POST", data=payload,
                 headers={"Content-Type": "application/json"}, timeout=1800) as r:
            ev = None
            for raw in r:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("event:"):
                    ev = line[6:].strip(); events.append(ev)
                elif line.startswith("data:") and ev == "delta":
                    try:
                        text.append(json.loads(line[5:].strip()).get("text", ""))
                    except ValueError:
                        pass
    except Exception as e:  # noqa: BLE001
        check("inferens tugadi", False, str(e))
    else:
        out = "".join(text).strip()
        check("SSE eventlari keldi", "start" in events, f"{len(events)} event")
        check("model matn qaytardi", len(out) > 20, f"{len(out)} belgi, {time.time()-t0:.1f}s")
        if out:
            print(f"      «{out[:110]}…»")

    # 6. Tozalash
    print("\n6 · Tozalash")
    try:
        req(f"/api/uploads/{uid}", method="DELETE", timeout=60).read()
        check("sinov arxivi o'chirildi", True)
    except Exception as e:  # noqa: BLE001
        check("sinov arxivi o'chirildi", False, str(e))

    print()
    if _failures:
        print(f"  \033[31m{_failures} ta tekshiruv muvaffaqiyatsiz.\033[0m\n")
        return 1
    print("  \033[32mHammasi joyida — servis ishlashga tayyor.\033[0m\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
