"""
Yuklangan DICOM arxivlarini avtomatik o'chirish (PHI saqlash muddati).

NIMA UCHUN KERAK
Bu servis IJARADAGI GPU mashinasida ishlaydi va unga bemor tasvirlari
yuboriladi. Tahlil tugagach ular u yerda turishi shart emas — qancha uzoq
tursa, shuncha katta xavf (disk obrazi, zaxira nusxalar, mashina qaytarilishi).

Shuning uchun `MG_RETENTION_HOURS` dan eski yuklamalar fonda o'chiriladi.
Standart 24 soat. 0 qo'yilsa o'chirish butunlay o'chadi (tavsiya etilmaydi).

DIQQAT: bu tahlil NATIJALARINI (matnli xulosa) emas, faqat XOM TASVIRLARNI
o'chiradi — xulosa AviRadiolog bazasida saqlanadi va u yerda audit qilinadi.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

_CHECK_INTERVAL_SEC = 30 * 60   # yarim soatda bir marta

# Yangi papkalarga tegmaslik oralig'i. /api/upload papkani DARROV yaratadi, bazaga
# yozuvni esa arxiv ochilib indekslangandan KEYIN qo'shadi — shu oraliqda ketayotgan
# yuklama ham "yetim" ko'rinadi. Bu chegara uni tasodifan o'chirib qo'yishdan saqlaydi.
_ORPHAN_GRACE = timedelta(hours=1)


def retention_hours() -> int:
    try:
        return max(0, int(os.getenv("MG_RETENTION_HOURS", "24")))
    except ValueError:
        return 24


def purge_once() -> int:
    """Muddati o'tgan yuklamalarni o'chiradi. Nechtasi o'chirilganini qaytaradi."""
    hours = retention_hours()
    if hours <= 0:
        return 0

    from . import store
    from .config import UPLOAD_DIR

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    removed = 0

    try:
        uploads = store.list_uploads()
    except Exception:  # noqa: BLE001 — tozalash asosiy xizmatni buzmasin
        log.exception("Yuklamalar ro'yxatini o'qib bo'lmadi")
        return 0

    # Ro'yxatni sikldan OLDIN olamiz: quyida o'chirilgan yozuvlar ham shu to'plamda
    # qoladi, ya'ni yetim skanerlash ularni ikkinchi marta hisoblab yubormaydi.
    known_ids = {up["id"] for up in uploads}

    for up in uploads:
        created = up.get("created_at")
        if not created:
            continue
        try:
            ts = datetime.fromisoformat(str(created))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts >= cutoff:
            continue

        # Diskdagi xom tasvirlar
        path = UPLOAD_DIR / up["id"]
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        except Exception:  # noqa: BLE001
            log.exception("Papkani o'chirib bo'lmadi: %s", path)
            continue

        try:
            store.delete_upload(up["id"])
        except Exception:  # noqa: BLE001
            log.exception("Yozuvni o'chirib bo'lmadi: %s", up["id"])
            continue

        removed += 1
        # Bemor identifikatorlari LOGGA YOZILMAYDI — faqat texnik id.
        log.info("Muddati o'tgan yuklama o'chirildi: %s", up["id"])

    return removed + _purge_orphan_dirs(known_ids, cutoff)


def _purge_orphan_dirs(known_ids: set[str], cutoff: datetime) -> int:
    """Bazada yozuvi qolmagan, lekin diskda turgan yuklama papkalarini o'chiradi.

    NIMA UCHUN KERAK: /api/upload avval `data/uploads/<id>` papkasini yaratib arxivni
    yozadi, bazaga yozuvni esa faqat indekslash muvaffaqiyatli tugagach qo'shadi.
    Oradagi har qanday uzilish — jarayon o'ldirilishi, OOM, kutilmagan xato — papkani
    bemor tasvirlari bilan diskda qoldiradi. Bunday papka `store.list_uploads()` da
    ko'rinmaydi, ya'ni yuqoridagi sikl unga HECH QACHON yetib bormaydi va PHI GPU
    mashinasida muddatsiz qolib ketadi. Shu sababli diskning o'zini ham skanerlaymiz.
    """
    from .config import UPLOAD_DIR

    # Ikkala shartning qat'iyrog'i: papka ham saqlash muddatidan, ham himoya
    # oralig'idan eski bo'lsagina o'chiriladi.
    limit = min(cutoff, datetime.now(timezone.utc) - _ORPHAN_GRACE)
    removed = 0

    try:
        entries = list(UPLOAD_DIR.iterdir())
    except FileNotFoundError:
        return 0
    except Exception:  # noqa: BLE001 — tozalash asosiy xizmatni buzmasin
        log.exception("Yuklamalar papkasini o'qib bo'lmadi")
        return 0

    for path in entries:
        if not path.is_dir() or path.name in known_ids:
            continue
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        except OSError:
            continue
        if mtime >= limit:
            continue

        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            # Keyingi siklda qayta uriniladi — shuning uchun xato emas, ogohlantirish.
            log.warning("Yetim papkani to'liq o'chirib bo'lmadi: %s", path.name)
            continue

        removed += 1
        # Log faqat texnik papka nomi (yuklama id'si) — bemor ma'lumoti emas.
        log.info("Bazada yozuvi yo'q yetim yuklama o'chirildi: %s", path.name)

    return removed


def start_background_purge() -> None:
    """Fon oqimida davriy tozalashni yoqadi (servis startida chaqiriladi)."""
    if retention_hours() <= 0:
        log.warning("MG_RETENTION_HOURS=0 — yuklangan DICOM fayllar O'CHIRILMAYDI.")
        return

    def _loop() -> None:
        while True:
            try:
                n = purge_once()
                if n:
                    log.info("Tozalash: %s ta yuklama o'chirildi", n)
            except Exception:  # noqa: BLE001
                log.exception("Tozalash xatosi")
            time.sleep(_CHECK_INTERVAL_SEC)

    t = threading.Thread(target=_loop, name="retention-purge", daemon=True)
    t.start()
    log.info("PHI tozalash yoqildi: %s soatdan eski yuklamalar o'chiriladi", retention_hours())
