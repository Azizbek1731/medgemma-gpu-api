"""
API kalit autentifikatsiyasi.

NIMA UCHUN KERAK
Bu servis internetga ochiladi va unga BEMOR TASVIRLARI yuboriladi. Manba
loyihada (MedGemma Radiology Lab) autentifikatsiya umuman yo'q edi — u bitta
ishonchli foydalanuvchi uchun lokal asbob edi. Ochiq domenda esa himoyasiz
qoldirilsa, har kim DICOM yuklashi, boshqalarning tekshiruvlarini o'qishi va
GPU'ni band qilishi mumkin.

DIZAYN
- Bitta joyda — MIDDLEWARE darajasida tekshiriladi. Har bir endpointga
  dekorator osish oson unutiladi (ayniqsa SSE va yangi qo'shilgan yo'llarga),
  shuning uchun "hamma yopiq, ochiqlari ro'yxatda" tamoyili ishlatiladi.
- Kalit `X-API-Key` sarlavhasida keladi. `Authorization: Bearer <key>` ham
  qabul qilinadi (ba'zi proksilar uchun qulay).
- Solishtirish `hmac.compare_digest` bilan — vaqt bo'yicha hujumdan himoya.
- Bir nechta kalit qo'llab-quvvatlanadi (vergul bilan): kalitni almashtirganda
  eskisini bir muddat qoldirib turish uchun.
"""
from __future__ import annotations

import hmac
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

# Autentifikatsiyasiz ochiq qoladigan yo'llar.
# /healthz — monitoring uchun (hech qanday ma'lumot bermaydi).
PUBLIC_PATHS = frozenset({"/healthz"})


def configured_keys() -> list[str]:
    """MG_API_KEYS (vergul bilan) yoki MG_API_KEY dan kalitlarni oladi."""
    raw = os.getenv("MG_API_KEYS") or os.getenv("MG_API_KEY") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _extract(request: Request) -> str:
    key = request.headers.get("x-api-key", "")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Barcha so'rovlarni yopadi; faqat PUBLIC_PATHS ochiq."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # CORS preflight — brauzer sarlavha qo'sha olmaydi, o'tkazamiz.
        # (Javobda hech qanday ma'lumot bo'lmaydi.)
        if request.method == "OPTIONS":
            return await call_next(request)

        if path in PUBLIC_PATHS:
            return await call_next(request)

        keys = configured_keys()
        if not keys:
            # Kalit sozlanmagan bo'lsa servis OCHIQ qolmaydi — to'xtaydi.
            # "Xavfsiz standart": noto'g'ri konfiguratsiya sukut bilan
            # himoyani o'chirib qo'ymasligi kerak.
            log.error("MG_API_KEY sozlanmagan — barcha so'rovlar rad etilmoqda.")
            return JSONResponse(
                {"detail": "Servis sozlanmagan (API kalit yo'q)."}, status_code=503
            )

        provided = _extract(request)
        if not provided or not any(hmac.compare_digest(provided, k) for k in keys):
            # Kalitning o'zi hech qachon logga yozilmaydi.
            log.warning("Ruxsatsiz so'rov: %s %s", request.method, path)
            return JSONResponse({"detail": "Noto'g'ri yoki yo'q API kalit."}, status_code=401)

        return await call_next(request)
