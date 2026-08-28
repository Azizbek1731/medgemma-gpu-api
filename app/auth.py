"""
API kalit autentifikatsiyasi.

NIMA UCHUN KERAK
Bu servis internetga ochiladi va unga BEMOR TASVIRLARI yuboriladi. Manba
loyihada autentifikatsiya umuman yo'q edi — u bitta
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
- CORS preflight (OPTIONS) uchun ISTISNO YO'Q. Bu servisga brauzer
  to'g'ridan-to'g'ri murojaat qilmaydi — barcha so'rovlar AviRadiolog'ning
  Django proksisidan keladi (README §6), va ilovada CORSMiddleware ham
  sozlanmagan, ya'ni preflight umuman bo'lmaydi. Istisno qoldirilsa esa
  kalitsiz odam OPTIONS bilan qaysi yo'llar borligini paypaslab bilib olardi.
"""
from __future__ import annotations

import hmac
import json
import logging
import os


log = logging.getLogger(__name__)

# Autentifikatsiyasiz ochiq qoladigan yo'llar.
# /healthz — monitoring uchun (hech qanday ma'lumot bermaydi).
PUBLIC_PATHS = frozenset({"/healthz"})


def configured_keys() -> list[str]:
    """MG_API_KEYS (vergul bilan) yoki MG_API_KEY dan kalitlarni oladi."""
    raw = os.getenv("MG_API_KEYS") or os.getenv("MG_API_KEY") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


class ApiKeyMiddleware:
    """
    Toza ASGI middleware — barcha so'rovlarni yopadi; faqat PUBLIC_PATHS ochiq.

    NEGA BaseHTTPMiddleware EMAS: u javobni o'z ichida qayta o'raydi va
    Starlette'ning ba'zi versiyalarida OQIM javoblarini (SSE) buferlab qo'yadi.
    Bu servisning ikkita asosiy endpointi (/api/infer, /api/batch) aynan oqim
    bilan ishlaydi, shuning uchun bu xavfni butunlay yo'q qilamiz: toza ASGI
    middleware javobga umuman tegmaydi, faqat kirishda tekshiradi.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        if path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        keys = configured_keys()
        if not keys:
            # Xavfsiz standart: kalit sozlanmasa servis OCHIQ qolmaydi.
            log.error("MG_API_KEY sozlanmagan — barcha so'rovlar rad etilmoqda.")
            await _deny(send, 503, "Servis sozlanmagan (API kalit yo'q).")
            return

        provided = _header(scope)
        # Solishtirish BAYTLARDA ketadi: `compare_digest` matnlarni faqat ikkalasi
        # ham ASCII bo'lganda solishtira oladi. Sarlavhada kirill harf yoki emoji
        # bo'lsa u TypeError chiqarardi, u esa ushlanmay 500 ga aylanardi — ya'ni
        # javob kodining o'zi kalit haqida ma'lumot berardi. Baytlarda har qanday
        # kirish bir xil 401 oladi va vaqt bo'yicha himoya ham saqlanadi.
        provided_bytes = provided.encode("utf-8")
        if not provided or not any(
            hmac.compare_digest(provided_bytes, k.encode("utf-8")) for k in keys
        ):
            # Kalitning o'zi hech qachon logga yozilmaydi.
            log.warning("Ruxsatsiz so'rov: %s %s", method, path)
            await _deny(send, 401, "Noto'g'ri yoki yo'q API kalit.")
            return

        await self.app(scope, receive, send)


def _header(scope) -> str:
    """`X-API-Key` yoki `Authorization: Bearer <key>` dan kalitni oladi."""
    for raw_name, raw_value in scope.get("headers", []):
        name = raw_name.decode("latin-1").lower()
        if name == "x-api-key":
            return raw_value.decode("latin-1").strip()
        if name == "authorization":
            v = raw_value.decode("latin-1")
            if v.lower().startswith("bearer "):
                return v[7:].strip()
    return ""


async def _deny(send, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({
        "type": "http.response.start",
        "status": status_code,
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})
