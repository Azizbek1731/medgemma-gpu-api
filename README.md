# AviRadiology AI — GPU servisi

AviRadiology AI tibbiy tasvir modelini **GPU'li mashinada** ishga tushirib, uni
[AviRadiolog](https://github.com/Azizbek1731/AviRadiolog) platformasiga ulash uchun
API servisi.

Bu **headless** servis — brauzer interfeysi yo'q. DICOM qabul qiladi, to'g'ri
windowing bilan render qiladi, modelga yuboradi va javobni oqim (SSE) bilan qaytaradi.

```
AviRadiolog (brauzer)
      │  JWT
      ▼
AviRadiolog backend (Django)     ← API kalit SHU YERDA qoladi
      │  X-API-Key + HTTPS
      ▼
AviRadiology AI GPU  ←── shu repo, GPU'li mashinada
      │
      ▼
   Model (transformers / CUDA)
```

> ⚠️ **Klinik ogohlantirish.** Bu model klinik qaror qabul qilish uchun tayyor
> mahsulot emas — model muallifining o'zi ham shuni aytadi. Har qanday xulosa
> radiolog tekshiruvidan o'tishi shart.

---

## 1. Talablar

| | Minimum | Tavsiya |
|---|---|---|
| GPU | NVIDIA, **8 GB VRAM** | 16 GB+ (T4, A10, RTX 4090…) |
| RAM | 16 GB | 32 GB |
| Disk | 40 GB | 100 GB (model + arxivlar) |
| OS | Ubuntu 22.04+ | — |
| Drayver | NVIDIA + CUDA 12.x | + `nvidia-container-toolkit` (Docker uchun) |

`medgemma-1.5-4b-it` bf16 da ~9 GB VRAM oladi. VRAM kam bo'lsa 4-bit kvantlash kerak.

**HuggingFace tokeni majburiy:** `google/medgemma-1.5-4b-it` — gated repo.
[Sahifaga kiring](https://huggingface.co/google/medgemma-1.5-4b-it) → *Access repository* →
[token yarating](https://huggingface.co/settings/tokens).

---

## 2. O'rnatish

### Variant A — Docker (tavsiya etiladi)

```bash
git clone https://github.com/Azizbek1731/medgemma-gpu-api.git
cd medgemma-gpu-api
cp .env.example .env

# .env ni to'ldiring (pastga qarang), so'ng:
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f
```

### Variant B — systemd

```bash
sudo useradd --system --home /srv/medgemma --shell /usr/sbin/nologin medgemma
sudo mkdir -p /srv/medgemma && sudo chown medgemma: /srv/medgemma
sudo -u medgemma git clone https://github.com/Azizbek1731/medgemma-gpu-api.git /srv/medgemma
cd /srv/medgemma

# Ubuntu'da venv alohida paketda — busiz `python3 -m venv` xato beradi
sudo apt-get update && sudo apt-get install -y python3-venv

sudo -u medgemma python3 -m venv venv
sudo -u medgemma venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
sudo -u medgemma venv/bin/pip install -r requirements.txt

# data/ va models/ repoda yo'q (.gitignore da). Unit faylidagi ReadWritePaths
# mavjud bo'lmagan yo'lni ko'rsa systemd mount namespace'ni qura olmaydi va
# servis 226/NAMESPACE bilan cheksiz qayta urinadi — shuning uchun oldindan yaratamiz.
sudo -u medgemma mkdir -p /srv/medgemma/data /srv/medgemma/models

sudo -u medgemma cp .env.example .env && sudo chmod 600 .env   # to'ldiring
sudo install -m 0644 deploy/systemd/medgemma-api.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now medgemma-api
```

---

## 3. Sozlash (`.env`)

Eng muhim ikkitasi:

```bash
# API kaliti — busiz servis HAMMA so'rovni rad etadi (503)
MG_API_KEY=$(openssl rand -hex 32)

# Gated model uchun HuggingFace tokeni
HF_TOKEN=hf_...
```

Qolganlari `.env.example` da izohi bilan. Diqqat qiling:

| O'zgaruvchi | Standart | Nima uchun muhim |
|---|---|---|
| `MG_LOCAL_ONLY` | `1` | Tarmoq engine'larini (OpenAI/Vertex) bloklaydi — bemor tasviri uchinchi tomonga ketmaydi |
| `MG_RETENTION_HOURS` | `24` | Yuklangan DICOM shu muddatdan keyin **avtomatik o'chiriladi** |
| `MG_MAX_UPLOAD_MB` | `512` | Ochiq endpointda diskni to'ldirishning oldini oladi |
| `MG_ENABLE_DOCS` | o'chiq | `/docs` va `/openapi.json` yopiq — keraksiz hujum yuzasi |

**Model keshi (`HF_HOME`).** systemd unit'i uni `/srv/medgemma/models` ga o'rnatadi,
Docker esa `/models` volume'iga. Standart yo'l (`$HOME/.cache/huggingface`)
`ProtectSystem=strict` ostida read-only bo'lgani uchun og'irliklar yuklanmaydi.
Boshqa papkani tanlasangiz, unit'dagi `ReadWritePaths` ni ham birga o'zgartiring.

---

## 4. Domen va TLS

```bash
sudo cp deploy/nginx/medgemma.conf /etc/nginx/sites-available/
sudo nano /etc/nginx/sites-available/medgemma.conf     # server_name ni o'zgartiring
sudo ln -sf /etc/nginx/sites-available/medgemma.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d ai.example.uz
```

> Servis **faqat `127.0.0.1:8077`** da tinglaydi. Tashqi kirish nginx orqali,
> TLS bilan. Konteynerni yoki portni to'g'ridan-to'g'ri internetga ochmang.

**Qo'shimcha himoya (tavsiya):** faqat AviRadiolog serverining IP'sini kiriting:

```bash
# ⚠️ ENG BIRINCHI: SSH'ni oching. ufw yoqilgach standart siyosat kiruvchi
# ulanishlarni rad etadi — bu qadamsiz serverga masofadan kira olmay qolasiz.
sudo ufw allow OpenSSH

# Tartib muhim: ufw birinchi mos kelgan qoidani qo'llaydi, shuning uchun
# ruxsat rad etishdan OLDIN turishi kerak.
sudo ufw allow from <AVIRADIOLOG_IP> to any port 443 proto tcp
sudo ufw deny 443

# Let's Encrypt sertifikatni HTTP-01 sinovi bilan yangilaydi, ya'ni 80-port
# INTERNETDAN ochiq turishi shart. Yopilsa `certbot renew` jimgina yiqiladi
# va sertifikat 90 kundan keyin muddati tugab, servis ishlamay qoladi.
sudo ufw allow 80/tcp

# ufw standart holda O'CHIQ — busiz yuqoridagi qoidalar hech narsa qilmaydi
sudo ufw enable
sudo ufw status verbose        # "Status: active" ekanini tasdiqlang
```

---

## 5. Tekshirish

Skript `pydicom` va `numpy` ni talab qiladi — ular tizim python'ida emas, o'rnatish
paytida yaratilgan muhitda turadi. Shuning uchun aynan o'sha python bilan chaqiring:

```bash
# Variant B (systemd) — bog'liqliklar venv ichida, kalit esa 0600 huquqli .env da,
# uni faqat medgemma foydalanuvchisi o'qiy oladi
cd /srv/medgemma
sudo -u medgemma venv/bin/python tools/smoke_test.py

# Variant A (Docker) — tools/ konteynerga ko'chirilmaydi, host'da vaqtinchalik muhit kerak.
# Ubuntu'da venv alohida paketda (§2 dagi kabi), busiz "ensurepip is not available" beradi.
sudo apt-get install -y python3-venv
python3 -m venv /tmp/smoke && /tmp/smoke/bin/pip install pydicom numpy
MG_API_KEY=<.env dagi kalit> /tmp/smoke/bin/python tools/smoke_test.py

# yoki masofadan (pydicom va numpy o'rnatilgan istalgan python bilan):
MG_URL=https://ai.example.uz MG_API_KEY=... python3 tools/smoke_test.py
```

Skript sun'iy DICOM yasab (bemor ma'lumoti kerak emas) quyidagilarni tekshiradi:
ulanish · **kalitsiz so'rov bloklanishi** · GPU tayyorligi · DICOM pipeline ·
PHI audit · SSE inferens · tozalash.

Kutiladigan yakun: `Hammasi joyida — servis ishlashga tayyor.`

---

## 6. AviRadiolog'ga ulash

AviRadiolog backendi (`backend/.env`) da:

```bash
MEDGEMMA_URL=https://ai.example.uz
MEDGEMMA_API_KEY=<MG_API_KEY bilan bir xil>
```

So'ng `sudo systemctl restart aviradiolog-api`.

Brauzer servisga **to'g'ridan-to'g'ri murojaat qilmaydi** — barcha so'rovlar
Django proksisi (`/api/mg/...`) orqali o'tadi. Sabablari:

- API kalit brauzerga hech qachon chiqmaydi;
- faqat tizimga kirgan **shifokor** murojaat qila oladi;
- bemor tasviri tashqi mashinaga ketgani **audit jurnaliga** yoziladi (`ai_remote`);
- proksi **oq ro'yxat** bilan ishlaydi — `/api/export.csv`, `/api/metrics` kabi
  yo'llar tashqariga umuman ochilmaydi.

---

## 7. API

Barcha yo'llar `X-API-Key: <kalit>` talab qiladi (`Authorization: Bearer <kalit>` ham
qabul qilinadi). Yagona istisno — `/healthz`.

| Yo'l | Vazifa |
|---|---|
| `GET /healthz` | Monitoring (ochiq, ma'lumot bermaydi) |
| `POST /api/upload` | DICOM `.zip` yoki `.dcm` yuklash |
| `GET /api/uploads` · `/api/uploads/{id}` | Arxivlar va tekshiruv daraxti |
| `DELETE /api/uploads/{id}` | Arxivni o'chirish |
| `GET /api/uploads/{id}/series` | Seriya tafsiloti + window presetlari |
| `GET /api/uploads/{id}/frame.png` | Kadr renderi (+ `X-Window-*` sarlavhalari) |
| `GET /api/uploads/{id}/frame-info` | PHI audit va DICOM meta |
| `GET /api/templates` · `/api/engines` | Prompt shablonlari, engine holati |
| `POST /api/infer` | Tanlangan kadrlar tahlili (**SSE**) |
| `POST /api/batch/plan` · `/api/batch` | Butun tekshiruv + yakuniy sintez (**SSE**) |

Batafsil: manba loyihaning `INTEGRATION.md` hujjati.

---

## 8. Xavfsizlik qarorlari

Bu servis bemor tasvirlari bilan ishlaydi va ochiq domenda turadi. Shuning uchun
manba loyihadan farqli o'laroq:

| Qaror | Sabab |
|---|---|
| **API kalit majburiy** (`app/auth.py`) | Manba loyihada autentifikatsiya umuman yo'q edi |
| Middleware darajasida — "hamma yopiq" | Dekorator osish unutiladi; yangi endpoint avtomatik ochilmaydi |
| Kalit sozlanmasa — **503**, ochiq emas | Noto'g'ri konfiguratsiya himoyani o'chirib qo'ymasligi kerak |
| `hmac.compare_digest` | Vaqt bo'yicha hujumdan himoya |
| `/docs` va `/openapi.json` yopiq | Endpoint ro'yxatini oshkor qilmaslik |
| **PHI avtomatik o'chirish** (`app/retention.py`) | Ijaradagi mashinada bemor tasviri qolib ketmasligi uchun |
| `MG_LOCAL_ONLY=1` standart | Tasvir OpenAI/Vertex'ga ketib qolmasin |
| Yuklash chegarasi 4096 → **512 MB** | Ochiq endpointda diskni to'ldirish oson |
| mlx / openai / vertex engine'lari olib tashlandi | GPU serverida kerak emas — hujum yuzasi va bog'liqlik kamayadi |
| Konteyner root'dan ishlamaydi | Standart himoya |

Kalit hech qachon logga yozilmaydi; ruxsatsiz urinish faqat yo'l va metod bilan qayd etiladi.

---

## 9. Cheklovlar

- **Bounding box faqat ko'krak rentgeni (CR/DX)** uchun o'lchangan (Chest ImaGenome,
  IoU 38.0). KT/MRT'da ramkalar chiqadi, lekin tasdiqlanmagan.
- **2D model** — KT/MRT'da tanlangan kesimlarni ko'radi, butun hajmni emas.
- **Javob ingliz tilida.** AviRadiolog uni shifokor tiliga o'giradi
  (`/api/ai/translate/`).
- **Bitta generatsiya navbati** — model obyekti thread-safe emas, so'rovlar
  ketma-ket bajariladi. Ko'p foydalanuvchi kerak bo'lsa gorizontal masshtab kerak.

---

## Manba

DICOM pipeline, prompt shablonlari va inferens mantig'i
[MedGemma Radiology Lab](https://github.com/Azizbek1731/Medgemmademo) loyihasidan
olingan (sinovdan o'tgan kod qayta yozilmadi). Bu repo unga GPU deployment,
autentifikatsiya va PHI saqlash muddatini qo'shadi.
