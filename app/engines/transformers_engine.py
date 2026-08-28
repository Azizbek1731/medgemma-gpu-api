"""PyTorch / transformers local inference (CUDA, Apple MPS, or CPU).

Slower than MLX on a Mac, but it is the only backend that can run the newest
``google/medgemma-1.5-4b-it`` weights today (no MLX conversion is published yet), and
it is the right choice on an NVIDIA box.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterator

from .base import Engine, EngineError, GenerationRequest

log = logging.getLogger(__name__)

DEFAULT_TORCH_MODEL = "google/medgemma-1.5-4b-it"

# Bekor qilingan generatsiyani kutish chegarasi. Bayroq qo'yilgach generate()
# joriy token qadamini tugatib to'xtaydi — bu odatda bir necha yuz millisekund.
# Chegara faqat oqim osilib qolgan holatda so'rovni abadiy bloklab qo'ymaslik uchun.
_STOP_JOIN_TIMEOUT_SEC = 30

# DIQQAT: chapdagi identifikatorlar — HuggingFace'dagi HAQIQIY repo nomlari.
# Ular o'zgartirilsa model umuman yuklanmaydi. O'ngdagi yorliqlargina
# foydalanuvchiga ko'rinadi.
KNOWN_MODELS = [
    ("google/medgemma-1.5-4b-it", "AviRadiology AI 1.5 — asosiy (CT/MRI 3D, bounding box)"),
    ("google/medgemma-4b-it", "AviRadiology AI 1.0 — birinchi versiya"),
    ("google/medgemma-27b-it", "AviRadiology AI XL — 55+ GB, katta GPU kerak"),
]


def _pick_device() -> tuple[str, str]:
    """Return ``(device, dtype_name)``."""
    import torch

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        return "cuda", "bfloat16" if major >= 8 else "float16"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "bfloat16"
    return "cpu", "float32"


class TransformersEngine(Engine):
    key = "transformers"
    label = "Transformers (PyTorch — CUDA / MPS / CPU, lokal)"
    requires_network = False
    default_model = DEFAULT_TORCH_MODEL

    def __init__(self, model_id: str = ""):
        super().__init__(model_id)
        self._model = None
        self._processor = None
        self._device = ""
        self._dtype_name = ""

    # -- availability ------------------------------------------------------------------

    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            # NEGA alohida buyruq: torch CUDA g'ildiragi PyPI'dan emas, drayverga mos
            # indeksdan olinadi — requirements.txt uni o'rnata olmaydi (README §2).
            return False, (
                "torch o'rnatilmagan: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, "transformers o'rnatilmagan: pip install -r requirements.txt"
        device, dtype = _pick_device()
        if device == "cpu":
            return True, "Faqat CPU topildi — juda sekin bo'ladi (rasm uchun bir necha daqiqa)."
        return True, f"Tayyor ({device}, {dtype})."

    def device_name(self) -> str:
        if self._device:
            return f"{self._device} / {self._dtype_name}"
        try:
            d, t = _pick_device()
            return f"{d} / {t}"
        except Exception:  # noqa: BLE001
            return ""

    # -- lifecycle ---------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        ok, reason = self.check_available()
        if not ok:
            raise EngineError(reason)

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._device, self._dtype_name = _pick_device()
        dtype = getattr(torch, self._dtype_name)

        log.info("Loading %s on %s (%s) ...", self.model_id, self._device, self._dtype_name)

        # NEGA device_map: `from_pretrained(...)` keyin `.to(device)` modelni avval
        # TO'LIQ CPU RAM'ga yuklaydi, so'ng GPU'ga ko'chiradi — ya'ni bir vaqtning
        # o'zida ~2× xotira kerak bo'ladi va start sekinlashadi. `device_map` bilan
        # accelerate og'irliklarni to'g'ridan-to'g'ri qatlam-qatlam GPU'ga joylaydi.
        # accelerate bo'lmasa eski yo'lga qaytamiz.
        kw = {"low_cpu_mem_usage": True}
        try:
            import accelerate  # noqa: F401
            kw["device_map"] = self._device
        except ImportError:
            log.info("accelerate topilmadi — model avval CPU'ga yuklanadi.")

        try:
            # transformers >= 5 renamed torch_dtype -> dtype
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id, dtype=dtype, **kw
                )
            except TypeError:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id, torch_dtype=dtype, **kw
                )
            if "device_map" not in kw:
                model = model.to(self._device)
            self._model = model.eval()
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Modelni yuklab bo'lmadi ({self.model_id}): {exc}\n"
                "Model litsenziyalangan (gated) — avval huggingface.co da litsenziyani qabul qiling, "
                "so'ng `hf auth login` bilan tizimga kiring."
            ) from exc

        self._loaded = True
        log.info("Model ready on %s.", self._device)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # -- inference ---------------------------------------------------------------------

    def _build_messages(self, req: GenerationRequest) -> list[dict]:
        messages: list[dict] = []
        if req.system:
            messages.append({"role": "system", "content": [{"type": "text", "text": req.system}]})
        content: list[dict] = [{"type": "image", "image": img} for img in req.images]
        content.append({"type": "text", "text": req.user_text})
        messages.append({"role": "user", "content": content})
        return messages

    def stream(self, req: GenerationRequest) -> Iterator[str]:
        if not self._loaded:
            self.load()

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        messages = self._build_messages(req)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # BatchFeature.to(dtype=...) only casts floating-point tensors, so input_ids
        # stay int64 while pixel_values follow the model dtype.
        inputs = inputs.to(self._model.device, dtype=getattr(torch, self._dtype_name))

        streamer = TextIteratorStreamer(
            self._processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.temperature > 0,
        )
        if req.temperature > 0:
            gen_kwargs["temperature"] = req.temperature

        # NEGA bekor qilish bayrog'i kerak: iste'molchi oqimni yarmida tashlab
        # ketishi mumkin (batch bekor qilindi yoki mijoz uzildi). O'zi bilan
        # `generate()` bundan xabar topmaydi — u max_new_tokens gacha GPU'da
        # ishlayveradi. Chaqiruvchi esa shu payt INFERENCE_LOCK ni bo'shatib
        # keyingi so'rovni boshlaydi, natijada tashlab ketilgan va yangi
        # generatsiya bitta GPU'ni talashadi (ikkalasi sekinlashadi yoki VRAM tugaydi).
        cancel = threading.Event()

        class _AbandonedByConsumer(StoppingCriteria):
            # generate() buni har bir token qadamidan keyin so'raydi, shuning uchun
            # bekor qilish eng ko'pi bilan bitta qadam ichida kuchga kiradi.
            def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
                return cancel.is_set()

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_AbandonedByConsumer()])

        error: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(**gen_kwargs)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            # Oqim qanday tugashidan qat'i nazar — normal yakun, xato yoki
            # tashlab ketilganda kelgan GeneratorExit — generatsiya haqiqatan
            # to'xtaganini shu yerda kutamiz. Chaqiruvchidagi `with generation_slot()`
            # shu `finally` dan KEYIN yopiladi, ya'ni lock GPU bo'shagach bo'shaydi.
            cancel.set()
            thread.join(timeout=_STOP_JOIN_TIMEOUT_SEC)
            if thread.is_alive():
                # Osilib qolgan oqim uchun so'rovni cheksiz ushlab turmaymiz, lekin
                # jimgina o'tkazib yuborsak GPU nega bandligi keyin tushunarsiz bo'ladi.
                log.warning(
                    "Generatsiya %s soniyada to'xtamadi — oqim fonda davom etmoqda.",
                    _STOP_JOIN_TIMEOUT_SEC,
                )
        if error:
            raise EngineError(f"Generatsiya xatosi: {error[0]}") from error[0]
