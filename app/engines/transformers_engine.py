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

KNOWN_MODELS = [
    ("google/medgemma-1.5-4b-it", "MedGemma 1.5 4B — eng yangi (CT/MRI 3D, bounding box)"),
    ("google/medgemma-4b-it", "MedGemma 4B — birinchi versiya"),
    ("google/medgemma-27b-it", "MedGemma 27B multimodal — 55+ GB, katta GPU kerak"),
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
            return False, "torch o'rnatilmagan: pip install -r requirements-torch.txt"
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, "transformers o'rnatilmagan: pip install -r requirements-torch.txt"
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
        try:
            # transformers >= 5 renamed torch_dtype -> dtype
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id, dtype=dtype, low_cpu_mem_usage=True
                )
            except TypeError:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
                )
            self._model = model.to(self._device).eval()
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Modelni yuklab bo'lmadi ({self.model_id}): {exc}\n"
                "MedGemma gated model — avval huggingface.co da litsenziyani qabul qiling, "
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
        from transformers import TextIteratorStreamer

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

        for text in streamer:
            if text:
                yield text
        thread.join(timeout=5)
        if error:
            raise EngineError(f"Generatsiya xatosi: {error[0]}") from error[0]
