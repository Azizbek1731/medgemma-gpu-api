"""Inference engine interface shared by every backend."""

from __future__ import annotations

import base64
import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from PIL import Image


@dataclass
class GenerationRequest:
    system: str
    user_text: str
    images: list[Image.Image] = field(default_factory=list)
    max_new_tokens: int = 1200
    temperature: float = 0.0


@dataclass
class EngineInfo:
    key: str
    label: str
    model_id: str
    available: bool
    reason: str = ""
    requires_network: bool = False
    device: str = ""
    loaded: bool = False


class EngineError(RuntimeError):
    pass


class Engine(ABC):
    key: str = "base"
    label: str = "Base"
    requires_network: bool = False
    default_model: str = ""

    def __init__(self, model_id: str = ""):
        self.model_id = model_id or self.default_model
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def check_available(cls) -> tuple[bool, str]:
        """``(available, human readable reason)`` — must not import heavy deps eagerly."""

    def load(self) -> None:
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    # -- inference ---------------------------------------------------------------------

    @abstractmethod
    def stream(self, req: GenerationRequest) -> Iterator[str]:
        """Yield text deltas."""

    def generate(self, req: GenerationRequest) -> str:
        return "".join(self.stream(req))

    # -- reporting ---------------------------------------------------------------------

    def device_name(self) -> str:
        return ""

    def info(self) -> EngineInfo:
        ok, reason = self.check_available()
        return EngineInfo(
            key=self.key,
            label=self.label,
            model_id=self.model_id,
            available=ok,
            reason=reason,
            requires_network=self.requires_network,
            device=self.device_name(),
            loaded=self._loaded,
        )


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def image_to_data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(image_to_png_bytes(img)).decode("ascii")


class Ticker:
    """Tiny throughput counter so the UI can show tokens/sec."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.chunks = 0

    def tick(self) -> None:
        self.chunks += 1

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start
