"""Engine registry.

Only one engine is kept resident at a time — a 4B model is several gigabytes and this is
expected to run on a laptop alongside a PACS viewer.
"""

from __future__ import annotations

import logging
import threading

from .base import Engine, EngineError, EngineInfo, GenerationRequest
from .mock_engine import MockEngine
from .transformers_engine import KNOWN_MODELS as TORCH_MODELS
from .transformers_engine import TransformersEngine

log = logging.getLogger(__name__)

ENGINE_CLASSES: dict[str, type[Engine]] = {
    "transformers": TransformersEngine,
    "mock": MockEngine,
}

# Preference order for MG_ENGINE=auto.
AUTO_ORDER = ("transformers", "mock")

MODEL_SUGGESTIONS: dict[str, list[dict]] = {
    "transformers": [{"id": i, "label": l} for i, l in TORCH_MODELS],
    "mock": [{"id": "mock-medgemma", "label": "Soxta backend"}],
}

_lock = threading.RLock()
_current: Engine | None = None

# Model objects (MLX and torch alike) are not safe to generate from concurrently: two
# overlapping requests share KV cache and sampler state and corrupt each other's output.
# Every caller must hold this while streaming, so generations queue instead of interleaving.
INFERENCE_LOCK = threading.Lock()


class _QueuedGeneration:
    """Context manager that serialises generation and reports when it had to wait."""

    def __init__(self) -> None:
        self.waited = False

    def __enter__(self) -> "_QueuedGeneration":
        if not INFERENCE_LOCK.acquire(blocking=False):
            self.waited = True
            INFERENCE_LOCK.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        INFERENCE_LOCK.release()


def generation_slot() -> _QueuedGeneration:
    return _QueuedGeneration()


def generation_busy() -> bool:
    if INFERENCE_LOCK.acquire(blocking=False):
        INFERENCE_LOCK.release()
        return False
    return True


def resolve_auto() -> str:
    for key in AUTO_ORDER:
        cls = ENGINE_CLASSES[key]
        try:
            ok, _ = cls.check_available()
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            return key
    return "mock"


def list_engines() -> list[dict]:
    out = []
    for key, cls in ENGINE_CLASSES.items():
        try:
            ok, reason = cls.check_available()
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, str(exc)
        out.append(
            {
                "key": key,
                "label": cls.label,
                "available": ok,
                "reason": reason,
                "requires_network": cls.requires_network,
                "default_model": cls.default_model,
                "models": MODEL_SUGGESTIONS.get(key, []),
                "loaded": _current is not None
                and _current.key == key
                and _current.loaded,
                "loaded_model": _current.model_id
                if (_current is not None and _current.key == key)
                else "",
            }
        )
    return out


def get_engine(key: str = "", model_id: str = "", *, autoload: bool = False) -> Engine:
    """Return the active engine, swapping (and unloading) if the request differs."""
    global _current

    from ..config import settings

    key = key or settings.engine or "auto"
    if key == "auto":
        key = resolve_auto()
    if key not in ENGINE_CLASSES:
        raise EngineError(f"Noma'lum engine: {key}")

    cls = ENGINE_CLASSES[key]
    model_id = model_id or settings.model_id or cls.default_model

    with _lock:
        if _current is not None and _current.key == key and _current.model_id == model_id:
            if autoload and not _current.loaded:
                _current.load()
            return _current

        if _current is not None:
            log.info("Unloading engine %s (%s)", _current.key, _current.model_id)
            try:
                _current.unload()
            except Exception as exc:  # noqa: BLE001
                log.warning("Unload failed: %s", exc)

        engine = cls(model_id=model_id)
        _current = engine
        if autoload:
            engine.load()
        return engine


def current_engine() -> Engine | None:
    return _current


def unload_current() -> None:
    global _current
    with _lock:
        if _current is not None:
            _current.unload()


__all__ = [
    "Engine",
    "EngineError",
    "EngineInfo",
    "GenerationRequest",
    "ENGINE_CLASSES",
    "MODEL_SUGGESTIONS",
    "INFERENCE_LOCK",
    "generation_slot",
    "generation_busy",
    "list_engines",
    "get_engine",
    "current_engine",
    "unload_current",
    "resolve_auto",
]
