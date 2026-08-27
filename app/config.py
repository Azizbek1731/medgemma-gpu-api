"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MG_DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "lab.sqlite3"


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    # --- inference ---
    engine: str = field(default_factory=lambda: os.getenv("MG_ENGINE", "auto"))
    model_id: str = field(default_factory=lambda: os.getenv("MG_MODEL", ""))
    max_new_tokens: int = field(default_factory=lambda: _int("MG_MAX_NEW_TOKENS", 1200))
    temperature: float = field(
        default_factory=lambda: float(os.getenv("MG_TEMPERATURE", "0.0"))
    )

    # --- OpenAI-compatible endpoint (vLLM / LM Studio / llama.cpp / Ollama) ---
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("MG_OPENAI_BASE_URL", "http://localhost:1234/v1")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("MG_OPENAI_API_KEY", "not-needed")
    )

    # --- Google Vertex AI ---
    vertex_project: str = field(default_factory=lambda: os.getenv("MG_VERTEX_PROJECT", ""))
    vertex_location: str = field(
        default_factory=lambda: os.getenv("MG_VERTEX_LOCATION", "us-central1")
    )
    vertex_endpoint_id: str = field(
        default_factory=lambda: os.getenv("MG_VERTEX_ENDPOINT_ID", "")
    )

    # --- Hugging Face Inference Providers ---
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    # --- privacy ---
    # Network engines are refused unless the payload has been de-identified.
    require_deid_for_remote: bool = field(
        default_factory=lambda: _bool("MG_REQUIRE_DEID_FOR_REMOTE", True)
    )
    # Hard block on any non-local engine. Set MG_LOCAL_ONLY=1 for clinical data.
    local_only: bool = field(default_factory=lambda: _bool("MG_LOCAL_ONLY", True))

    # --- rendering ---
    # Longest edge of the PNG handed to the model. Gemma 3 / SigLIP works at 896px;
    # we send a little larger and let the processor downsample.
    inference_image_px: int = field(default_factory=lambda: _int("MG_INFER_PX", 1024))
    # Longest edge of the PNG shown in the browser viewer.
    viewer_image_px: int = field(default_factory=lambda: _int("MG_VIEWER_PX", 1024))
    # Refuse to send more than this many images in one request.
    max_images_per_request: int = field(default_factory=lambda: _int("MG_MAX_IMAGES", 8))

    # --- ingest limits ---
    max_upload_mb: int = field(default_factory=lambda: _int("MG_MAX_UPLOAD_MB", 512))
    max_files_per_zip: int = field(default_factory=lambda: _int("MG_MAX_FILES", 20_000))


settings = Settings()

LOCAL_ENGINES = {"mlx", "transformers", "mock"}
REMOTE_ENGINES = {"openai", "vertex", "hf"}


def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOAD_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_remote(engine_name: str) -> bool:
    """`openai` pointed at localhost still counts as local traffic."""
    if engine_name not in REMOTE_ENGINES:
        return False
    if engine_name == "openai":
        url = settings.openai_base_url
        return not any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))
    return True
