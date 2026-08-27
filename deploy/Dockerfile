# MedGemma GPU API — CUDA konteyneri
#
# Qurish:  docker build -f deploy/Dockerfile -t medgemma-api .
# Ishga:   docker compose -f deploy/docker-compose.yml up -d
#
# Talab: NVIDIA drayver + nvidia-container-toolkit (host mashinada).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Model og'irliklari shu yerga keshlanadi (volume bilan saqlanadi)
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

# torch'ni CUDA g'ildiragidan alohida o'rnatamiz (hajmi katta, alohida qatlam)
COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip \
 && python3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cu124 \
 && python3.11 -m pip install -r requirements.txt

COPY app ./app

# Xizmat root'dan ishlamaydi
RUN useradd --system --uid 10001 medgemma \
 && mkdir -p /data /models \
 && chown -R medgemma:medgemma /srv/app /data /models
USER medgemma

ENV MG_DATA_DIR=/data \
    MG_HOST=0.0.0.0 \
    MG_PORT=8077

EXPOSE 8077

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8077/healthz || exit 1

CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077"]
