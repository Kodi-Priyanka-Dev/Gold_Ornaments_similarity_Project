# syntax=docker/dockerfile:1.7-labs
# ---------------------------------------------------------------------------
# Build stage: compile wheels into a venv and bake the ML weights into the image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only torch wheels. The default PyPI wheel bundles the CUDA runtime and is
# several GB, which made every deploy and container restart far slower than it
# needed to be. Override with --build-arg TORCH_INDEX_URL=https://pypi.org/simple
# if you deploy to a GPU host.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN apt-get update && apt-get install --yes --no-install-recommends \
        build-essential \
        g++ \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serve.txt .
RUN pip install --upgrade pip \
    && pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements-serve.txt

# Pre-download every model weight at build time. Previously the container
# fetched ~1 GB from HuggingFace on each cold start; when that download was slow
# the app could not answer the platform health probe in time and the site
# returned 503 until an attempt happened to be fast enough.
ENV HF_HOME=/opt/model-cache/huggingface \
    TORCH_HOME=/opt/model-cache/torch \
    SENTENCE_TRANSFORMERS_HOME=/opt/model-cache/sentence-transformers \
    XDG_CACHE_HOME=/opt/model-cache

RUN python -c "import timm; timm.create_model('convnext_base', pretrained=True, num_classes=0)" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"

# Voice search only: best-effort so a hiccup here does not fail the build.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" \
    || echo "WARNING: whisper weights not pre-cached; first voice search will download them"

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/model-cache/huggingface \
    TORCH_HOME=/opt/model-cache/torch \
    SENTENCE_TRANSFORMERS_HOME=/opt/model-cache/sentence-transformers \
    XDG_CACHE_HOME=/opt/model-cache \
    PORT=5000

# Offline mode: every weight is already baked in above, and this guarantees no
# startup path can stall on a HuggingFace round trip. If the best-effort whisper
# prefetch above printed its warning, set HF_HUB_OFFLINE=0 to let voice search
# download its model on first use.
ENV HF_HUB_OFFLINE=1

# libgomp1 is required by torch; curl backs the healthcheck. The compilers from
# the build stage are deliberately left behind.
RUN apt-get update && apt-get install --yes --no-install-recommends \
        libgomp1 \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/model-cache /opt/model-cache

WORKDIR /app
# models/best.pt is ~1 GB. In its own layer it is only rebuilt and re-pushed
# when the weights change, not on every code edit.
COPY models/ models/
COPY --exclude=models . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=90s \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Serve with gunicorn, not the Flask development server.
WORKDIR /app/backend
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:app"]
