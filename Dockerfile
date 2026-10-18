# (c) Copyright 2026 Bhaskar Kaphle. MIT License.
#
# Dockerfile for OmniFlow. Builds a slim Python 3.11 image that runs the
# offline test suite and exposes the FastAPI service on :8000.
#
#   docker build -t omniflow:0.1.0 .
#   docker run --rm -p 8000:8000 omniflow:0.1.0
#
# Heavy optional backends (torch, transformers, easyocr, layoutparser,
# z-ai-web-dev-sdk) are intentionally NOT installed here. The image runs
# the full 69-test suite and the bundled mock-benchmark offline, with no
# GPU/network/API-key requirements. For a GPU-enabled image, use the
# `nvidia/cuda:12.2.0-runtime-ubuntu22.04` base and add the backends in
# a separate stage.

FROM python:3.11-slim AS base

# --- metadata ----------------------------------------------------------
LABEL org.opencontainers.image.title="OmniFlow" \
      org.opencontainers.image.description="Training-free omni-modal LLM orchestration framework" \
      org.opencontainers.image.authors="Bhaskar Kaphle <bhaskar.kaphle@example.org>" \
      org.opencontainers.image.source="https://github.com/BhaskarKaphle/omniflow" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="0.1.0"

# --- system deps (kept minimal; slim base has most of what we need) ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- python deps ------------------------------------------------------
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# --- copy the source + config ----------------------------------------
COPY src/ /app/src/
COPY config/ /app/config/
COPY tests/ /app/tests/
COPY pyproject.toml README.md /app/

# --- install the package in editable mode so entrypoints are available
RUN pip install --no-cache-dir --no-deps -e .

# --- runtime config ---------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMNIFLOW_CONFIG=/app/config/config.yaml \
    OMNIFLOW_HOST=0.0.0.0 \
    OMNIFLOW_PORT=8000

EXPOSE 8000

# --- healthcheck ------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --silent --fail http://localhost:8000/health || exit 1

# --- default command: run tests, then serve ---------------------------
# To run only the tests:    docker run --rm omniflow:0.1.0 pytest -q
# To start the API:         docker run --rm -p 8000:8000 omniflow:0.1.0
# To run the mock bench:    docker run --rm omniflow:0.1.0 python -m omnimodal.evaluation.benchmark
CMD ["sh", "-c", "pytest -q && python -c \"print('All OmniFlow tests passed. Starting FastAPI service on http://0.0.0.0:8000 ...')\" && exec uvicorn omnimodal.api.server:app --host 0.0.0.0 --port 8000"]
