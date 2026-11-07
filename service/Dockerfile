# syntax=docker/dockerfile:1
# Build from the REPOSITORY ROOT (the image needs the pd-model package for the preprocessor):
#   docker build -f service/Dockerfile -t pd-scoring-service:1.1.0 .
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY service/requirements.txt ./requirements.txt
COPY pyproject.toml ./pd-model/pyproject.toml
COPY src ./pd-model/src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r requirements.txt \
 && /opt/venv/bin/pip install ./pd-model   # runtime deps only (no training extras)

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH" \
    PD_ENVIRONMENT=prod PD_LOG_JSON=true PD_MODEL_PATH=/app/artifacts/model_card.json
RUN groupadd --system app && useradd --system --gid app --no-create-home app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY service/app ./app
COPY service/artifacts ./artifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).status == 200 else 1)"
# One process per container; scale horizontally with replicas (keeps Prometheus counters per-process correct).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header", "--timeout-graceful-shutdown", "20"]
