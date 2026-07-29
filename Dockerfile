FROM python:3.11-slim AS core-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt /tmp/requirements-core.txt
RUN python -m venv /opt/venvs/core \
    && /opt/venvs/core/bin/pip install --upgrade pip \
    && /opt/venvs/core/bin/pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 \
    && /opt/venvs/core/bin/pip install -r /tmp/requirements-core.txt


FROM python:3.11-slim AS mineru-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-mineru.txt constraints-mineru.txt /tmp/
RUN python -m venv /opt/venvs/mineru \
    && /opt/venvs/mineru/bin/pip install --upgrade pip \
    && /opt/venvs/mineru/bin/pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 \
        torchvision==0.26.0 \
    && /opt/venvs/mineru/bin/pip install \
        -r /tmp/requirements-mineru.txt \
        -c /tmp/constraints-mineru.txt


FROM python:3.11-slim AS runtime

ENV PATH=/opt/venvs/core/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES="" \
    TOKENIZERS_PARALLELISM=false \
    DISABLE_MINERU_LLM=1 \
    MINERU_PYTHON=/opt/venvs/mineru/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ocrmypdf \
        tesseract-ocr-eng \
        tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home --shell /usr/sbin/nologin rag

COPY --from=core-builder /opt/venvs/core /opt/venvs/core
COPY --from=mineru-builder /opt/venvs/mineru /opt/venvs/mineru

WORKDIR /app
COPY --chown=rag:rag . /app

RUN mkdir -p /app/storage/raw /app/storage/parsed /tmp/prometheus \
    && mkdir -p /home/rag/.cache \
    && mkdir -p /opt/venvs/mineru/lib/python3.11/site-packages/rapid_table/models \
    && chown -R rag:rag \
        /app/storage \
        /tmp/prometheus \
        /home/rag/.cache \
        /opt/venvs/mineru/lib/python3.11/site-packages/rapid_table/models \
    && sed -i 's/\r$//' /app/scripts/docker-entrypoint.sh \
    && chmod 0755 /app/scripts/docker-entrypoint.sh

USER rag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=3).read()"]

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2} --limit-concurrency ${UVICORN_LIMIT_CONCURRENCY:-32} --backlog 128 --timeout-keep-alive 5 --timeout-graceful-shutdown 60"]
