FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements.txt requirements-mineru.txt constraints-mineru.txt ./

RUN python -m venv /app/.venv && \
    /app/.venv/bin/pip install --upgrade pip && \
    /app/.venv/bin/pip install -r requirements-core.txt

RUN python -m venv /app/.venv-mineru && \
    /app/.venv-mineru/bin/pip install --upgrade pip && \
    /app/.venv-mineru/bin/pip install -r requirements-mineru.txt -c constraints-mineru.txt

COPY . .

ENV MINERU_PY=/app/.venv-mineru/bin/python
ENV DISABLE_MINERU_LLM=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
