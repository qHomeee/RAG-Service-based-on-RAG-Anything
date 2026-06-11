# Local Embedding Models

Use a local SentenceTransformer model in production when HuggingFace is blocked, SSL is unavailable, or the deployment must run without network access.

Recommended `.env` values are cross-platform because the path is relative to the project root:

```env
EMBED_MODEL=storage/models/all-MiniLM-L6-v2
EMBED_OFFLINE=true
FAIL_ON_EMBEDDING_FALLBACK=true
```

`EMBED_OFFLINE=true` makes the service pass `local_files_only=True` to `SentenceTransformer`, so startup does not contact `huggingface.co`. If `EMBED_MODEL` points to a local path, the service also forces local-only loading and validates that the folder contains a saved SentenceTransformer model.

## Download Once

Windows PowerShell:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('storage/models/all-MiniLM-L6-v2')"
```

Linux/bash:

```bash
python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").save("storage/models/all-MiniLM-L6-v2")'
```

Then restart the API process:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## Docker/Linux Deployment

Before starting the container:

- download the model to `storage/models/all-MiniLM-L6-v2`;
- set `EMBED_MODEL=storage/models/all-MiniLM-L6-v2`;
- set `EMBED_OFFLINE=true`;
- make sure `storage/models/all-MiniLM-L6-v2` is copied into the image or mounted as a volume.

For `docker-compose.vps.yml`, the app service mounts local models read-only:

```yaml
volumes:
  - ./storage/models:/app/storage/models:ro
```

If you prefer baking the model into an image, keep the same relative path inside the image, for example:

```dockerfile
COPY storage/models/all-MiniLM-L6-v2 storage/models/all-MiniLM-L6-v2
```

Do this only when the model folder is present during `docker build`; otherwise use the volume mount.
