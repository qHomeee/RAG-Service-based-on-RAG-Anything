#!/usr/bin/env bash
set -Eeuo pipefail

cd /opt/presentonika/RAG-Service-v2

REPARSE="${REPARSE:-true}"
if [[ "$REPARSE" != "true" && "$REPARSE" != "false" ]]; then
  echo "REPARSE must be true or false" >&2
  exit 2
fi

LOG_FILE="/root/rag-backups/reindex-$(date -u +%Y%m%d-%H%M%S).log"
STATUS_FILE="/root/rag-backups/reindex-latest.status"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "status=running session=rag-reindex started_at=$(date -u +%FT%TZ)" |
  tee "$STATUS_FILE"
echo "log=$LOG_FILE"

on_exit() {
  code=$?
  if [ "$code" -eq 0 ]; then
    state=completed
  else
    state=failed
  fi
  echo "status=$state exit_code=$code finished_at=$(date -u +%FT%TZ) log=$LOG_FILE" |
    tee "$STATUS_FILE"
}
trap on_exit EXIT

ADMIN_API_KEY="$(docker exec rag-service-app-1 printenv ADMIN_API_KEY)"

curl --fail-with-body --silent --show-error --max-time 0 \
  -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -d "{
    \"input_path\":\"/app/storage/raw\",
    \"collection\":\"default\",
    \"reindex\":true,
    \"reparse\":$REPARSE
  }"

echo
API_KEY="$(docker exec rag-service-app-1 printenv API_KEY)"
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/readyz \
  -H "X-API-Key: $API_KEY"
echo
