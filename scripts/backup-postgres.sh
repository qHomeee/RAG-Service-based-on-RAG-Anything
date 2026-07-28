#!/bin/sh
set -eu

umask 077

backup_dir="${RAG_BACKUP_DIR:-/root/rag-backups}"
retention_days="${RAG_BACKUP_RETENTION_DAYS:-14}"
postgres_container="${RAG_POSTGRES_CONTAINER:-rag-service-postgres-1}"
postgres_user="${RAG_POSTGRES_USER:-rag}"
postgres_db="${RAG_POSTGRES_DB:-rag}"

case "$backup_dir" in
  /*) ;;
  *)
    echo "RAG_BACKUP_DIR must be an absolute path" >&2
    exit 2
    ;;
esac
case "$backup_dir" in
  /|/root|/var|/opt)
    echo "RAG_BACKUP_DIR is too broad: $backup_dir" >&2
    exit 2
    ;;
esac
case "$retention_days" in
  *[!0-9]*|"")
    echo "RAG_BACKUP_RETENTION_DAYS must be a non-negative integer" >&2
    exit 2
    ;;
esac

mkdir -p "$backup_dir"
timestamp="$(date -u +%Y%m%d-%H%M%SZ)"
final_path="$backup_dir/rag-logical-$timestamp.dump"
partial_path="$final_path.partial"

cleanup() {
  rm -f -- "$partial_path"
}
trap cleanup EXIT HUP INT TERM

docker inspect "$postgres_container" >/dev/null
docker exec "$postgres_container" \
  pg_dump --format=custom --no-owner --no-acl \
  --username="$postgres_user" --dbname="$postgres_db" >"$partial_path"
test -s "$partial_path"
mv -- "$partial_path" "$final_path"
sha256sum "$final_path" >"$final_path.sha256"

find "$backup_dir" -maxdepth 1 -type f \
  \( -name 'rag-logical-*.dump' -o -name 'rag-logical-*.dump.sha256' \) \
  -mtime "+$retention_days" -delete

trap - EXIT HUP INT TERM
echo "backup=$final_path"
echo "checksum=$final_path.sha256"
