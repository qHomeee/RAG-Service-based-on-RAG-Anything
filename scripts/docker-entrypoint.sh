#!/bin/sh
set -eu

if [ -n "${PROMETHEUS_MULTIPROC_DIR:-}" ]; then
    case "$PROMETHEUS_MULTIPROC_DIR" in
        /tmp/prometheus)
            mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
            find "$PROMETHEUS_MULTIPROC_DIR" -mindepth 1 -maxdepth 1 -type f -delete
            ;;
        *)
            echo "Refusing to clean unexpected PROMETHEUS_MULTIPROC_DIR" >&2
            exit 1
            ;;
    esac
fi

exec "$@"
