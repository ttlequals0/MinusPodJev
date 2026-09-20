#!/bin/sh
set -e

echo "Starting jevproxy..."
mkdir -p /app/logs

# nginx (8080) + uvicorn (8000) are supervised together.
exec /opt/venv/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
