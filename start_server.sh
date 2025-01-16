#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export FLASK_APP=server.py

# Load environment variables if .env exists
if [ -f .env ]; then
    source .env
fi

# Use PORT from environment or default to 8787
PORT="${PORT:-8787}"

# Start the server using gunicorn
gunicorn --bind "0.0.0.0:${PORT}" \
         --workers "${WORKERS:-4}" \
         --timeout "${TIMEOUT:-30}" \
         --access-logfile - \
         --error-logfile - \
         'server:create_app()'

