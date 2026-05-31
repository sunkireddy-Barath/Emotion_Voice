#!/bin/bash
# Start the Emotion Voice API server
# Usage: ./scripts/start_api.sh [--prod]
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
fi

# Create required directories
mkdir -p logs data/voice_profiles data/voice_samples models/tts_model models/emotion_classifier

PROD_MODE=false
if [[ "$1" == "--prod" ]]; then
    PROD_MODE=true
fi

echo "==================================================="
echo "  Emotion-Aware Speech Foundation Model API"
echo "==================================================="
echo "  URL     : http://0.0.0.0:${PORT:-8000}"
echo "  Docs    : http://localhost:${PORT:-8000}/docs"
echo "  Health  : http://localhost:${PORT:-8000}/health"
echo "  Metrics : http://localhost:${PORT:-8000}/metrics"
echo "  Mode    : $([ $PROD_MODE = true ] && echo 'PRODUCTION (gunicorn)' || echo 'DEVELOPMENT (uvicorn)')"
echo "  Auth    : $([ -n "$API_KEY" ] && echo 'ENABLED' || echo 'DISABLED (set API_KEY to enable)')"
echo "==================================================="
echo ""

if [ "$PROD_MODE" = true ]; then
    echo "Starting with gunicorn (production)..."
    exec gunicorn -c gunicorn.conf.py api.main:app
else
    echo "Starting with uvicorn (development — auto-reload enabled)..."
    exec uvicorn api.main:app \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8000}" \
        --workers 1 \
        --log-level info \
        --reload
fi
