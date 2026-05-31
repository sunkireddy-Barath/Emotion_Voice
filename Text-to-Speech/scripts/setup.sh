#!/bin/bash
# Full first-time setup: install deps, generate API key, download models
set -e

cd "$(dirname "$0")/.."

echo "==================================================="
echo "  Emotion Voice — First-Time Setup"
echo "==================================================="
echo ""

# ── 1. System packages ──────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        ffmpeg libsndfile1 libsndfile1-dev build-essential \
        portaudio19-dev espeak-ng espeak python3-dev \
        2>/dev/null || true
elif command -v brew &>/dev/null; then
    brew install ffmpeg libsndfile portaudio espeak 2>/dev/null || true
fi
echo "  ✓ System packages"

# ── 2. Python packages ──────────────────────────────────────────────────────
echo "[2/6] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  ✓ Python packages"

# ── 3. Create directories ───────────────────────────────────────────────────
echo "[3/6] Creating directory structure..."
mkdir -p \
    data/{raw,processed/{mel_specs,prosody_features,cleaned_wav},voice_samples,voice_profiles,datasets,metadata} \
    models/{tts_model,emotion_classifier,prosody_predictor,vocoder} \
    training/{checkpoints/{emotion,prosody,tts},logs} \
    logs
echo "  ✓ Directories created"

# ── 4. Generate API key ─────────────────────────────────────────────────────
echo "[4/6] Generating API key..."
python3 scripts/generate_api_key.py
echo "  ✓ API key generated"

# ── 5. Download TTS model ───────────────────────────────────────────────────
echo "[5/6] Downloading TTS model (VITS, ~200MB)..."
python3 scripts/download_models.py --model vits
echo "  ✓ TTS model downloaded"

# ── 6. Download emotion model ───────────────────────────────────────────────
echo "[6/6] Downloading emotion model (~300MB)..."
python3 scripts/download_models.py --model emotion
echo "  ✓ Emotion model downloaded"

# ── Frontend ────────────────────────────────────────────────────────────────
if command -v npm &>/dev/null; then
    echo ""
    echo "[Optional] Installing frontend dependencies..."
    cd frontend && npm install --silent && cd ..
    echo "  ✓ Frontend ready"
fi

echo ""
echo "==================================================="
echo "  Setup Complete!"
echo "==================================================="
echo ""
echo "  Start API server:"
echo "    ./scripts/start_api.sh"
echo ""
echo "  Start frontend (in a separate terminal):"
echo "    ./scripts/start_frontend.sh"
echo ""
echo "  Record your voice (pick an emotion):"
echo "    ./scripts/record_voice.sh neutral"
echo "    ./scripts/record_voice.sh happy"
echo ""
echo "  API docs:   http://localhost:8000/docs"
echo "  Frontend:   http://localhost:3000"
echo ""
