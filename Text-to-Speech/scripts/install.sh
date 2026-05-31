#!/bin/bash
# Install all dependencies for the Emotion Voice project
set -e

cd "$(dirname "$0")/.."

echo "=== Emotion Voice Installation ==="

# System dependencies
echo "1. Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ffmpeg libsndfile1 libsndfile1-dev build-essential \
    portaudio19-dev espeak-ng python3-dev

# Python dependencies
echo "2. Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "3. Creating directory structure..."
mkdir -p data/{raw,processed/{mel_specs,prosody_features,cleaned_wav},voice_samples,voice_profiles,datasets,metadata}
mkdir -p models/{tts_model,emotion_classifier,prosody_predictor,vocoder}
mkdir -p training/{checkpoints/{emotion,prosody,tts},logs}
mkdir -p logs

# Frontend dependencies
echo "4. Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "✓ Installation complete!"
echo ""
echo "Quick start:"
echo "  ./scripts/start_api.sh        # Start API server"
echo "  ./scripts/start_frontend.sh   # Start frontend"
echo "  ./scripts/record_voice.sh     # Record voice samples"
echo ""
echo "Or with Docker:"
echo "  docker-compose up --build"
