#!/usr/bin/env python3
"""Generate a secure API key and print setup instructions."""
import secrets
import sys
from pathlib import Path

key = secrets.token_urlsafe(32)

print("\n=== Emotion Voice — API Key Generator ===\n")
print(f"Your API key:\n\n  {key}\n")
print("To enable authentication, set these environment variables:\n")
print(f"  export API_KEY=\"{key}\"")
print("  export REQUIRE_API_KEY=true\n")
print("Or add to .env file:")
print(f"  API_KEY={key}")
print("  REQUIRE_API_KEY=true\n")
print("Use in requests:")
print("  curl -H 'X-API-Key: <your-key>' http://localhost:8000/tts ...\n")

# Optionally write to .env
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    env_path.write_text(
        f"# Emotion Voice configuration\n"
        f"API_KEY={key}\n"
        f"REQUIRE_API_KEY=true\n"
        f"DEVICE=cpu\n"
        f"TTS_MODEL_TYPE=vits\n"
    )
    print(f"✓ Written to {env_path} (do not commit this file to git)\n")
else:
    print(f"Note: {env_path} already exists — update it manually.\n")
