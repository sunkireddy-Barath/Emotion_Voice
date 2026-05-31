"""Gunicorn + Uvicorn production server configuration.

Usage:
    gunicorn -c gunicorn.conf.py api.main:app
"""
import multiprocessing
import os

# ─── Workers ──────────────────────────────────────────────────────────────────
# Use 1 worker for TTS (models are large — multiple workers multiply RAM usage)
# Scale horizontally with Docker/K8s instead of multiple workers per process
workers = int(os.environ.get("GUNICORN_WORKERS", 1))
worker_class = "uvicorn.workers.UvicornWorker"
threads = int(os.environ.get("GUNICORN_THREADS", 4))

# ─── Network ──────────────────────────────────────────────────────────────────
bind = f"{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8000')}"
backlog = 2048

# ─── Timeouts ─────────────────────────────────────────────────────────────────
# TTS can take 10-30s on CPU — set generous timeouts
timeout = 120          # worker timeout
graceful_timeout = 30  # time to finish active requests on shutdown
keepalive = 5

# ─── Logging ──────────────────────────────────────────────────────────────────
loglevel = os.environ.get("LOG_LEVEL", "info")
accesslog = "logs/access.log"
errorlog  = "logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ─── Process naming ───────────────────────────────────────────────────────────
proc_name = "emotion-voice-api"

# ─── SSL (optional — use nginx/traefik for TLS termination instead) ───────────
# keyfile  = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"
