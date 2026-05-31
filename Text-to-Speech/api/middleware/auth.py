"""API key authentication middleware.

Checks X-API-Key header (or ?api_key= query param) against the configured key.
Disabled when require_api_key=false or api_key is empty.

Set your key:
  export API_KEY="your-secret-key-here"
  export REQUIRE_API_KEY=true
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Paths that are always public (no auth required)
PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates X-API-Key header (or api_key query param) for protected routes."""

    def __init__(self, app, api_key: str, enabled: bool = True):
        super().__init__(app)
        # Store hashed key — never compare raw secrets directly
        self._key_hash = hashlib.sha256(api_key.encode()).hexdigest() if api_key else ""
        self.enabled = enabled and bool(api_key)

        if self.enabled:
            logger.info("API key authentication ENABLED")
        else:
            logger.warning("API key authentication DISABLED — set API_KEY env var to enable")

    def _verify(self, provided: str) -> bool:
        if not provided:
            return False
        provided_hash = hashlib.sha256(provided.encode()).hexdigest()
        return hmac.compare_digest(provided_hash, self._key_hash)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Skip public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Skip WebSocket upgrade (auth handled in WS handler)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # Try X-API-Key header first, then query param
        api_key = (
            request.headers.get("X-API-Key", "")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            or request.query_params.get("api_key", "")
        )

        if not self._verify(api_key):
            logger.warning(f"Unauthorized access attempt: {request.method} {request.url.path} "
                           f"from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Unauthorized. Provide a valid API key via X-API-Key header.",
                    "hint": "Set X-API-Key: <your-key> in the request header.",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)


def generate_api_key() -> str:
    """Generate a cryptographically secure random API key."""
    return secrets.token_urlsafe(32)
