import time
from collections import defaultdict
from typing import Dict, List

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _get_client_id(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health check and websocket
        if request.url.path in {"/health", "/docs", "/openapi.json"} or \
                request.url.path.startswith("/ws/"):
            return await call_next(request)

        client_id = self._get_client_id(request)
        now = time.monotonic()
        window = 60.0

        # Remove requests older than window
        self._requests[client_id] = [
            t for t in self._requests[client_id] if now - t < window
        ]

        if len(self._requests[client_id]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers={"Retry-After": "60"},
            )

        self._requests[client_id].append(now)
        return await call_next(request)
