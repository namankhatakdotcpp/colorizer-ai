import os
import logging
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi import FastAPI

logger = logging.getLogger("api.rate_limiter")

def get_client_ip(request: Request) -> str:
    """
    Safely extract the real client IP, even when hosted behind reverse proxies
    like Render, Nginx, or Cloudflare.
    """
    # 1. Check for standard proxy forwarding headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain a comma-separated list of IPs. The first is the actual client.
        client_ip = forwarded_for.split(",")[0].strip()
        return client_ip
        
    # 2. Check X-Real-IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
        
    # 3. Fallback to direct remote address
    return get_remote_address(request)


# Initialize the Limiter. 
# In Phase 1, this uses memory (`Dict`).
# For Phase 2 (Multiple Gunicorn Workers / Scaling), we can trivially inject a Redis backend here
# by passing `storage_uri="redis://..."`
limiter = Limiter(
    key_func=get_client_ip,
    default_limits=["5/minute"], # Default strict baseline
    strategy="fixed-window"
)

def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Format standard 429 Error responses nicely for the Next.js frontend to parse.
    """
    client_ip = get_client_ip(request)
    logger.warning(f"Rate limit exceeded for IP: {client_ip} on path {request.url.path}")
    
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Too many requests.",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "message": f"Maximum of 5 requests per minute permitted. Please wait.",
            "retry_after": exc.headers.get("Retry-After", 60)
        }
    )

def setup_rate_limiting(app: FastAPI):
    """
    Hooks the SlowAPI limitation engine cleanly into the FastAPI main app.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)
    logger.info("SlowAPI Rate Limiting configured (5 req/min per IP).")
