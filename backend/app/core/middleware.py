import time
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY, FILE_UPLOAD_SIZE
from app.core.internal_metrics import metrics_tracker

logger = logging.getLogger("api.middleware")

class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track API request lifecycles.
    Generates Correlation IDs and logs total request durations.
    """
    async def dispatch(self, request: Request, call_next):
        # 1. Generate Unique Request ID
        request_id = str(uuid.uuid4())
        client_ip = request.client.host if request.client else "unknown"
        
        # Attach request_id to fastAPI request state for downstream use in services
        request.state.request_id = request_id
        
        start_time = time.perf_counter()
        
        try:
            # 2. Process Request
            response = await call_next(request)
            
            process_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract content_length to log payload size
            content_length = request.headers.get("content-length")
            upload_size_mb = round(int(content_length) / (1024 * 1024), 2) if content_length else 0
            
            # Record Prometheus Metrics
            path = request.url.path
            REQUEST_COUNT.labels(method=request.method, endpoint=path, status=response.status_code).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(process_time_ms / 1000.0)
            if path == "/colorize" and upload_size_mb > 0:
                FILE_UPLOAD_SIZE.observe(upload_size_mb)
            
            # 3. Log Success
            # Avoid logging excessive data for basic health checks
            if request.url.path not in ["/health", "/metrics", "/admin/metrics"]:
                metrics_tracker.record_request(process_time_ms)
                logger.info("Request processed successfully", extra={
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(process_time_ms, 2),
                    "upload_size_mb": upload_size_mb
                })
            
            # Add correlation ID to response headers for client debugging
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            process_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Record failed metric
            path = request.url.path
            REQUEST_COUNT.labels(method=request.method, endpoint=path, status=500).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(process_time_ms / 1000.0)
            
            logger.error("Request failed unexpectedly", exc_info=True, extra={
                "request_id": request_id,
                "client_ip": client_ip,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": round(process_time_ms, 2)
            })
            raise
