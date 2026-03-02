import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import make_asgi_app

from app.config import settings
from app.services.inference_service import InferenceService
from app.core.logging_config import setup_logging
from app.core.middleware import RequestMetricsMiddleware
from app.core.rate_limiter import setup_rate_limiting, limiter
from app.core.internal_metrics import metrics_tracker

# Configure JSON structured logging globally
setup_logging()
logger = logging.getLogger(__name__)

# Global container for our model service to ensure it loads exactly once
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI. 
    Loads the heavy PyTorch model into memory right before the API starts accepting requests.
    Cleans it up on shutdown.
    """
    logger.info("Starting up FastAPI Lifecycle...")
    ml_models["colorizer"] = InferenceService()
    yield # API runs while this is yielding
    logger.info("Shutting down API. Cleaning up ML resources...")
    if "colorizer" in ml_models:
        ml_models["colorizer"].graceful_shutdown()
    ml_models.clear()

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# ✅ FIRST MIDDLEWARE — CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Mount Prometheus HTTP Endpoint at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 1. Bind custom Request Metrics tracking
app.add_middleware(RequestMetricsMiddleware)

# 2. Hook up Rate Limiting (SlowAPI)
setup_rate_limiting(app)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/health", tags=["System"])
def health():
    """Simple health check endpoint for Docker / Load Balancers."""
    return {"status": "ok"}

# ---------------------------------------------------------
# Admin MVP Dashboard Tracking
# ---------------------------------------------------------
api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials")
    return api_key

@app.get("/admin/metrics", tags=["Admin"])
async def internal_dashboard_metrics(api_key: str = Security(get_api_key)):
    """Simple in-memory MVP dashboard returning backend statistics."""
    return metrics_tracker.get_stats()
# ---------------------------------------------------------


@app.post("/colorize", tags=["Inference"])
@limiter.limit("5/minute")
async def colorize_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Main endpoint for colorizing uploaded black and white images.
    - Validates MIME type
    - Limits file size
    - Processes async to not block event loop during I/O
    """
    # 1. MIME Type Validation
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_MIME_TYPES}"
        )

    # 2. File Size Validation (Soft limit by reading bytes)
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB. Found {file_size_mb:.2f}MB"
        )
        
    # 3. Model Inference execution
    try:
        # Offload CPU bound ML operations out of the async loop using async guarantees.
        inference_svc: InferenceService = ml_models["colorizer"]
        
        # Grab correlation ID set by the middleware
        request_id = getattr(request.state, "request_id", "internal")
        
        logger.info(f"Starting async colorization for image size: {file_size_mb:.2f}MB", extra={"request_id": request_id})
        result_bytes = await inference_svc.colorize_async(file_bytes, request_id)
        
        return Response(content=result_bytes, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error processing image inference: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during image processing."
        )
