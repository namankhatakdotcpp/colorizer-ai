import io
from fastapi.testclient import TestClient
from PIL import Image

# Import the FastAPI app
from app.main import app
from app.core.rate_limiter import limiter

# Disable rate limiting strictly during Pytest so tests don't get blocked
limiter.enabled = False

def test_read_root():
    # Wrap in app testclient with context manager to trigger lifespan events
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_colorize_endpoint_with_valid_image():
    # Create a simple valid image (grayscale)
    img = Image.new('RGB', (100, 100), color='gray')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    with TestClient(app) as client:
        response = client.post(
            "/colorize",
            files={"file": ("test_image.jpg", img_bytes, "image/jpeg")}
        )
        
        # Check if we get a 200 OK and a valid image response
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
    
def test_colorize_endpoint_with_invalid_file():
    with TestClient(app) as client:
        response = client.post(
            "/colorize",
            files={"file": ("test.txt", b"Hello world", "text/plain")}
        )
        assert response.status_code == 415
        assert "Invalid file type" in response.json()["detail"]

def test_admin_metrics_unauthorized():
    with TestClient(app) as client:
        response = client.get("/admin/metrics")
        assert response.status_code == 403
        
def test_admin_metrics_authorized():
    from app.config import settings
    with TestClient(app) as client:
        response = client.get("/admin/metrics", headers={"X-Admin-Key": settings.ADMIN_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "active_inference_queue" in data

