from prometheus_client import Counter, Histogram, Gauge

# 1. API Level Metrics
REQUEST_COUNT = Counter(
    "fastapi_requests_total",
    "Total number of HTTP requests processed by FastAPI",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "fastapi_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

# 2. Inference Specific Metrics
INFERENCE_LATENCY = Histogram(
    "colorizer_inference_duration_seconds",
    "Pure GPU/CPU inference time in seconds",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
)

INFERENCE_QUEUE_SIZE = Gauge(
    "colorizer_inference_queue_size",
    "Current number of requests waiting for the semaphore to enter the GPU"
)

# 3. Hardware / Memory Metrics
GPU_MEMORY_ALLOCATED_MB = Gauge(
    "colorizer_gpu_memory_allocated_mb",
    "Current allocated GPU VRAM in Megabytes"
)

FILE_UPLOAD_SIZE = Histogram(
    "colorizer_upload_file_size_mb",
    "Uploaded image file sizes in Megabytes",
    buckets=[0.5, 1.0, 2.5, 5.0, 8.0, 10.0, 15.0]
)
