# syntax=docker/dockerfile:1

# ------------------------------------------------------------------------------
# BUILDER STAGE
# Compile dependencies in an isolated stage to keep the final image clean
# ------------------------------------------------------------------------------
# Using an official NVIDIA CUDA base image compatible with PyTorch 2.0+
# Note: For strict CPU-only environments, switch to python:3.11-slim
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

# Setup Python environment on base Ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to seamlessly transfer to final stage
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install dependencies into the completely isolated virtual environment
# We use --no-cache-dir because the builder layer size doesn't strictly matter
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------------------------
# FINAL RUNTIME STAGE
# ------------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Define environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    WORKERS=1

# Install strict runtime dependencies (OpenCV/skimage require libgl dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root system user for strict security compliance
RUN groupadd -r fastapi && useradd -r -g fastapi fastapi

WORKDIR /app

# Copy the pre-built, isolated virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Install production specific servers
RUN pip install gunicorn==21.2.0

# Copy application code into container, asserting strict ownership to our secure wrapper user
COPY --chown=fastapi:fastapi . /app

# Expose standard FastAPI port
EXPOSE 8000

# Downgrade privileges to prevent root escalation paths inside the container
USER fastapi

# Native Docker healthchecking utilizing FastAPI's /health route
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Execute the ASGI server. Uvicorn dynamically binds to the PORT env variable automatically.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --proxy-headers
