# ---- Base image: Python with CPU‑only TensorFlow ----
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install project ----
WORKDIR /app

# Copy project metadata first (for caching)
COPY pyproject.toml .
COPY README.md .

# Install dependencies (CPU TensorFlow)
RUN pip install --no-cache-dir .

# Copy the actual source code
COPY vnet ./vnet
COPY api ./api
COPY cli ./cli

# Expose FastAPI port
EXPOSE 8000

# Default command: run FastAPI server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
