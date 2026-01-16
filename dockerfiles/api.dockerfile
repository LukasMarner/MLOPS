# API Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files and README (needed for package build)
COPY pyproject.toml uv.lock README.md ./

# Copy source code (needed for package build)
COPY src/ ./src/
COPY configs/ ./configs/

# Install dependencies (now it can build the package)
RUN uv sync --frozen

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port for FastAPI
EXPOSE 8000

# Default command (can be overridden)
CMD ["uv", "run", "uvicorn", "mlops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]

