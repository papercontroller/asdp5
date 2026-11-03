# Multi-stage Dockerfile for Campus IoT Intrusion Detection System
# Base Python runtime
FROM python:3.11-slim AS base

# Set working directory & environment
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and tests
COPY src/ /app/src/
COPY tests/ /app/tests/

# Pre-train initial baseline model
RUN python -m src.ids.cli --train --output /app/campus_ids_model.joblib

# Stage for running CI tests inside Docker
FROM base AS test
CMD ["pytest", "-v", "--tb=short", "tests/"]

# Production deployment stage
FROM base AS release
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os; assert os.path.exists('/app/campus_ids_model.joblib')" || exit 1

CMD ["python", "-m", "src.ids.cli", "--test-flow", "--attack-type", "DoS_Publish_Flood", "--output", "/app/campus_ids_model.joblib"]
