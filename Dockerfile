# Multi-stage Dockerfile for PDF Q&A System
# Best practices: small image size, security, caching optimization

# Stage 1: Base image with Python and system dependencies
FROM python:3.14-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required by docling and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Stage 2: Dependencies installation
FROM base as builder

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 3: Final runtime image
FROM base as runtime

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py main_rag.py ./

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/.cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add user's local bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH \
    HOME=/home/appuser \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

# Health check script
COPY --chown=appuser:appuser docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Expose port (if running as API in future)
EXPOSE 8000

# Volume for persistent data
VOLUME ["/app/data", "/app/models"]

# Default entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "main.py", "--help"]
