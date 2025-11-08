# Production Deployment

Guide to deploying the PDF Q&A System in production environments.

## Overview

This guide covers best practices for deploying the system in production, including security, scalability, monitoring, and maintenance.

## Deployment Options

### Option 1: Docker Compose (Single Server)

Best for: Small to medium workloads, single server deployments

```yaml
# docker-compose.prod.yml
version: "3.8"

services:
  pdf-qa:
    build:
      context: .
      dockerfile: Dockerfile
    image: pdf-qa-system:latest
    restart: unless-stopped
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data:ro
      - chroma_data:/app/chroma_db
      - model_cache:/app/models
    ports:
      - "127.0.0.1:8000:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

volumes:
  chroma_data:
    driver: local
  model_cache:
    driver: local
```

**Deploy:**

```bash
# Build
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Health check
docker-compose -f docker-compose.prod.yml ps
```

### Option 2: Kubernetes (Scalable)

Best for: Large workloads, multi-server deployments

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pdf-qa-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pdf-qa
  template:
    metadata:
      labels:
        app: pdf-qa
    spec:
      containers:
      - name: pdf-qa
        image: ghcr.io/yourusername/pdf-qa-system:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: pdf-qa-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: chroma-storage
          mountPath: /app/chroma_db
        - name: model-cache
          mountPath: /app/models
      volumes:
      - name: chroma-storage
        persistentVolumeClaim:
          claimName: chroma-pvc
      - name: model-cache
        persistentVolumeClaim:
          claimName: models-pvc
```

**Deploy:**

```bash
# Create secrets
kubectl create secret generic pdf-qa-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY

# Deploy
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl logs -f deployment/pdf-qa-system
```

### Option 3: Cloud Platform (Managed)

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build -t pdf-qa-system .
docker tag pdf-qa-system:latest <account>.dkr.ecr.us-east-1.amazonaws.com/pdf-qa-system:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/pdf-qa-system:latest

# Create ECS task definition and service
# Use AWS Console or CLI
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/pdf-qa-system
gcloud run deploy pdf-qa-system \
  --image gcr.io/PROJECT-ID/pdf-qa-system \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
```

## Security Best Practices

### 1. API Key Management

**DO NOT** hardcode API keys. Use secrets management:

```bash
# Docker secrets
echo "sk-..." | docker secret create openai_api_key -

# Kubernetes secrets
kubectl create secret generic pdf-qa-secrets \
  --from-literal=openai-api-key="sk-..."

# AWS Secrets Manager
aws secretsmanager create-secret \
  --name pdf-qa/openai-key \
  --secret-string "sk-..."

# Environment variables (least secure)
export OPENAI_API_KEY="sk-..."
```

### 2. Network Security

```yaml
# Restrict access
services:
  pdf-qa:
    ports:
      - "127.0.0.1:8000:8000"  # Only localhost

# Use reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

### 3. Container Security

```dockerfile
# Run as non-root
USER appuser

# Read-only root filesystem
docker run --read-only --tmpfs /tmp pdf-qa-system

# Drop capabilities
docker run --cap-drop=ALL pdf-qa-system
```

### 4. Input Validation

```python
# Validate file uploads
import os
from pathlib import Path

def validate_pdf(file_path):
    # Check file exists
    if not os.path.exists(file_path):
        raise ValueError("File not found")

    # Check file size (e.g., max 50MB)
    if os.path.getsize(file_path) > 50 * 1024 * 1024:
        raise ValueError("File too large")

    # Check extension
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("Not a PDF file")

    return True
```

## Scalability

### Horizontal Scaling

```yaml
# Docker Swarm
services:
  pdf-qa:
    deploy:
      replicas: 5
      update_config:
        parallelism: 2
        delay: 10s

# Kubernetes
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

### Load Balancing

```nginx
# nginx.conf
upstream pdf_qa {
    least_conn;
    server pdf-qa-1:8000;
    server pdf-qa-2:8000;
    server pdf-qa-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://pdf_qa;
    }
}
```

### Caching

```python
# Cache embeddings
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_embed(text_hash):
    return embedder.embed_text(text)

def embed_with_cache(text):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return cached_embed(text_hash)
```

## Monitoring

### Health Checks

```python
# health.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    try:
        # Check database connection
        count = store.get_count()

        # Check LLM availability
        llm_available = llm.is_available()

        return jsonify({
            "status": "healthy",
            "database": "connected",
            "documents": count,
            "llm": "available" if llm_available else "unavailable"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
```

### Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[handler, logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Use in code
logger.info(f"Processing PDF: {pdf_path}")
logger.error(f"Failed to process: {error}")
```

### Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
requests_total = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
embeddings_generated = Counter('embeddings_generated', 'Embeddings generated')

# Track metrics
requests_total.inc()
with request_duration.time():
    result = process_document(pdf_path)
embeddings_generated.inc(len(chunks))

# Expose metrics
start_http_server(9090)
```

## Backup and Recovery

### Database Backup

```bash
# Backup ChromaDB
docker run --rm \
  -v chroma_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/chroma_$(date +%Y%m%d).tar.gz -C /data .

# Restore
docker run --rm \
  -v chroma_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/chroma_20240101.tar.gz -C /data
```

### Automated Backups

```bash
# backup.sh
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker run --rm \
  -v chroma_data:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/chroma_$DATE.tar.gz -C /data .

# Keep only last 7 days
find $BACKUP_DIR -name "chroma_*.tar.gz" -mtime +7 -delete

# crontab
# 0 2 * * * /path/to/backup.sh
```

## Performance Optimization

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

### Connection Pooling

```python
# For database connections
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    )
)
```

## Troubleshooting

### Check Logs

```bash
# Docker
docker logs pdf-qa-system

# Kubernetes
kubectl logs -f deployment/pdf-qa-system

# System logs
journalctl -u pdf-qa.service -f
```

### Common Issues

**Out of Memory**

```bash
# Increase memory limit
docker run --memory="8g" pdf-qa-system
```

**Slow Performance**

```python
# Reduce batch size
chunks = chunker.chunk_text(text)
for i in range(0, len(chunks), 50):
    batch = chunks[i:i+50]
    embeddings = embedder.embed_batch(batch)
```

## Maintenance

### Updates

```bash
# Rolling update (zero downtime)
docker-compose -f docker-compose.prod.yml up -d --no-deps --build pdf-qa

# Kubernetes
kubectl set image deployment/pdf-qa-system pdf-qa=pdf-qa-system:v2.0
kubectl rollout status deployment/pdf-qa-system
```

### Cleanup

```bash
# Remove old images
docker image prune -a

# Clean up logs
find /var/log -name "*.log" -mtime +30 -delete
```

## Next Steps

- [Environment Configuration](environment.md)
- [Docker Guide](../user-guide/docker.md)
- [Troubleshooting](../user-guide/troubleshooting.md)
