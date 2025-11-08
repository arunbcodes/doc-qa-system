# Docker Deployment

Complete guide to deploying the PDF Q&A System with Docker.

## Quick Start

### Build Image

```bash
docker build -t pdf-qa-system:latest .
```

### Run Container

```bash
# Phase 1 (Semantic Search)
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  pdf-qa-system:latest \
  python main.py /app/data/sample.pdf

# Phase 2 (RAG with LLM)
docker run -it --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/data:/app/data \
  pdf-qa-system:latest \
  python main_rag.py /app/data/sample.pdf
```

## Docker Compose

### Phase 1: Semantic Search

```bash
docker-compose --profile search run --rm pdf-qa-search
```

### Phase 2: RAG with LLM

```bash
docker-compose --profile rag run --rm pdf-qa-rag
```

### With Ollama (Local LLM)

```bash
# Start Ollama service
docker-compose --profile ollama up -d ollama

# Pull model
docker-compose exec ollama ollama pull llama3.2

# Run RAG
docker-compose --profile rag run --rm pdf-qa-rag
```

## Configuration

### Environment Variables

Pass environment variables with `-e`:

```bash
docker run -it --rm \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e CHUNK_SIZE=1000 \
  -e CHUNK_OVERLAP=100 \
  -v $(pwd)/data:/app/data \
  pdf-qa-system:latest \
  python main_rag.py /app/data/sample.pdf
```

### Volume Mounts

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data:ro \           # PDF files (read-only)
  -v $(pwd)/output:/app/output \          # Output files
  -v pdf-qa-models:/app/models \          # Model cache
  -v pdf-qa-db:/app/.cache \              # Vector DB
  pdf-qa-system:latest \
  python main.py /app/data/sample.pdf
```

### Custom Dockerfile

```dockerfile
FROM pdf-qa-system:latest

# Add custom dependencies
RUN pip install custom-package

# Copy custom configuration
COPY config.yaml /app/

# Set custom entrypoint
ENTRYPOINT ["python", "custom_script.py"]
```

## Production Deployment

### Multi-Stage Build

The Dockerfile uses multi-stage builds for optimization:

```dockerfile
# Stage 1: Base (system dependencies)
FROM python:3.11-slim as base
RUN apt-get update && apt-get install -y build-essential

# Stage 2: Builder (Python dependencies)
FROM base as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 3: Runtime (minimal final image)
FROM base as runtime
COPY --from=builder /root/.local /home/appuser/.local
COPY src/ ./src/
```

**Benefits:**
- Smaller final image (~600MB)
- No build tools in production
- Faster deployments

### Health Checks

Built-in health check:

```bash
docker run --rm pdf-qa-system:latest health
```

Docker Compose with health check:

```yaml
services:
  pdf-qa:
    image: pdf-qa-system:latest
    healthcheck:
      test: ["CMD", "python", "-c", "import src"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Resource Limits

```bash
docker run -it --rm \
  --memory=4g \
  --cpus=2 \
  -v $(pwd)/data:/app/data \
  pdf-qa-system:latest \
  python main.py /app/data/sample.pdf
```

Docker Compose:

```yaml
services:
  pdf-qa:
    image: pdf-qa-system:latest
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Kubernetes Deployment

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pdf-qa-system
spec:
  replicas: 2
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
        image: ghcr.io/arunbcodes/doc-qa-system:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-key
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: pdf-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### Secret Management

```bash
# Create secret
kubectl create secret generic llm-secrets \
  --from-literal=openai-key=sk-...

# Use in deployment
env:
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: llm-secrets
      key: openai-key
```

## Container Registry

### GitHub Container Registry

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag
docker tag pdf-qa-system:latest ghcr.io/arunbcodes/doc-qa-system:latest

# Push
docker push ghcr.io/arunbcodes/doc-qa-system:latest

# Pull
docker pull ghcr.io/arunbcodes/doc-qa-system:latest
```

### Docker Hub

```bash
# Login
docker login

# Tag
docker tag pdf-qa-system:latest arunbcodes/pdf-qa-system:latest

# Push
docker push arunbcodes/pdf-qa-system:latest
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker logs <container-id>
```

### Volume Permissions

Fix permissions:
```bash
# On host
chmod -R 755 data/

# Or run as root (not recommended for production)
docker run --user root ...
```

### Memory Issues

Increase memory limit:
```bash
docker run --memory=8g ...
```

### Network Issues with Ollama

If Ollama is on host:
```bash
docker run --add-host=host.docker.internal:host-gateway ...
```

## Next Steps

- [Production Deployment Guide](production.md)
- [Environment Variables](environment.md)
- [Troubleshooting](../user-guide/troubleshooting.md)
