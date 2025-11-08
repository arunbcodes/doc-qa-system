# Docker Deployment

Complete guide to running the PDF Q&A System with Docker.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/arunbcodes/doc-qa-system.git
cd doc-qa-system

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
nano .env

# Run with Docker Compose
docker-compose up
```

## Docker Compose

### Basic Setup

The `docker-compose.yml` provides a complete setup:

```yaml
services:
  pdf-qa:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - chroma_data:/app/chroma_db
    ports:
      - "8000:8000"
```

### With Ollama (Local LLM)

```yaml
services:
  pdf-qa:
    build: .
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
```

### Running Services

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f pdf-qa

# Stop services
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v
```

## Docker CLI

### Building the Image

```bash
# Basic build
docker build -t pdf-qa-system .

# Build with specific tag
docker build -t pdf-qa-system:2.0.0 .

# Build without cache
docker build --no-cache -t pdf-qa-system .
```

### Running Containers

#### Interactive Mode

```bash
# Run with environment variables
docker run -it --rm \
  -e OPENAI_API_KEY="sk-..." \
  -v $(pwd)/data:/app/data \
  pdf-qa-system
```

#### Process a PDF

```bash
# Phase 1 (Semantic Search only)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v chroma_data:/app/chroma_db \
  pdf-qa-system python main.py data/document.pdf

# Phase 2 (RAG with LLM)
docker run --rm \
  -e OPENAI_API_KEY="sk-..." \
  -v $(pwd)/data:/app/data \
  -v chroma_data:/app/chroma_db \
  pdf-qa-system python main_rag.py data/document.pdf
```

#### Background Service

```bash
# Run as daemon
docker run -d \
  --name pdf-qa \
  -e OPENAI_API_KEY="sk-..." \
  -v $(pwd)/data:/app/data \
  pdf-qa-system

# View logs
docker logs -f pdf-qa

# Execute commands
docker exec -it pdf-qa python main.py data/doc.pdf

# Stop container
docker stop pdf-qa
docker rm pdf-qa
```

## Volume Management

### Data Persistence

```bash
# Create named volumes
docker volume create chroma_data
docker volume create model_cache

# Run with named volumes
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v chroma_data:/app/chroma_db \
  -v model_cache:/app/models \
  pdf-qa-system
```

### Backup and Restore

```bash
# Backup ChromaDB
docker run --rm \
  -v chroma_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/chroma_backup.tar.gz -C /data .

# Restore ChromaDB
docker run --rm \
  -v chroma_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/chroma_backup.tar.gz -C /data
```

## Environment Configuration

### Using .env File

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
CHUNK_SIZE=500
N_RESULTS=3
LOG_LEVEL=INFO
EOF

# Run with .env
docker run --rm --env-file .env pdf-qa-system
```

### Docker Compose .env

```yaml
# docker-compose.yml
services:
  pdf-qa:
    env_file: .env
    environment:
      - CHUNK_SIZE=1000  # Override .env
```

## Multi-Stage Build

The Dockerfile uses multi-stage builds for optimization:

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
```

**Benefits:**
- Smaller final image (~800MB vs 1.5GB)
- Faster builds with layer caching
- More secure (no build tools in runtime)

## Health Checks

Add health checks to monitor container status:

```yaml
services:
  pdf-qa:
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Resource Limits

Control container resources:

```yaml
services:
  pdf-qa:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

Or with Docker CLI:

```bash
docker run --rm \
  --cpus="2.0" \
  --memory="4g" \
  pdf-qa-system
```

## Development Setup

### Hot Reload

```yaml
services:
  pdf-qa:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - /app/.venv  # Don't mount venv
    command: python main.py data/doc.pdf
```

### Debug Mode

```bash
# Run with pdb
docker run -it --rm \
  -v $(pwd):/app \
  pdf-qa-system python -m pdb main.py data/doc.pdf

# With debugpy for VS Code
docker run -it --rm \
  -p 5678:5678 \
  -v $(pwd):/app \
  pdf-qa-system \
  python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
```

## Networking

### Connect Multiple Containers

```yaml
services:
  pdf-qa:
    networks:
      - app-network

  ollama:
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

### Expose Ports

```yaml
services:
  pdf-qa:
    ports:
      - "8000:8000"  # host:container
      - "127.0.0.1:8001:8001"  # Bind to localhost only
```

## Security

### Run as Non-Root

```dockerfile
# Create user
RUN useradd -m -u 1000 appuser
USER appuser
```

### Scan for Vulnerabilities

```bash
# With Trivy
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image pdf-qa-system

# With Docker Scout
docker scout cves pdf-qa-system
```

### Secrets Management

```yaml
services:
  pdf-qa:
    secrets:
      - openai_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key

secrets:
  openai_key:
    file: ./secrets/openai_api_key.txt
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs pdf-qa

# Inspect container
docker inspect pdf-qa

# Check resource usage
docker stats pdf-qa
```

### Permission Issues

```bash
# Fix volume permissions
docker run --rm -v chroma_data:/data alpine chown -R 1000:1000 /data

# Run as specific user
docker run --rm --user 1000:1000 pdf-qa-system
```

### Out of Memory

```bash
# Increase memory limit
docker run --rm --memory="8g" pdf-qa-system

# Or in docker-compose.yml
services:
  pdf-qa:
    mem_limit: 8g
```

### Slow Performance

```bash
# Check resource usage
docker stats

# Increase CPU allocation
docker run --rm --cpus="4.0" pdf-qa-system

# Use tmpfs for temporary data
docker run --rm --tmpfs /tmp:rw,size=2g pdf-qa-system
```

## Production Deployment

### Docker Hub

```bash
# Login
docker login

# Tag image
docker tag pdf-qa-system username/pdf-qa-system:2.0.0

# Push
docker push username/pdf-qa-system:2.0.0

# Pull and run
docker run --rm username/pdf-qa-system:2.0.0
```

### GitHub Container Registry

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag
docker tag pdf-qa-system ghcr.io/username/pdf-qa-system:2.0.0

# Push
docker push ghcr.io/username/pdf-qa-system:2.0.0
```

### Docker Swarm

```yaml
version: "3.8"
services:
  pdf-qa:
    image: pdf-qa-system:2.0.0
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

Deploy:

```bash
docker stack deploy -c docker-compose.yml pdf-qa-stack
```

## Next Steps

- [Configuration Guide](configuration.md)
- [Production Deployment](../deployment/production.md)
- [Troubleshooting](troubleshooting.md)
