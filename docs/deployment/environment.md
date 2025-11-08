# Environment Setup

Complete guide to configuring environments for development, testing, and production.

## Environment Variables

### Core Configuration

```bash
# .env
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
N_RESULTS=3

# Storage
CHROMA_PERSIST_DIR=./chroma_db
HF_HOME=./models
TRANSFORMERS_CACHE=./models

# Application
PYTHONUNBUFFERED=1
LOG_LEVEL=INFO
```

## Environment Types

### Development

```bash
# .env.development
LOG_LEVEL=DEBUG
CHUNK_SIZE=300
N_RESULTS=2

# Use local models for faster iteration
OLLAMA_MODEL=llama3.2
# Or use mock LLM
USE_MOCK_LLM=true
```

**Setup:**

```bash
cp .env.development .env
source .env
python main_rag.py data/test.pdf
```

### Testing

```bash
# .env.test
LOG_LEVEL=WARNING
USE_MOCK_LLM=true
CHROMA_PERSIST_DIR=./test_db

# Smaller models for faster tests
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=200
```

**Setup:**

```bash
# Run tests with test environment
export $(cat .env.test | xargs)
pytest
```

### Staging

```bash
# .env.staging
LOG_LEVEL=INFO
OPENAI_API_KEY=${OPENAI_API_KEY}
CHUNK_SIZE=500
N_RESULTS=3
CHROMA_PERSIST_DIR=/data/staging/chroma_db
```

### Production

```bash
# .env.production
LOG_LEVEL=WARNING
OPENAI_API_KEY=${OPENAI_API_KEY}
CHUNK_SIZE=500
N_RESULTS=3
CHROMA_PERSIST_DIR=/data/production/chroma_db

# Performance settings
MAX_WORKERS=4
BATCH_SIZE=100
```

## Loading Environment Variables

### Python (dotenv)

```python
from dotenv import load_dotenv
import os

# Load from specific file
load_dotenv('.env.production')

# Access variables
api_key = os.getenv('OPENAI_API_KEY')
chunk_size = int(os.getenv('CHUNK_SIZE', 500))
```

### Docker

```yaml
# docker-compose.yml
services:
  pdf-qa:
    env_file:
      - .env.production
    environment:
      - LOG_LEVEL=INFO  # Override specific variable
```

### Kubernetes

```yaml
# Create from file
kubectl create configmap pdf-qa-config --from-env-file=.env.production

# Use in pod
spec:
  containers:
  - name: pdf-qa
    envFrom:
    - configMapRef:
        name: pdf-qa-config
```

### Shell

```bash
# Load all variables
export $(cat .env.production | xargs)

# Load specific file
set -a
source .env.production
set +a
```

## Secrets Management

### Development (Local)

```bash
# .env.local (gitignored)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Docker Secrets

```bash
# Create secrets
echo "sk-..." | docker secret create openai_api_key -
echo "sk-ant-..." | docker secret create anthropic_api_key -

# Use in compose
services:
  pdf-qa:
    secrets:
      - openai_api_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key

secrets:
  openai_api_key:
    external: true
```

### Kubernetes Secrets

```bash
# Create secret
kubectl create secret generic pdf-qa-secrets \
  --from-literal=openai-api-key="sk-..." \
  --from-literal=anthropic-api-key="sk-ant-..."

# Use in deployment
spec:
  containers:
  - name: pdf-qa
    env:
    - name: OPENAI_API_KEY
      valueFrom:
        secretKeyRef:
          name: pdf-qa-secrets
          key: openai-api-key
```

### AWS Secrets Manager

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Use in code
secrets = get_secret('pdf-qa/production')
os.environ['OPENAI_API_KEY'] = secrets['openai_api_key']
```

### HashiCorp Vault

```python
import hvac

client = hvac.Client(url='https://vault.example.com')
client.token = os.getenv('VAULT_TOKEN')

# Read secret
secret = client.secrets.kv.v2.read_secret_version(path='pdf-qa/production')
os.environ['OPENAI_API_KEY'] = secret['data']['data']['openai_api_key']
```

## Configuration by Use Case

### High Volume Processing

```bash
# .env.high-volume
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
N_RESULTS=2
BATCH_SIZE=100
MAX_WORKERS=8
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast model
```

### High Quality Analysis

```bash
# .env.high-quality
CHUNK_SIZE=500
CHUNK_OVERLAP=100
N_RESULTS=5
EMBEDDING_MODEL=all-mpnet-base-v2  # Better quality
OPENAI_MODEL=gpt-4
```

### Cost Optimization

```bash
# .env.cost-optimized
USE_OLLAMA=true
OLLAMA_MODEL=llama3.2
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
N_RESULTS=3
```

### Privacy-First

```bash
# .env.privacy
# Use only local models
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# No external API keys needed
# All processing happens locally
```

## Environment Validation

```python
# validate_env.py
import os
import sys

REQUIRED_VARS = [
    'EMBEDDING_MODEL',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
]

OPTIONAL_VARS = {
    'OPENAI_API_KEY': 'OpenAI',
    'ANTHROPIC_API_KEY': 'Anthropic',
    'HF_TOKEN': 'HuggingFace',
}

def validate_environment():
    """Validate environment configuration."""
    errors = []

    # Check required variables
    for var in REQUIRED_VARS:
        if not os.getenv(var):
            errors.append(f"Missing required variable: {var}")

    # Check optional variables (need at least one LLM)
    llm_found = False
    for var, provider in OPTIONAL_VARS.items():
        if os.getenv(var):
            llm_found = True
            print(f"✓ {provider} configured")

    if not llm_found:
        # Check if Ollama is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                llm_found = True
                print("✓ Ollama configured")
        except:
            pass

    if not llm_found:
        errors.append("No LLM configured (need OpenAI, Anthropic, or Ollama)")

    # Validate numeric values
    try:
        chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        if chunk_size < 100 or chunk_size > 2000:
            errors.append("CHUNK_SIZE should be between 100 and 2000")
    except ValueError:
        errors.append("CHUNK_SIZE must be a number")

    if errors:
        print("Environment validation failed:")
        for error in errors:
            print(f"  ✗ {error}")
        sys.exit(1)
    else:
        print("✓ Environment validation passed")

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    validate_environment()
```

**Usage:**

```bash
python validate_env.py
```

## Multi-Environment Management

### Using direnv

```bash
# .envrc
source_env .env.development

# Auto-load on directory change
direnv allow
```

### Using conda

```bash
# Create environment
conda create -n pdf-qa-dev python=3.11
conda activate pdf-qa-dev

# Save environment variables
conda env config vars set OPENAI_API_KEY=sk-...
conda env config vars set CHUNK_SIZE=500
```

### Using pyenv

```bash
# Install Python version
pyenv install 3.11.0

# Set local version
pyenv local 3.11.0

# Create virtualenv
pyenv virtualenv 3.11.0 pdf-qa-prod
pyenv activate pdf-qa-prod
```

## Best Practices

### 1. Never Commit Secrets

```bash
# .gitignore
.env
.env.local
.env.*.local
*.key
secrets/
```

### 2. Use .env.example

```bash
# .env.example (commit this)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 3. Validate on Startup

```python
# main.py
from validate_env import validate_environment

if __name__ == '__main__':
    validate_environment()
    # Continue with application
```

### 4. Document Required Variables

```markdown
# Required Environment Variables

- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` or Ollama running
- `CHUNK_SIZE` (default: 500)
- `CHUNK_OVERLAP` (default: 50)
```

### 5. Use Defaults

```python
# Use sensible defaults
chunk_size = int(os.getenv('CHUNK_SIZE', 500))
chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
log_level = os.getenv('LOG_LEVEL', 'INFO')
```

## Troubleshooting

### Variables Not Loading

```python
# Debug environment loading
from dotenv import load_dotenv
import os

print("Loading .env...")
load_dotenv(verbose=True)

print("\nEnvironment variables:")
for key in ['OPENAI_API_KEY', 'CHUNK_SIZE', 'LOG_LEVEL']:
    value = os.getenv(key)
    print(f"{key}: {'✓ Set' if value else '✗ Not set'}")
```

### Wrong Environment

```bash
# Check which environment is active
echo "Environment: $ENV_NAME"
echo "API Key: ${OPENAI_API_KEY:0:8}..."
echo "Log Level: $LOG_LEVEL"
```

### Docker Environment Issues

```bash
# Check environment in container
docker exec pdf-qa-system env | grep -E "OPENAI|CHUNK|LOG"
```

## Next Steps

- [Production Deployment](production.md)
- [Configuration Guide](../user-guide/configuration.md)
- [Docker Guide](../user-guide/docker.md)
