# Usage Guide

## Quick Start

### Basic Vulnerability Scanning

Analyze a single Python file:
```bash
python -m catastrphe.predict --file vulnerable_code.py
```

Scan an entire directory:
```bash
python -m catastrphe.predict --dir ./src --recursive
```

### Output Format

```
Analyzing: vulnerable_code.py
[VULNERABLE] Score: 0.732
Potential security issues detected!

Summary:
- Files analyzed: 1
- Vulnerabilities found: 1
- Average score: 0.732
```

## Command Line Interface

### Training the Model

```bash
python -m catastrphe.train [OPTIONS]

Options:
  --data-path PATH      Path to training dataset (default: data/dataset.json)
  --epochs INT          Number of training epochs (default: 10)
  --batch-size INT      Batch size for training (default: 32)
  --learning-rate FLOAT Learning rate (default: 0.001)
  --model-path PATH     Path to save trained model (default: hf_models/)
  --push-to-hub         Push model to Hugging Face Hub
```

### Prediction and Analysis

```bash
python -m catastrphe.predict [OPTIONS]

Options:
  --file PATH           Analyze a single file
  --dir PATH            Analyze all files in directory
  --recursive           Include subdirectories
  --threshold FLOAT     Vulnerability threshold (default: 0.5)
  --output FORMAT       Output format: json|csv|text (default: text)
  --verbose             Show detailed analysis
```

### Model Evaluation

```bash
python -m catastrphe.evaluate [OPTIONS]

Options:
  --test-data PATH      Path to test dataset
  --model-path PATH     Path to model weights
  --metrics             Show detailed metrics
  --save-report PATH    Save evaluation report
```

## Python API Usage

### Basic Integration

```python
from catastrphe.predict import VulnerabilityDetector

# Initialize detector
detector = VulnerabilityDetector()

# Analyze code string
code = '''
def unsafe_query(user_input):
    query = f"SELECT * FROM users WHERE id = {user_input}"
    return execute_query(query)
'''

result = detector.analyze_code(code)
print(f"Vulnerability Score: {result['score']}")
print(f"Is Vulnerable: {result['is_vulnerable']}")
```

### Batch Processing

```python
from catastrphe.predict import VulnerabilityDetector
import glob

detector = VulnerabilityDetector()

# Analyze multiple files
files = glob.glob("src/**/*.py", recursive=True)
results = []

for file_path in files:
    with open(file_path, 'r') as f:
        code = f.read()
    
    result = detector.analyze_code(code)
    results.append({
        'file': file_path,
        'score': result['score'],
        'vulnerable': result['is_vulnerable']
    })

# Generate report
vulnerable_files = [r for r in results if r['vulnerable']]
print(f"Found {len(vulnerable_files)} vulnerable files")
```

### Custom Threshold Configuration

```python
from catastrphe.predict import VulnerabilityDetector

# Strict security (lower threshold)
strict_detector = VulnerabilityDetector(threshold=0.3)

# Relaxed detection (higher threshold)
relaxed_detector = VulnerabilityDetector(threshold=0.7)
```

## GitHub Bot Usage

### Setting Up the Bot

1. **Create GitHub App**:
   - Go to GitHub Settings → Developer settings → GitHub Apps
   - Create new GitHub App with these permissions:
     - Pull requests: Read & Write
     - Contents: Read
     - Metadata: Read

2. **Configure Webhook**:
   ```
   Webhook URL: https://your-domain.com/webhook
   Webhook secret: your_secret_key
   Events: Pull request
   ```

3. **Deploy the Bot**:
   ```bash
   # Using Docker Compose
   docker-compose up -d github-bot
   
   # Or manually
   cd github_bot
   gunicorn app:app --bind 0.0.0.0:8080
   ```

### Bot Configuration

Environment variables for the bot:
```bash
# Required
GITHUB_TOKEN=your_github_app_token
GITHUB_WEBHOOK_SECRET=your_webhook_secret

# Optional
VULNERABILITY_THRESHOLD=0.5
COMMENT_ON_SAFE_CODE=false
MAX_FILES_PER_PR=50
```

### Bot Behavior

The bot automatically:
1. Monitors new pull requests
2. Analyzes changed Python files
3. Comments with vulnerability report
4. Updates status checks

Example bot comment:
```markdown
## 🔍 CATastrophe Security Analysis

I've analyzed the Python files in this PR. Here's what I found:

### 📊 Summary
- Files analyzed: 5
- Vulnerabilities found: 2
- Average score: 0.421

### 🚨 Vulnerable Files

1. **src/database.py** (Score: 0.823)
   - High vulnerability score detected
   - Please review for potential SQL injection

2. **src/auth.py** (Score: 0.651)
   - Medium vulnerability score
   - Check for hardcoded credentials

### ✅ Safe Files
- src/utils.py (Score: 0.123)
- src/models.py (Score: 0.234)
- tests/test_auth.py (Score: 0.187)

Please address the security concerns before merging.
```

## Docker Usage

### Running with Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  catastrophe:
    build: .
    volumes:
      - ./data:/app/data
      - ./hf_models:/app/hf_models
    environment:
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
    command: python -m catastrphe.predict --dir /app/data/code

  github-bot:
    build: ./github_bot
    ports:
      - "8080:8080"
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - GITHUB_WEBHOOK_SECRET=${GITHUB_WEBHOOK_SECRET}
```

### Building Custom Images

```dockerfile
# Custom Dockerfile for production
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY hf_models/ ./hf_models/

ENTRYPOINT ["python", "-m", "catastrphe.predict"]
```

## Integration Examples

### CI/CD Pipeline (GitHub Actions)

```yaml
name: Security Analysis

on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install CATastrophe
        run: |
          pip install -r requirements.txt
      
      - name: Run Security Analysis
        run: |
          python -m catastrphe.predict --dir src/ --threshold 0.5
        continue-on-error: true
      
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: security-report
          path: vulnerability_report.json
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: catastrophe
        name: CATastrophe Security Check
        entry: python -m catastrphe.predict
        language: system
        files: \.py$
        args: ['--file']
```

## Best Practices

1. **Regular Model Updates**: Retrain quarterly with new vulnerability data
2. **Threshold Tuning**: Adjust based on false positive tolerance
3. **Incremental Scanning**: Analyze only changed files in CI/CD
4. **Result Caching**: Cache results for unchanged files
5. **Human Review**: Always review high-score detections manually