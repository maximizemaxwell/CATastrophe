# CATastrophe Deployment Guide

This guide covers how to train the vulnerability detection model and deploy the GitHub bot.

## Prerequisites

- Docker and Docker Compose installed
- GitHub account and personal access token
- Hugging Face account and API token
- Python 3.10+ (if running locally)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/your-repo/catastrophe.git
cd catastrophe
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file with your credentials:

```env
# Hugging Face Configuration
HF_TOKEN=your_hugging_face_token_here
HF_REPO_ID=your-username/catastrophe-vulnerability-detector

# GitHub Bot Configuration
GITHUB_TOKEN=your_github_token_here
GITHUB_WEBHOOK_SECRET=your_webhook_secret_here

# Bot Configuration
BOT_PORT=8080
VULNERABILITY_THRESHOLD=0.7
```

### 3. Prepare Dataset

Place your vulnerability dataset in `data/dataset.json`. The format should be:

```json
{"message": "Fix SQL injection vulnerability", "func": "def query_db(user_input): cursor.execute(f'SELECT * FROM users WHERE id = {user_input}')"}
{"message": "Update authentication", "func": "def login(username, password): # vulnerable code here"}
```

## Training the Model

### Using Docker (Recommended)

```bash
# Build and train the model
docker-compose --profile train up --build

# The model will be automatically uploaded to Hugging Face
```

### Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m catastrphe.train
```

## Deploying the GitHub Bot

### Using Docker Compose

```bash
# Start the GitHub bot
docker-compose up -d github-bot

# Check logs
docker-compose logs -f github-bot

# Stop the bot
docker-compose down
```

### Manual Deployment

```bash
# Build the bot image
docker build -f github_bot/Dockerfile -t catastrophe-bot .

# Run the bot
docker run -d \
  --name catastrophe-bot \
  -p 8080:8080 \
  --env-file .env \
  catastrophe-bot
```

## GitHub Webhook Configuration

1. Go to your GitHub repository settings
2. Navigate to Webhooks → Add webhook
3. Configure:
   - **Payload URL**: `http://your-server:8080/webhook`
   - **Content type**: `application/json`
   - **Secret**: Your `GITHUB_WEBHOOK_SECRET` from `.env`
   - **Events**: Select "Pull requests"

## Bot Features

The bot will:
- Analyze all Python code changes in pull requests
- Calculate vulnerability scores for each commit
- Post a detailed comment with:
  - List of vulnerable commits
  - Vulnerability scores
  - Summary table of all analyzed files

## Configuration Options

### Vulnerability Threshold

Adjust `VULNERABILITY_THRESHOLD` in `.env` (default: 0.7):
- Higher values = fewer false positives
- Lower values = more sensitive detection

### Model Architecture

The enhanced model uses:
- 4-layer encoder: Input → 1024 → 512 → 256 → 128
- Batch normalization and dropout for regularization
- Early stopping to prevent overfitting
- Learning rate scheduling

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Logs

```bash
# Docker Compose
docker-compose logs -f github-bot

# Docker
docker logs -f catastrophe-bot
```

## Troubleshooting

### Model Not Loading

1. Check Hugging Face credentials in `.env`
2. Verify model exists at the specified `HF_REPO_ID`
3. Check bot logs for specific error messages

### Webhook Not Triggering

1. Verify webhook secret matches `.env` configuration
2. Check GitHub webhook delivery logs
3. Ensure bot is accessible from the internet

### High False Positive Rate

1. Adjust `VULNERABILITY_THRESHOLD` higher
2. Retrain model with more diverse dataset
3. Consider adding more training epochs

## Security Considerations

- Never commit `.env` file
- Use strong webhook secrets
- Restrict GitHub token permissions to minimum required
- Run bot in isolated environment
- Regularly update dependencies

## Performance Optimization

- Use GPU for training if available
- Adjust `BATCH_SIZE` based on available memory
- Enable model caching in production
- Consider using model quantization for faster inference
