FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY github_bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY github_bot/ ./github_bot/
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Run the bot
CMD ["python", "github_bot/app.py"]