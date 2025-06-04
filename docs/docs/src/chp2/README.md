# Requirements and Installation

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 2 GB free space
- **GPU**: Optional (CPU inference supported)

#### Recommended Requirements
- **CPU**: 8+ cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 5 GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM (CUDA 11.0+)

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 20.04+, Debian 10+, RHEL 8+
- **macOS**: 10.15+ (Catalina or later)
- **Windows**: Windows 10/11 with WSL2

#### Python Environment
- **Python**: 3.8, 3.9, or 3.10
- **pip**: 20.0+
- **virtualenv**: Recommended for isolation

## Installation Methods

### 1. Local Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/catastrophe.git
cd catastrophe
```

#### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Verify Installation
```bash
python -c "import catastrphe; print('Installation successful!')"
```

### 2. Docker Installation

#### Using Docker Compose
```bash
# Build and run all services
docker-compose up --build

# Run specific service
docker-compose up github-bot
```

#### Using Dockerfile
```bash
# Build image
docker build -t catastrophe:latest .

# Run container
docker run -it catastrophe:latest
```

### 3. Development Installation

For contributors and developers:

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/catastrophe.git
cd catastrophe

# Install in editable mode
pip install -e .

# Install development dependencies
pip install pylint pytest black isort
```

## Dependencies Overview

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Neural network framework |
| scikit-learn | ≥1.3.0 | Feature extraction and preprocessing |
| numpy | ≥1.24.0 | Numerical computations |
| pandas | ≥2.0.0 | Data manipulation |
| huggingface-hub | ≥0.20.0 | Model versioning and sharing |

### Additional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python-dotenv | ≥1.0.0 | Environment variable management |
| tqdm | ≥4.65.0 | Progress bars |
| matplotlib | ≥3.7.0 | Visualization |
| seaborn | ≥0.12.0 | Statistical plots |
| PyGithub | ≥2.1.0 | GitHub API integration |

### GitHub Bot Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| flask | ≥3.0.0 | Web framework |
| gunicorn | ≥21.2.0 | WSGI server |
| requests | ≥2.31.0 | HTTP client |

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_token_here
HUGGINGFACE_REPO_ID=your-username/catastrophe-model

# GitHub Bot Configuration
GITHUB_TOKEN=your_github_token
GITHUB_WEBHOOK_SECRET=your_webhook_secret

# Model Configuration
VULNERABILITY_THRESHOLD=0.5
MODEL_PATH=./hf_models/
```

### Optional Configuration

```bash
# Performance Tuning
BATCH_SIZE=32
MAX_FEATURES=2000
LEARNING_RATE=0.001

# Logging
LOG_LEVEL=INFO
LOG_FILE=catastrophe.log
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Solution: Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Import Errors
```bash
# Ensure all dependencies are installed
pip install --upgrade -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 3. Memory Issues
```bash
# Reduce batch size
export BATCH_SIZE=16

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
```

#### 4. Permission Errors
```bash
# Fix file permissions
chmod -R 755 ./data
chmod -R 755 ./hf_models
```

## Platform-Specific Notes

### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies: `brew install python@3.10`

### Windows
- Use WSL2 for best compatibility
- Install Visual C++ Build Tools for native extensions

### Linux
- Install Python development headers: `sudo apt-get install python3-dev`
- Ensure libgomp1 is installed for PyTorch: `sudo apt-get install libgomp1`