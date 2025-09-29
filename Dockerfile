# Multi-stage build for HiLabs Contract Classification System
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Pre-download sentence transformer model (comment out for faster builds if not using semantic mode)
# RUN python -c "from sentence_transformers import SentenceTransformer; \
#     SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p output cache logs temp \
    HiLabsAIQuest_ContractsAI/Contracts/TN \
    HiLabsAIQuest_ContractsAI/Contracts/WA \
    HiLabsAIQuest_ContractsAI/Standard_Templates \
    && chmod -R 755 /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create non-root user for security
RUN useradd -m -u 1000 hilabs && chown -R hilabs:hilabs /app
USER hilabs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run Streamlit dashboard
CMD ["streamlit", "run", "dashboard.py", \
    "--server.maxUploadSize", "200", \
    "--server.enableCORS", "false", \
    "--server.enableXsrfProtection", "true"]