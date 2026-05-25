# Base Python image
FROM python:3.11-slim

# Don't write .pyc files & keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    PYTHONPATH=/app/src

# Set work directory
WORKDIR /app

# Install system deps (for lxml/html parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for Docker layer caching
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY src/ /app/src/
COPY pyproject.toml .

# Install the sdk package itself (for metaai-sdk imports)
RUN pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /data/uploads /data/generations

# Expose port
EXPOSE 8000

# Volume for persistent data
VOLUME ["/data"]

# Launch Uvicorn
CMD ["uvicorn", "metaai_api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
