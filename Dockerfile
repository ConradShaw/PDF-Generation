# Shaw Strengths Matrix™ PDF Generator - Cloud Run Service
# Dockerfile for Google Cloud Run deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system dependencies (for reportlab)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY logo.png .

# Expose port
EXPOSE 8080

# Run with uvicorn for production
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1

