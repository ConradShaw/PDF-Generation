# Shaw Strengths Matrix™ PDF Generator - Cloud Run Service
# Dockerfile for Railway deployment

# Use Python base image
FROM 3.11.9-slim-bullseye

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Copy requirements first for better caching
COPY requirements.txt .

# Install system dependencies (for reportlab)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY main.py .
COPY pdf_helpers.py .
COPY logo.png .

# Expose port
EXPOSE 8080

# Run with uvicorn for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

