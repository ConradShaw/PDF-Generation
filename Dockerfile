# Shaw Strengths Matrix™ PDF Generator Service
# Dockerfile for Railway / Cloud Run deployment

# Use Python base image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Required at runtime – set via Railway / Cloud Run env vars:
#   SUPABASE_URL
#   SUPABASE_SERVICE_ROLE_KEY
#   RESEND_API_KEY
#   APP_URL  (optional, defaults to http://localhost:5173)

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY pdf_helpers.py .
COPY main.py email_service.
COPY logo.png .

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2

