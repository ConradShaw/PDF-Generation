# Shaw Strengths Matrix™ PDF Generator Service
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Required at runtime (set via Railway env vars):
#   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, RESEND_API_KEY

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY pdf_helpers.py .
COPY logo.png .

EXPOSE 8080

# Single worker keeps idle CPU near zero on Railway.
# The asyncio semaphore (MAX_CONCURRENT_PDFS) limits concurrency within the worker.
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1

