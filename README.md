# Shaw Strengths Matrix™ PDF Generator - Cloud Run Service

A stateless HTTP service that generates PDF reports from Excel assessment data. Designed for deployment on Google Cloud Run.

## Features

- **FastAPI**: Modern, fast Python web framework with automatic OpenAPI docs
- **Base64 API**: Accept Excel files as base64, return PDF as base64
- **File Upload API**: Direct file upload with PDF download response
- **Health Check**: Endpoint for Cloud Run health monitoring
- **CORS Enabled**: Ready for cross-origin requests from your SvelteKit app
- **Interactive Docs**: Swagger UI available at `/docs`

## API Endpoints

### `GET /health`
Health check endpoint for Cloud Run.

**Response:**
```json
{
  "status": "healthy",
  "service": "ssm-pdf-generator",
  "version": "1.0.0"
}
```

### `POST /generate-pdf-base64`
Generate PDF from base64-encoded Excel file.

**Request Body:**
```json
{
  "excel_base64": "base64-encoded-excel-file",
  "filename": "optional-original-filename.xlsx"
}
```

**Response:**
```json
{
  "success": true,
  "pdf_base64": "base64-encoded-pdf-file",
  "filename": "SSM_FirstName_LastName_2025-01-01_v1.pdf"
}
```

### `POST /generate-pdf`
Generate PDF from uploaded Excel file. Returns PDF file directly.

**Request:** Multipart form with `file` field containing Excel file.

**Response:** PDF file download.

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Navigate to the project directory
cd pdf-generator-cloudrun

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
python main.py
```

The service will start on `http://localhost:8080`.

### Interactive API Documentation

FastAPI provides automatic interactive docs:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Test the API

```bash
# Health check
curl http://localhost:8080/health

# Generate PDF (with base64)
curl -X POST http://localhost:8080/generate-pdf-base64 \
  -H "Content-Type: application/json" \
  -d '{"excel_base64": "YOUR_BASE64_EXCEL_HERE", "filename": "test.xlsx"}'
```

## Deployment to Google Cloud Run

### Prerequisites
1. Google Cloud SDK installed (`gcloud` CLI)
2. Google Cloud project with billing enabled
3. Cloud Run API enabled

### Enable Required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### Deploy to Cloud Run

#### Option 1: Deploy from Source (Recommended)

```bash
# Navigate to the project directory
cd pdf-generator-cloudrun

# Deploy to Cloud Run (builds and deploys automatically)
gcloud run deploy ssm-pdf-generator \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --timeout 60s \
  --max-instances 10
```

#### Option 2: Build and Deploy Separately

```bash
# Build the container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ssm-pdf-generator

# Deploy to Cloud Run
gcloud run deploy ssm-pdf-generator \
  --image gcr.io/YOUR_PROJECT_ID/ssm-pdf-generator \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --timeout 60s
```

### Get Service URL

After deployment, Cloud Run will output the service URL like:
```
Service URL: https://ssm-pdf-generator-xxxxx-uc.a.run.app
```

### Verify Deployment

```bash
curl https://ssm-pdf-generator-xxxxx-uc.a.run.app/health
```

## Configure SvelteKit Backend

After deploying to Cloud Run, update your SvelteKit `.env` file:

```env
# Cloud Run PDF Generator Service URL
CLOUD_RUN_API_URL=https://ssm-pdf-generator-xxxxx-uc.a.run.app

# Optional: Bearer token for authenticated services
# CLOUD_RUN_API_TOKEN=your-token-here
```

### Environment Variable Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `CLOUD_RUN_API_URL` | Cloud Run service URL | Yes |
| `CLOUD_RUN_API_TOKEN` | Bearer token for auth | No (if service is public) |

## Security Options

### Public Access (Current Setup)
The `--allow-unauthenticated` flag makes the service publicly accessible. This is simpler but less secure.

### Authenticated Access
For production, consider using IAM authentication:

```bash
# Deploy with authentication required
gcloud run deploy ssm-pdf-generator \
  --source . \
  --platform managed \
  --region us-central1 \
  --no-allow-unauthenticated \
  --memory 512Mi
```

Then use a service account token in your SvelteKit backend to authenticate requests.

## Architecture

```
┌─────────────────────┐      ┌──────────────────────────┐
│   SvelteKit App     │      │   Cloud Run Service      │
│                     │      │                          │
│  /api/pdf/generate  │─────►│  /generate-pdf-base64    │
│                     │      │                          │
│  1. Convert form    │      │  1. Decode Excel base64  │
│     to Excel base64 │      │  2. Parse Instructions   │
│  2. Call Cloud Run  │      │  3. Parse Survey         │
│  3. Return PDF      │      │  4. Calculate rankings   │
│                     │◄─────│  5. Generate PDF         │
│                     │      │  6. Return PDF base64    │
└─────────────────────┘      └──────────────────────────┘
```

## Monitoring & Logs

View logs in Google Cloud Console:

```bash
# View logs
gcloud run logs read ssm-pdf-generator --limit 50

# Stream logs
gcloud run logs tail ssm-pdf-generator
```

## Cost Estimation

Cloud Run pricing (as of 2024):
- First 2 million requests/month: Free
- CPU: $0.00002400/vCPU-second
- Memory: $0.00000250/GiB-second

For typical usage (< 1000 PDFs/month), the service should stay within the free tier.

## Troubleshooting

### "Could not find logo.png"
Ensure `logo.png` is in the same directory as `main.py` and is included in the Docker build.

### "Memory limit exceeded"
Increase the memory allocation:
```bash
gcloud run services update ssm-pdf-generator --memory 1Gi
```

### "Request timeout"
Increase the timeout:
```bash
gcloud run services update ssm-pdf-generator --timeout 120s
```

### CORS Issues
The service has CORS enabled by default. If you need to restrict origins, modify the Flask-CORS configuration in `main.py`.

## License

Copyright 2025 ShawSight Pty Ltd. All rights reserved.

