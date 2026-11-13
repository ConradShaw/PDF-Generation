# Shaw Strengths Matrix® PDF Generator (JS / BullMQ)

Node + TypeScript service that scores assessments, renders HTML→PDF with Playwright,
uploads to Supabase Storage, and emails via Resend. Jobs are processed by **BullMQ**
workers backed by **Redis**.

## Endpoints

| Method | Path | Behaviour |
|--------|------|-----------|
| GET | `/health` | Health check |
| POST | `/jobs/individual` | Enqueue individual report `{ "assessment_id": "..." }` → 202 |
| POST | `/jobs/team` | Enqueue team report `{ "company_id": "...", "force": false }` → 202 |
| POST | `/process-assessment` | Alias of `/jobs/individual` (SvelteKit compat) |
| POST | `/process-team-report` | Alias of `/jobs/team` (SvelteKit compat) |

## Env

```
PORT=8080
REDIS_URL=redis://127.0.0.1:6379
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
RESEND_API_KEY=...
DISABLE_REPORT_EMAILS=false
RUN_WORKER=1
```

## Local

```bash
npm install
npx playwright install chromium
# Redis must be running
npm start          # API + in-process workers
# or
RUN_WORKER=0 npm start
npm run worker     # workers only
npm test
```

## Deploy

Docker image includes Playwright Chromium. Provide `REDIS_URL` (Railway Redis / Upstash)
plus Supabase and Resend secrets. Website `PDF_API_URL` should point at this service.
