# Shaw Strengths Matrix® PDF Generator (JS / BullMQ)

Node + TypeScript service that scores assessments, renders HTML→PDF with Playwright,
uploads to Supabase Storage, and emails via Resend. Jobs are processed by **BullMQ**
workers backed by **Redis**.

## Endpoints

| Method | Path | Behaviour |
|--------|------|-----------|
| GET | `/health` | Health check |
| GET | `/admin/queues` | Bull Board UI (all PDF queues). Optional basic auth via `BULL_BOARD_USER` / `BULL_BOARD_PASSWORD` |
| POST | `/jobs/individual` | Enqueue individual report `{ "assessment_id": "..." }` → 202 |
| POST | `/jobs/team` | Enqueue team report `{ "company_id": "...", "force": false }` → 202 |
| POST | `/process-assessment` | Alias of `/jobs/individual` (SvelteKit compat) |
| POST | `/process-team-report` | Alias of `/jobs/team` (SvelteKit compat) |
| POST | `/jobs/career-match` | Enqueue the paid Career Match report `{ "purchase_id": "..." }` → 202 |

## Env

```
PORT=8080
REDIS_URL=redis://127.0.0.1:6379
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
RESEND_API_KEY=...
DISABLE_REPORT_EMAILS=false
RUN_WORKER=1
APP_URL=https://your-site   # used for the 'Unlock Career Matches' link in the free report email
BULL_BOARD_USER=admin       # optional; with PASSWORD, protects /admin/queues
BULL_BOARD_PASSWORD=...
```

## Local

Put the settings above in a `.env` file in this folder: `npm run dev` and `npm start` read it automatically
(real environment variables always win, and production has no `.env`, so nothing changes there).
Redis must be running, and `REDIS_URL` must point at it.

```bash
npm install
npx playwright install chromium
# Redis must be running
npm run dev        # API + in-process workers, restarts on change
npm start          # API + in-process workers
# Open http://localhost:8080/admin/queues for Bull Board
# or
RUN_WORKER=0 npm start
npm run worker     # workers only
npm test
```

## Deploy

Docker image includes Playwright Chromium. Provide `REDIS_URL` (Railway Redis / Upstash)
plus Supabase and Resend secrets. Website `PDF_API_URL` should point at this service.

## Strength Profile (free) and Career Match (paid)

- **Free Strength Profile**: assessments of type `psychometrics` get a short report (12 ranked strengths,
  Work Archetype text for the #1 strength, unlock link) instead of the full individual report.
- **Career Match**: `/jobs/career-match` runs two stages from the stored 66 answers plus the RIASEC result saved after payment
  (`career_match_purchases`): (1) interest shortlist: the 150 occupations whose interests correlate best with the
  customer's, (2) strengths re-rank (geometric weights on the customer's strength ranks), then the **final top 25** with
  stars in equal groups of the list (five results per star level).
- Constants (shortlist size, list size, weight ratio, star bands) live in `src/careers/constants.ts`.
- The strength formerly called Courage is now **Initiative**. Old question-database rows still say "Courage"; `src/scoring/fromAnswers.ts` maps them.

### Owner data (must be supplied before production)

| File | What | Status |
|------|------|--------|
| `data/archetypes.json` | The 12 Work Archetype texts | Drafted by Claude, approved by the owner 2026-11-19 (editable) |
| `data/ssm-mapping.json` | SSM strength → O*NET Work Style | Derived from the existing report copy; confirmed by the owner 2026-11-19 |
| `data/onet-occupations.json` | Generated from the public O*NET database | Regenerate with the script below |

With `NODE_ENV=production` (the Docker image) the free-profile and career jobs **refuse to run while a file is flagged
`"placeholder": true`**, so customers never receive placeholder text. Set the flag back to `true` to block sending again.

Regenerate occupations (download the CSV database from https://www.onetcenter.org/database.html):

```bash
npx tsx scripts/import-onet.ts <dir-with-csvs> 31.0
```

O*NET attribution (CC BY 4.0) is printed in the Career Match report; keep it.
