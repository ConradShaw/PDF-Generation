import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import type { Context } from 'hono';
import { env } from './env.js';
import {
  enqueueCareerMatch,
  enqueueRoleFitReport,
  enqueueIndividualReport,
  enqueueTeamReport,
} from './queue/queues.js';
import { readJsonBody } from './http.js';
import { startWorkers } from './worker.js';

const app = new Hono();
app.use('*', cors());

app.get('/health', (c) =>
  c.json({
    status: 'healthy',
    service: 'ssm-pdf-generator',
    version: '2.0.0',
    queue: 'bullmq',
  }),
);

async function enqueueIndividual(c: Context) {
  const body = await readJsonBody(c);
  const assessmentId = String(body.assessment_id ?? body.assessmentId ?? '');
  if (!assessmentId) {
    return c.json({ success: false, message: 'assessment_id is required' }, 400);
  }
  try {
    const job = await enqueueIndividualReport(assessmentId);
    return c.json(
      {
        success: true,
        jobId: job.id,
        message: 'Individual report job enqueued',
      },
      202,
    );
  } catch (err) {
    console.error('[api] enqueue individual failed:', err);
    return c.json(
      { success: false, message: err instanceof Error ? err.message : 'Enqueue failed' },
      500,
    );
  }
}

async function enqueueTeam(c: Context) {
  const body = await readJsonBody(c);
  const companyId = String(body.company_id ?? body.companyId ?? '');
  const force = Boolean(body.force);
  if (!companyId) {
    return c.json({ success: false, message: 'company_id is required' }, 400);
  }
  try {
    const job = await enqueueTeamReport(companyId, force);
    return c.json(
      {
        success: true,
        jobId: job.id,
        message: 'Team report job enqueued',
      },
      202,
    );
  } catch (err) {
    console.error('[api] enqueue team failed:', err);
    return c.json(
      { success: false, message: err instanceof Error ? err.message : 'Enqueue failed' },
      500,
    );
  }
}

async function enqueueCareer(c: Context) {
  const body = await readJsonBody(c);
  const purchaseId = String(body.purchase_id ?? body.purchaseId ?? '');
  if (!purchaseId) {
    return c.json({ success: false, message: 'purchase_id is required' }, 400);
  }
  try {
    const job = await enqueueCareerMatch(purchaseId);
    return c.json(
      { success: true, jobId: job.id, message: 'Career match job enqueued' },
      202,
    );
  } catch (err) {
    console.error('[api] enqueue career failed:', err);
    return c.json(
      { success: false, message: err instanceof Error ? err.message : 'Enqueue failed' },
      500,
    );
  }
}

async function enqueueRoleFit(c: Context) {
  const body = await readJsonBody(c);
  const companyId = String(body.company_id ?? body.companyId ?? '');
  if (!companyId) {
    return c.json({ success: false, message: 'company_id is required' }, 400);
  }
  try {
    const job = await enqueueRoleFitReport(companyId);
    return c.json({ success: true, jobId: job.id, message: 'Role Fit report job enqueued' }, 202);
  } catch (err) {
    console.error('[api] enqueue role fit failed:', err);
    return c.json(
      { success: false, message: err instanceof Error ? err.message : 'Enqueue failed' },
      500,
    );
  }
}

app.post('/jobs/role-fit', enqueueRoleFit);
app.post('/jobs/career-match', enqueueCareer);
app.post('/jobs/individual', enqueueIndividual);
app.post('/jobs/team', enqueueTeam);
app.post('/process-assessment', enqueueIndividual);
app.post('/process-team-report', enqueueTeam);

const runWorker =
  process.env.RUN_WORKER !== '0' && process.env.RUN_WORKER !== 'false';

if (runWorker) {
  startWorkers();
  console.log('[boot] BullMQ workers started in-process');
}

serve({ fetch: app.fetch, port: env.port }, (info) => {
  console.log(`[boot] PDF API listening on :${info.port}`);
});
