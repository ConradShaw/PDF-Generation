import { Worker } from 'bullmq';
import {
  redisConnectionOptions,
  QUEUE_INDIVIDUAL,
  QUEUE_TEAM,
  QUEUE_CAREER,
  QUEUE_ROLEFIT,
  type IndividualJobData,
  type TeamJobData,
  type CareerJobData,
  type RoleFitJobData,
} from './queue/connection.js';
import { processIndividualReport } from './jobs/individualReport.js';
import { processCareerMatchReport } from './jobs/careerMatchReport.js';
import { runRoleFitJob } from './jobs/roleFitReport.js';
import { isFinalAttempt } from './jobs/shared.js';
import { processTeamReport } from './jobs/teamReport.js';
import { closeBrowser } from './pdf/render.js';
import { getSupabase } from './supabase.js';

export function startWorkers(): { individual: Worker; team: Worker; career: Worker; roleFit: Worker } {
  const individual = new Worker<IndividualJobData>(
    QUEUE_INDIVIDUAL,
    async (job) => processIndividualReport(job.data.assessmentId),
    { connection: redisConnectionOptions(), concurrency: 1 },
  );

  const team = new Worker<TeamJobData>(
    QUEUE_TEAM,
    async (job) => {
      const { companyId, force } = job.data;
      try {
        return await processTeamReport(companyId, force);
      } catch (err) {
        if (force) {
          try {
            const sb = getSupabase();
            await sb
              .from('companies')
              .update({
                team_report_generated: false,
                team_report_generated_at: null,
              })
              .eq('id', companyId)
              .is('team_pdf_url', null);
          } catch {
            /* ignore */
          }
        }
        throw err;
      }
    },
    { connection: redisConnectionOptions(), concurrency: 1 },
  );

  const career = new Worker<CareerJobData>(
    QUEUE_CAREER,
    async (job) =>
      processCareerMatchReport(job.data.purchaseId, {
        isLastAttempt: isFinalAttempt(job.attemptsMade, job.opts.attempts),
      }),
    { connection: redisConnectionOptions(), concurrency: 1 },
  );
  const roleFit = new Worker<RoleFitJobData>(
    QUEUE_ROLEFIT,
    async (job) =>
      runRoleFitJob(job.data.companyId, {
        isLastAttempt: isFinalAttempt(job.attemptsMade, job.opts.attempts),
      }),
    { connection: redisConnectionOptions(), concurrency: 1 },
  );
  roleFit.on('failed', (job, err) => {
    console.error(`[worker:role-fit] job ${job?.id} failed:`, err.message);
  });
  roleFit.on('completed', (job) => {
    console.log(`[worker:role-fit] job ${job.id} completed`);
  });

  career.on('failed', (job, err) => {
    console.error(`[worker:career] job ${job?.id} failed:`, err.message);
  });
  career.on('completed', (job) => {
    console.log(`[worker:career] job ${job.id} completed`);
  });

  individual.on('failed', (job, err) => {
    console.error(`[worker:individual] job ${job?.id} failed:`, err.message);
  });
  team.on('failed', (job, err) => {
    console.error(`[worker:team] job ${job?.id} failed:`, err.message);
  });
  individual.on('completed', (job) => {
    console.log(`[worker:individual] job ${job.id} completed`);
  });
  team.on('completed', (job) => {
    console.log(`[worker:team] job ${job.id} completed`);
  });

  const shutdown = async () => {
    console.log('[worker] shutting down…');
    await Promise.all([individual.close(), team.close(), career.close(), roleFit.close()]);
    await closeBrowser();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  return { individual, team, career, roleFit };
}

const isDirectRun = process.argv[1]?.includes('worker');
if (isDirectRun) {
  startWorkers();
  console.log('[worker] standalone workers running');
}
