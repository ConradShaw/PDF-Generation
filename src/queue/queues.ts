import { Queue } from 'bullmq';
import {
  redisConnectionOptions,
  QUEUE_INDIVIDUAL,
  QUEUE_TEAM,
  type IndividualJobData,
  type TeamJobData,
} from './connection.js';

let individualQueue: Queue<IndividualJobData> | null = null;
let teamQueue: Queue<TeamJobData> | null = null;

export function getIndividualQueue(): Queue<IndividualJobData> {
  if (!individualQueue) {
    individualQueue = new Queue<IndividualJobData>(QUEUE_INDIVIDUAL, {
      connection: redisConnectionOptions(),
      defaultJobOptions: {
        attempts: 3,
        backoff: { type: 'exponential', delay: 5000 },
        removeOnComplete: 100,
        removeOnFail: 200,
      },
    });
  }
  return individualQueue;
}

export function getTeamQueue(): Queue<TeamJobData> {
  if (!teamQueue) {
    teamQueue = new Queue<TeamJobData>(QUEUE_TEAM, {
      connection: redisConnectionOptions(),
      defaultJobOptions: {
        attempts: 3,
        backoff: { type: 'exponential', delay: 5000 },
        removeOnComplete: 100,
        removeOnFail: 200,
      },
    });
  }
  return teamQueue;
}

export async function enqueueIndividualReport(assessmentId: string) {
  const queue = getIndividualQueue();
  const jobId = `individual:${assessmentId}`;
  const existing = await queue.getJob(jobId);
  if (existing) {
    const state = await existing.getState();
    if (state === 'completed' || state === 'failed') {
      await existing.remove();
    } else {
      return existing;
    }
  }
  return queue.add(
    'individual-report',
    { assessmentId },
    { jobId },
  );
}

export async function enqueueTeamReport(companyId: string, force = false) {
  const queue = getTeamQueue();
  const jobId = force
    ? `team:${companyId}:force:${Date.now()}`
    : `team:${companyId}`;

  if (!force) {
    const existing = await queue.getJob(jobId);
    if (existing) {
      const state = await existing.getState();
      if (state === 'completed' || state === 'failed') {
        await existing.remove();
      } else {
        return existing;
      }
    }
  }

  return queue.add(
    'team-report',
    { companyId, force },
    { jobId },
  );
}
