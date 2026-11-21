import { Queue } from 'bullmq';
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
} from './connection.js';

const DEFAULT_JOB_OPTIONS = {
  attempts: 3,
  backoff: { type: 'exponential' as const, delay: 5000 },
  removeOnComplete: 100,
  removeOnFail: 200,
};

let individualQueue: Queue<IndividualJobData> | null = null;
let teamQueue: Queue<TeamJobData> | null = null;

export function getIndividualQueue(): Queue<IndividualJobData> {
  if (!individualQueue) {
    individualQueue = new Queue<IndividualJobData>(QUEUE_INDIVIDUAL, {
      connection: redisConnectionOptions(),
      defaultJobOptions: DEFAULT_JOB_OPTIONS,
    });
  }
  return individualQueue;
}

export function getTeamQueue(): Queue<TeamJobData> {
  if (!teamQueue) {
    teamQueue = new Queue<TeamJobData>(QUEUE_TEAM, {
      connection: redisConnectionOptions(),
      defaultJobOptions: DEFAULT_JOB_OPTIONS,
    });
  }
  return teamQueue;
}

/**
 * BullMQ (5.x) rejects custom job ids containing ':' ("Custom Id cannot contain :"), so ids use '-'.
 * Add a job unless an identical one (same jobId) is still waiting or running, in which case that
 * job is returned. A finished job (completed or failed) is removed first so the report can re-run.
 */
async function addUnique<T extends object>(
  queue: Queue<T>,
  name: string,
  jobId: string,
  data: T,
) {
  const existing = await queue.getJob(jobId);
  if (existing) {
    const state = await existing.getState();
    if (state === 'completed' || state === 'failed') {
      await existing.remove();
    } else {
      return existing;
    }
  }
  return queue.add(name as never, data as never, { jobId });
}

export function enqueueIndividualReport(assessmentId: string) {
  return addUnique(getIndividualQueue(), 'individual-report', `individual-${assessmentId}`, { assessmentId });
}

let careerQueue: Queue<CareerJobData> | null = null;

export function getCareerQueue(): Queue<CareerJobData> {
  if (!careerQueue) {
    careerQueue = new Queue<CareerJobData>(QUEUE_CAREER, {
      connection: redisConnectionOptions(),
      defaultJobOptions: DEFAULT_JOB_OPTIONS,
    });
  }
  return careerQueue;
}

export function enqueueCareerMatch(purchaseId: string) {
  return addUnique(getCareerQueue(), 'career-match-report', `career-${purchaseId}`, { purchaseId });
}

let roleFitQueue: Queue<RoleFitJobData> | null = null;

export function getRoleFitQueue(): Queue<RoleFitJobData> {
  if (!roleFitQueue) {
    roleFitQueue = new Queue<RoleFitJobData>(QUEUE_ROLEFIT, {
      connection: redisConnectionOptions(),
      defaultJobOptions: DEFAULT_JOB_OPTIONS,
    });
  }
  return roleFitQueue;
}

export function enqueueRoleFitReport(companyId: string) {
  return addUnique(getRoleFitQueue(), 'role-fit-report', `rolefit-${companyId}`, { companyId });
}

/** The report a company gets once all its members have finished: Role Fit or the team report. */
export function enqueueCompanyReport(company: { id: string; report_kind?: string | null }) {
  return company.report_kind === 'role_fit'
    ? enqueueRoleFitReport(company.id)
    : enqueueTeamReport(company.id, true);
}

export async function enqueueTeamReport(companyId: string, force = false) {
  const queue = getTeamQueue();
  const jobId = force
    ? `team-${companyId}-force-${Date.now()}`
    : `team-${companyId}`;

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
