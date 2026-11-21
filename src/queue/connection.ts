import { env } from '../env.js';

/** Fresh connection options for each Queue/Worker (BullMQ requires non-shared blocking clients). */
export function redisConnectionOptions() {
  return {
    url: env.redisUrl,
    maxRetriesPerRequest: null as null,
  };
}

export const QUEUE_INDIVIDUAL = 'pdf-individual';
export const QUEUE_TEAM = 'pdf-team';
export const QUEUE_CAREER = 'pdf-career';
export const QUEUE_ROLEFIT = 'pdf-rolefit';

export type RoleFitJobData = {
  companyId: string;
};

export type CareerJobData = {
  purchaseId: string;
};

export type IndividualJobData = {
  assessmentId: string;
};

export type TeamJobData = {
  companyId: string;
  force: boolean;
};
