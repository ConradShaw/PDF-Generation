import { describe, it, expect, vi, beforeEach } from 'vitest';

const queue = { getJob: vi.fn(), add: vi.fn() };
vi.mock('bullmq', () => ({ Queue: vi.fn(() => queue) }));
import { Queue } from 'bullmq';

import { enqueueCareerMatch, enqueueIndividualReport } from './queues.js';

const job = (state: string) => ({ getState: async () => state, remove: vi.fn() });

beforeEach(() => {
  queue.getJob.mockReset();
  queue.add.mockReset().mockResolvedValue({ id: 'new' });
});

describe.each([
  ['enqueueIndividualReport', enqueueIndividualReport, 'individual-x1', 'individual-report', { assessmentId: 'x1' }],
  ['enqueueCareerMatch', enqueueCareerMatch, 'career-x1', 'career-match-report', { purchaseId: 'x1' }],
])('%s', (_name, enqueue, jobId, jobName, data) => {
  it('adds a job keyed by the record id, so one record has one job at a time', async () => {
    queue.getJob.mockResolvedValue(null);
    await enqueue('x1');
    expect(queue.add).toHaveBeenCalledWith(jobName, data, { jobId });
  });

  it('returns the existing job instead of queueing a duplicate while one is waiting or running', async () => {
    for (const state of ['waiting', 'active', 'delayed']) {
      const existing = job(state);
      queue.getJob.mockResolvedValue(existing);
      expect(await enqueue('x1')).toBe(existing);
      expect(existing.remove).not.toHaveBeenCalled();
    }
    expect(queue.add).not.toHaveBeenCalled();
  });

  it('removes a finished job and queues a fresh one, so a report can be re-run', async () => {
    for (const state of ['completed', 'failed']) {
      const old = job(state);
      queue.getJob.mockResolvedValue(old);
      await enqueue('x1');
      expect(old.remove).toHaveBeenCalled();
    }
    expect(queue.add).toHaveBeenCalledTimes(2);
  });
});

describe('queue configuration', () => {
  it('creates each queue once under its own name, with three attempts and exponential backoff so a transient failure is retried', async () => {
    queue.getJob.mockResolvedValue(null);
    await enqueueIndividualReport('a');
    await enqueueIndividualReport('b');
    await enqueueCareerMatch('a');
    await enqueueCareerMatch('b');

    const created = vi.mocked(Queue).mock.calls;
    const names = created.map(([name]) => name);
    expect(names.filter((n) => n === 'pdf-individual')).toHaveLength(1);
    expect(names.filter((n) => n === 'pdf-career')).toHaveLength(1);
    for (const [, options] of created) {
      expect(options?.defaultJobOptions).toMatchObject({ attempts: 3, backoff: { type: 'exponential' } });
    }
  });
});

describe('job ids', () => {
  it('never contain a colon, which BullMQ rejects, and are never purely numeric (also rejected): every enqueue would fail with a 500 otherwise', async () => {
    const { enqueueTeamReport } = await import('./queues.js');
    queue.getJob.mockResolvedValue(null);
    await enqueueIndividualReport('11111111-2222-3333-4444-555555555555');
    await enqueueCareerMatch('11111111-2222-3333-4444-555555555555');
    await enqueueTeamReport('company-1');
    await enqueueTeamReport('company-1', true);
    const ids = queue.add.mock.calls.map(([, , opts]) => opts.jobId as string);
    expect(ids).toHaveLength(4);
    for (const id of ids) {
      expect(id).not.toContain(':');
      expect(String(parseInt(id, 10))).not.toBe(id);
    }
  });

  it('keep one job per record but let a forced team report run again', async () => {
    const { enqueueTeamReport } = await import('./queues.js');
    queue.getJob.mockResolvedValue(null);
    await enqueueTeamReport('c1');
    await enqueueTeamReport('c1');
    await enqueueTeamReport('c1', true);
    const ids = queue.add.mock.calls.map(([, , opts]) => opts.jobId as string);
    expect(ids[0]).toBe(ids[1]);
    expect(ids[2]).not.toBe(ids[0]);
  });
});

describe('role fit queue and the report a finished company gets', () => {
  it('queues a role-fit job keyed by company, under its own queue name', async () => {
    const { enqueueRoleFitReport } = await import('./queues.js');
    queue.getJob.mockResolvedValue(null);
    await enqueueRoleFitReport('co1');
    expect(queue.add).toHaveBeenCalledWith('role-fit-report', { companyId: 'co1' }, { jobId: 'rolefit-co1' });
    expect(vi.mocked(Queue).mock.calls.map(([n]) => n)).toContain('pdf-rolefit');
  });

  it('sends a Role Fit company to the Role Fit report and any other company to the team report', async () => {
    const { enqueueCompanyReport } = await import('./queues.js');
    queue.getJob.mockResolvedValue(null);
    await enqueueCompanyReport({ id: 'co-rf', report_kind: 'role_fit' });
    await enqueueCompanyReport({ id: 'co-team', report_kind: 'team' });
    await enqueueCompanyReport({ id: 'co-old' }); // companies created before Role Fit existed
    const added = queue.add.mock.calls.map(([name, data]) => [name, data.companyId]);
    expect(added).toEqual([['role-fit-report', 'co-rf'], ['team-report', 'co-team'], ['team-report', 'co-old']]);
  });
});
