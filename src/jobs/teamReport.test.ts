import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ladderBundle } from '../../tests/fixtures/bundle.js';
import { TRAITS } from '../scoring/traits.js';

type Row = Record<string, unknown>;
let company: Row;
let members: Row[];
let updates: Row[];
const order: string[] = [];

/** Just enough of the Supabase query builder for what processTeamReport reads and writes. */
const fakeSupabase = {
  from: (table: string) => {
    const q: Record<string, unknown> = {};
    q.select = () => q;
    q.eq = () => q;
    q.update = (payload: Row) => {
      updates.push(payload);
      order.push('mark-generated');
      return q;
    };
    q.single = async () => ({ data: table === 'companies' ? company : null, error: null });
    q.then = (resolve: (v: unknown) => unknown) =>
      resolve({ data: table === 'team_assessment_members' ? members : null, error: null });
    return q;
  },
};

vi.mock('../supabase.js', () => ({
  getSupabase: () => fakeSupabase,
  fetchAssessmentData: vi.fn(),
  uploadPdfToStorage: vi.fn(async () => {
    order.push('upload');
  }),
}));
vi.mock('../pdf/render.js', () => ({ htmlToPdf: vi.fn(async () => Buffer.from('%PDF-fake')) }));
vi.mock('../email.js', () => ({
  sendTeamReportEmail: vi.fn(async () => {
    order.push('email');
  }),
}));

import { env } from '../env.js';
import * as db from '../supabase.js';
import { htmlToPdf } from '../pdf/render.js';
import { sendTeamReportEmail } from '../email.js';
import { processTeamReport } from './teamReport.js';

const finished = (id: string, orderOfTraits: readonly string[] = TRAITS) => ({
  id,
  bundle: ladderBundle({ id, status: 'completed' }, { name: id, email: `${id}@x.test` }, orderOfTraits),
});

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  order.length = 0;
  updates = [];
  company = { id: 'co1', company_name: 'Acme Pty Ltd', company_code: 'ACME1', contact_email: 'lead@acme.test', team_size: 2, team_report_generated: false };
  const people = [finished('a1'), finished('a2', [...TRAITS].reverse())];
  members = people.map((p) => ({ assessment_id: p.id }));
  vi.mocked(db.fetchAssessmentData).mockImplementation(async (id) => people.find((p) => p.id === id)!.bundle);
  vi.mocked(sendTeamReportEmail).mockImplementation(async () => {
    order.push('email');
  });
});
afterEach(() => {
  env.disableReportEmails = false;
});

describe('processTeamReport', () => {
  it('stores the PDF, emails it to the company contact, and only then marks the company generated', async () => {
    const result = await processTeamReport('co1');

    expect(result.pdfUrl).toMatch(/^team_reports\/co1\/SSM_Team_Acme_Pty_Ltd_\d{4}-\d{2}-\d{2}_v1\.pdf$/);
    expect(sendTeamReportEmail).toHaveBeenCalledWith('lead@acme.test', 'Acme Pty Ltd', 'ACME1', 2, expect.any(Buffer), expect.stringMatching(/\.pdf$/));
    expect(order).toEqual(['upload', 'email', 'mark-generated']);
    expect(updates[0]).toMatchObject({ team_report_generated: true, team_pdf_url: result.pdfUrl, team_pdf_storage_path: result.pdfUrl });
  });

  it('fails the job when the email cannot be sent, so BullMQ retries it instead of the customer silently never getting the report', async () => {
    vi.mocked(sendTeamReportEmail).mockRejectedValueOnce(new Error('Resend is down'));
    await expect(processTeamReport('co1')).rejects.toThrow('Resend is down');
  });

  it('does not mark the company generated when the email failed, so a retry is not mistaken for "already done"', async () => {
    vi.mocked(sendTeamReportEmail).mockRejectedValueOnce(new Error('Resend is down'));
    await expect(processTeamReport('co1')).rejects.toThrow();
    expect(updates).toEqual([]);
  });

  it('a retry after an email failure sends the report: the second attempt must not skip the email', async () => {
    vi.mocked(sendTeamReportEmail).mockRejectedValueOnce(new Error('Resend is down'));
    await expect(processTeamReport('co1')).rejects.toThrow();

    const second = await processTeamReport('co1'); // not forced, company still unmarked
    expect(second.message).toBe('Team report generated successfully');
    expect(sendTeamReportEmail).toHaveBeenCalledTimes(2);
    expect(updates).toHaveLength(1);
  });

  it('does nothing when the report was already generated, unless forced', async () => {
    company.team_report_generated = true;
    company.team_pdf_url = 'team_reports/co1/old.pdf';
    const skipped = await processTeamReport('co1');
    expect(skipped).toEqual({ pdfUrl: 'team_reports/co1/old.pdf', message: 'Team report already generated' });
    expect(htmlToPdf).not.toHaveBeenCalled();
    expect(sendTeamReportEmail).not.toHaveBeenCalled();

    await processTeamReport('co1', true);
    expect(htmlToPdf).toHaveBeenCalledTimes(1);
    expect(sendTeamReportEmail).toHaveBeenCalledTimes(1);
  });

  it('skips the email when emails are disabled, but still stores the report and marks it generated', async () => {
    env.disableReportEmails = true;
    await processTeamReport('co1');
    expect(sendTeamReportEmail).not.toHaveBeenCalled();
    expect(order).toEqual(['upload', 'mark-generated']);
  });

  it('still completes when the company has no contact email', async () => {
    company.contact_email = null;
    await processTeamReport('co1');
    expect(sendTeamReportEmail).not.toHaveBeenCalled();
    expect(order).toEqual(['upload', 'mark-generated']);
  });

  it('fails when nobody has finished, rather than producing an empty report', async () => {
    vi.mocked(db.fetchAssessmentData).mockImplementation(async (id) => ladderBundle({ id, status: 'submitted' }));
    await expect(processTeamReport('co1')).rejects.toThrow('No completed individual results found');
    expect(sendTeamReportEmail).not.toHaveBeenCalled();
  });

  it('fails clearly for a company that does not exist', async () => {
    company = null as never;
    await expect(processTeamReport('missing')).rejects.toThrow('Company not found');
  });
});
