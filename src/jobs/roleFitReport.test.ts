import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ladderBundle } from '../../tests/fixtures/bundle.js';
import { TRAITS } from '../scoring/traits.js';
import { occupations } from '../careers/data.js';
import { targetProfile } from '../rolefit/engine.js';

vi.mock('../supabase.js', () => ({
  fetchAssessmentData: vi.fn(),
  fetchCompany: vi.fn(),
  fetchMemberAssessmentIds: vi.fn(),
  markCompanyReport: vi.fn(),
  releaseCompanyReportLock: vi.fn(),
  uploadPdfToStorage: vi.fn(),
}));
vi.mock('../pdf/render.js', () => ({ htmlToPdf: vi.fn(async () => Buffer.from('%PDF-fake')) }));
vi.mock('../email.js', () => ({ sendRoleFitReportEmail: vi.fn() }));

import { env } from '../env.js';
import * as db from '../supabase.js';
import { htmlToPdf } from '../pdf/render.js';
import { sendRoleFitReportEmail } from '../email.js';
import { processRoleFitReport, runRoleFitJob } from './roleFitReport.js';

const CEO = '11-1011.00';
const company = (over: Record<string, unknown> = {}) => ({
  id: 'co1', company_name: 'Acme Pty Ltd', team_name: 'Sales Team', contact_email: 'buyer@acme.test',
  team_size: 4, report_kind: 'role_fit', target_occupation_code: CEO, team_pdf_url: null, ...over,
});
/** Strengths in the order a Chief Executive needs them: the perfect candidate for CEO, and its exact opposite. */
const ceoOrder = targetProfile(occupations.find((o) => o.code === '11-1011.00')!).map((r) => r.trait);
const reversed = [...ceoOrder].reverse();
const member = (id: string, name: string, over: { status?: string; consent?: boolean; order?: readonly string[] } = {}) => ({
  id,
  bundle: ladderBundle(
    { id, status: over.status ?? 'completed', participant_name: name },
    { name, email: `${name}@x.test`, ...(over.consent === false ? {} : { roleFitConsent: 'Yes' }) },
    over.order ?? TRAITS,
  ),
});
const setMembers = (members: ReturnType<typeof member>[]) => {
  vi.mocked(db.fetchMemberAssessmentIds).mockResolvedValue(members.map((m) => m.id));
  vi.mocked(db.fetchAssessmentData).mockImplementation(async (id) => members.find((m) => m.id === id)!.bundle);
};
const html = () => vi.mocked(htmlToPdf).mock.calls[0][0];

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.mocked(sendRoleFitReportEmail).mockReset().mockResolvedValue(undefined);
  vi.mocked(db.fetchCompany).mockResolvedValue(company());
  setMembers([member('a1', 'Alice Fit', { order: ceoOrder }), member('a2', 'Bob Far', { order: reversed }), member('a3', 'Carol Mid')]);
});
afterEach(() => {
  env.disableReportEmails = false;
});

describe('processRoleFitReport', () => {
  it('ranks every consenting, finished candidate against the chosen occupation and stores the PDF under the company', async () => {
    const result = await processRoleFitReport('co1');

    expect(result).toEqual({ pdfPath: 'role_fit_reports/co1/' + result.pdfPath.split('/').pop(), candidates: 3 });
    expect(result.pdfPath).toMatch(/^role_fit_reports\/co1\/SSM_Role_Fit_Acme_Pty_Ltd_\d{4}-\d{2}-\d{2}\.pdf$/);
    expect(db.uploadPdfToStorage).toHaveBeenCalledWith(env.reportsBucket, result.pdfPath, expect.any(Buffer));
    expect(html()).toContain('Occupation: Chief Executives');
    expect(html().indexOf('Alice Fit')).toBeLessThan(html().indexOf('Bob Far')); // the better fit is listed first
  });

  it('leaves out a candidate who has not consented: their name must never reach the buyer', async () => {
    setMembers([member('a1', 'Alice Fit'), member('a2', 'Private Pat', { consent: false })]);
    const result = await processRoleFitReport('co1');
    expect(result.candidates).toBe(1);
    expect(html()).toContain('Alice Fit');
    expect(html()).not.toContain('Private Pat');
  });

  it('leaves out a candidate who has not finished the assessment', async () => {
    setMembers([member('a1', 'Alice Fit'), member('a2', 'Still Working', { status: 'submitted' })]);
    await processRoleFitReport('co1');
    expect(html()).not.toContain('Still Working');
  });

  it('treats anything other than an explicit yes as no consent', async () => {
    const m = member('a2', 'Maybe Mo');
    m.bundle.demographics.roleFitConsent = 'No';
    setMembers([member('a1', 'Alice Fit'), m]);
    await processRoleFitReport('co1');
    expect(html()).not.toContain('Maybe Mo');
  });

  it('fails loudly, uploading and emailing nothing, when nobody has both finished and consented', async () => {
    setMembers([member('a1', 'No Consent', { consent: false }), member('a2', 'Unfinished', { status: 'submitted' })]);
    await expect(processRoleFitReport('co1')).rejects.toThrow(/no candidate/);
    expect(db.uploadPdfToStorage).not.toHaveBeenCalled();
    expect(sendRoleFitReportEmail).not.toHaveBeenCalled();
    expect(db.markCompanyReport).not.toHaveBeenCalled();
  });

  it('refuses a company that did not buy Role Fit, and an unknown occupation', async () => {
    vi.mocked(db.fetchCompany).mockResolvedValue(company({ report_kind: 'team' }));
    await expect(processRoleFitReport('co1')).rejects.toThrow(/not a Role Fit/);
    vi.mocked(db.fetchCompany).mockResolvedValue(company({ target_occupation_code: '00-0000.00' }));
    await expect(processRoleFitReport('co1')).rejects.toThrow(/Unknown target occupation/);
    expect(htmlToPdf).not.toHaveBeenCalled();
  });

  it('emails the buyer with the PDF, then records the report on the company', async () => {
    const { pdfPath } = await processRoleFitReport('co1');
    expect(sendRoleFitReportEmail).toHaveBeenCalledWith('buyer@acme.test', 'Acme Pty Ltd', expect.any(Buffer), expect.stringContaining('Role_Fit'));
    expect(db.markCompanyReport).toHaveBeenCalledWith('co1', pdfPath);
    const emailOrder = vi.mocked(sendRoleFitReportEmail).mock.invocationCallOrder[0];
    const markOrder = vi.mocked(db.markCompanyReport).mock.invocationCallOrder[0];
    expect(emailOrder).toBeLessThan(markOrder); // a failed email must not leave the company marked as done
  });

  it('throws when the email fails so the job is retried, and does not mark the company done', async () => {
    vi.mocked(sendRoleFitReportEmail).mockRejectedValue(new Error('resend down'));
    await expect(processRoleFitReport('co1')).rejects.toThrow(/resend down/);
    expect(db.markCompanyReport).not.toHaveBeenCalled();
  });

  it('stores the report but sends no email when emails are disabled, or the company has no contact email', async () => {
    env.disableReportEmails = true;
    await processRoleFitReport('co1');
    expect(sendRoleFitReportEmail).not.toHaveBeenCalled();
    expect(db.markCompanyReport).toHaveBeenCalled();
    env.disableReportEmails = false;
    vi.mocked(db.fetchCompany).mockResolvedValue(company({ contact_email: null }));
    await processRoleFitReport('co1');
    expect(sendRoleFitReportEmail).not.toHaveBeenCalled();
  });

  it('falls back to the organisation name for the team, and names a candidate with no name "Candidate N"', async () => {
    vi.mocked(db.fetchCompany).mockResolvedValue(company({ team_name: null }));
    const nameless = member('a1', '');
    nameless.bundle.demographics = { roleFitConsent: 'Yes' };
    nameless.bundle.assessment.participant_name = '';
    setMembers([nameless]);
    await processRoleFitReport('co1');
    expect(html()).toContain('Team: Acme Pty Ltd');
    expect(html()).toContain('Candidate 1');
  });

  it('keeps the storage path ASCII for an accented organisation name', async () => {
    vi.mocked(db.fetchCompany).mockResolvedValue(company({ company_name: 'Zoë Müller & Co' }));
    const { pdfPath } = await processRoleFitReport('co1');
    expect(pdfPath).toMatch(/^[A-Za-z0-9/._-]+$/);
  });
});

describe('runRoleFitJob: the lock set when the last candidate finished', () => {
  beforeEach(() => {
    vi.mocked(db.releaseCompanyReportLock).mockReset().mockResolvedValue(undefined);
    vi.mocked(db.fetchCompany).mockResolvedValue(company({ report_kind: 'team' })); // makes the report fail
  });

  it('keeps the lock while retries remain, so a retry is not triggered twice', async () => {
    await expect(runRoleFitJob('co1', { isLastAttempt: false })).rejects.toThrow();
    expect(db.releaseCompanyReportLock).not.toHaveBeenCalled();
  });

  it('releases the lock after the last attempt fails, so the report can be triggered again', async () => {
    await expect(runRoleFitJob('co1', { isLastAttempt: true })).rejects.toThrow(/not a Role Fit/);
    expect(db.releaseCompanyReportLock).toHaveBeenCalledWith('co1');
  });

  it('still reports the original failure if releasing the lock fails too', async () => {
    vi.mocked(db.releaseCompanyReportLock).mockRejectedValue(new Error('db down'));
    await expect(runRoleFitJob('co1', { isLastAttempt: true })).rejects.toThrow(/not a Role Fit/);
  });

  it('does not touch the lock when the report succeeds', async () => {
    vi.mocked(db.fetchCompany).mockResolvedValue(company());
    await runRoleFitJob('co1', { isLastAttempt: true });
    expect(db.releaseCompanyReportLock).not.toHaveBeenCalled();
  });
});

describe('processRoleFitReport: details the buyer sees', () => {
  it('shows the number of candidates the buyer paid for, not just those who consented', async () => {
    await processRoleFitReport('co1'); // team_size 4, three candidates
    expect(html()).toMatch(/Number of Team members:\s*(<[^>]*>\s*)*4/);
  });

  it('names a candidate from their demographics, then from the assessment, and otherwise as "Candidate N"', async () => {
    const noName = (id: string, participant: string, demoName?: string) => ({
      id,
      bundle: ladderBundle({ id, status: 'completed', participant_name: participant }, { ...(demoName ? { name: demoName } : {}), roleFitConsent: 'Yes' }, TRAITS),
    });
    setMembers([noName('b1', 'Assessment Ann', 'Demo Dan'), noName('b2', 'Assessment Ann'), noName('b3', '')]);
    await processRoleFitReport('co1');
    expect(html()).toContain('Demo Dan');
    expect(html()).toContain('Assessment Ann');
    expect(html()).toContain('Candidate 3');
    expect(html()).not.toContain('Candidate 1');
  });

  it('dates the report today, in a readable long form', async () => {
    vi.useFakeTimers({ toFake: ['Date'], now: new Date('2026-03-05T10:00:00Z') });
    try {
      await processRoleFitReport('co1');
      expect(html()).toContain('Date: 5 March 2026');
    } finally {
      vi.useRealTimers();
    }
  });

  it('records how many candidates were left out and why, so support can answer "where is my candidate?"', async () => {
    setMembers([member('a1', 'Alice Fit'), member('a2', 'Late Lee', { status: 'submitted' }), member('a3', 'Private Pat', { consent: false })]);
    await processRoleFitReport('co1');
    expect(vi.mocked(console.log).mock.calls.map((c) => c[0]).join('\n')).toContain('1 candidates (1 incomplete, 1 without consent)');
  });
});
