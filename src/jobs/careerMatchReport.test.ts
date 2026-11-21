import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ladderBundle } from '../../tests/fixtures/bundle.js';
import { TRAITS } from '../scoring/traits.js';
import { archetypeFor } from '../careers/data.js';

vi.mock('../supabase.js', () => ({
  fetchAssessmentData: vi.fn(),
  fetchCareerPurchase: vi.fn(),
  updateCareerPurchase: vi.fn(),
  uploadPdfToStorage: vi.fn(),
}));
vi.mock('../pdf/render.js', () => ({ htmlToPdf: vi.fn(async () => Buffer.from('%PDF-fake')) }));
vi.mock('../email.js', () => ({ sendCareerMatchEmail: vi.fn() }));
// Keep the real data, but make the production guard observable and controllable.
vi.mock('../careers/data.js', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../careers/data.js')>()),
  assertOwnerDataReady: vi.fn(),
}));

import { env } from '../env.js';
import * as db from '../supabase.js';
import { htmlToPdf } from '../pdf/render.js';
import { sendCareerMatchEmail } from '../email.js';
import { assertOwnerDataReady } from '../careers/data.js';
import { processCareerMatchReport } from './careerMatchReport.js';

const riasec = { R: 8, I: 18, A: 9, S: 12, E: 22, C: 15 };
const purchase = (over: Record<string, unknown> = {}) => ({
  id: 'p1',
  assessment_id: 'assess-1',
  status: 'interests_submitted',
  riasec,
  customer_email: 'payer@example.com',
  ...over,
});
const updates = () => vi.mocked(db.updateCareerPurchase).mock.calls.map(([, fields]) => fields as Record<string, unknown>);
const statuses = () => updates().map((f) => f.status);

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(db.fetchCareerPurchase).mockResolvedValue(purchase());
  vi.mocked(db.fetchAssessmentData).mockResolvedValue(ladderBundle());
  vi.spyOn(console, 'log').mockImplementation(() => {});
});
afterEach(() => {
  env.isProduction = false;
  env.disableReportEmails = false;
});

describe('processCareerMatchReport: success', () => {
  it('reads the stored 66 answers (no retake), stores the PDF, emails it and records when it was emailed', async () => {
    const { pdfPath } = await processCareerMatchReport('p1');

    expect(db.fetchAssessmentData).toHaveBeenCalledWith('assess-1');
    expect(pdfPath).toBe('assess-1/career/SSM_Career_Matches_Ada_Lovelace_2026-11-19.pdf');
    expect(db.uploadPdfToStorage).toHaveBeenCalledWith(env.reportsBucket, pdfPath, expect.any(Buffer));
    expect(sendCareerMatchEmail).toHaveBeenCalledWith('ada@example.com', 'Ada', expect.any(Buffer), expect.stringContaining('Career_Matches'));
    expect(statuses()).toEqual(['processing', 'completed']);
    expect(updates()[1]).toMatchObject({ pdf_storage_path: pdfPath, emailed_at: expect.any(String) });
  });

  it('renders the Work Archetype of all 12 strengths, the #1 strength first, plus star ratings and the O*NET attribution', async () => {
    vi.mocked(db.fetchAssessmentData).mockResolvedValue(ladderBundle({}, undefined, ['Empathy', ...TRAITS.filter((t) => t !== 'Empathy')]));
    await processCareerMatchReport('p1');
    const html = vi.mocked(htmlToPdf).mock.calls[0][0];
    for (const t of TRAITS) expect(html).toContain(archetypeFor(t));
    expect(html.indexOf(archetypeFor('Empathy'))).toBeLessThan(html.indexOf(archetypeFor('Initiative')));
    expect(html).toContain('data-stars="5"');
    expect(html).toContain('data-stars="1"');
    expect(html).toContain('<td class="rank">25</td>'); // the final list is the top 25
    expect(html).toContain('CC BY 4.0');
  });

  it('keeps the storage path ASCII for accented names: Supabase Storage rejects other keys', async () => {
    vi.mocked(db.fetchAssessmentData).mockResolvedValue(ladderBundle({}, { name: 'José Zoë', email: 'j@example.com' }));
    const { pdfPath } = await processCareerMatchReport('p1');
    expect(pdfPath).toMatch(/^[A-Za-z0-9/._-]+$/);
    expect(pdfPath).toContain('Jose_Zoe');
  });

  it('falls back to the payer\'s email when the assessment has none, so a paid report is never undeliverable', async () => {
    vi.mocked(db.fetchAssessmentData).mockResolvedValue(ladderBundle({}, { name: 'Ada Lovelace' }));
    await processCareerMatchReport('p1');
    expect(sendCareerMatchEmail).toHaveBeenCalledWith('payer@example.com', 'Ada', expect.any(Buffer), expect.any(String));
  });

  it('stores the report but sends no email, and records no emailed_at, when emails are disabled', async () => {
    env.disableReportEmails = true;
    await processCareerMatchReport('p1');
    expect(db.uploadPdfToStorage).toHaveBeenCalled();
    expect(sendCareerMatchEmail).not.toHaveBeenCalled();
    expect(updates()[1].emailed_at).toBeNull();
  });
});

describe('processCareerMatchReport: failures', () => {
  it('fails loudly, and sends nothing, when the interest result is missing', async () => {
    vi.mocked(db.fetchCareerPurchase).mockResolvedValue(purchase({ status: 'paid', riasec: null }));
    await expect(processCareerMatchReport('p1')).rejects.toThrow(/RIASEC/);
    expect(statuses()).toEqual(['processing', 'failed']);
    expect(db.uploadPdfToStorage).not.toHaveBeenCalled();
    expect(sendCareerMatchEmail).not.toHaveBeenCalled();
  });

  it('rejects an interest result with a missing area, because a partial RIASEC would corrupt the career filter', async () => {
    const { C: _omitted, ...partial } = riasec;
    vi.mocked(db.fetchCareerPurchase).mockResolvedValue(purchase({ riasec: partial }));
    await expect(processCareerMatchReport('p1')).rejects.toThrow(/RIASEC/);
    expect(sendCareerMatchEmail).not.toHaveBeenCalled();
  });

  it('checks the owner data guard before doing any work, and sends nothing when it refuses', async () => {
    vi.mocked(assertOwnerDataReady).mockImplementationOnce(() => {
      throw new Error('Refusing to send report with placeholder owner data');
    });
    await expect(processCareerMatchReport('p1')).rejects.toThrow(/placeholder/i);
    expect(db.uploadPdfToStorage).not.toHaveBeenCalled();
    expect(statuses()).toEqual(['processing', 'failed']);
    expect(sendCareerMatchEmail).not.toHaveBeenCalled();
  });

  it('throws when there is nowhere to send the report, rather than completing a paid order silently', async () => {
    vi.mocked(db.fetchAssessmentData).mockResolvedValue(ladderBundle({}, { name: 'Ada' }));
    vi.mocked(db.fetchCareerPurchase).mockResolvedValue(purchase({ customer_email: null }));
    await expect(processCareerMatchReport('p1')).rejects.toThrow(/no recipient email/);
    expect(db.uploadPdfToStorage).not.toHaveBeenCalled();
  });

  it('throws when the email fails so BullMQ retries it, and never records emailed_at or "completed"', async () => {
    vi.mocked(sendCareerMatchEmail).mockRejectedValue(new Error('resend down'));
    await expect(processCareerMatchReport('p1')).rejects.toThrow(/resend down/);
    expect(statuses()).not.toContain('completed');
    expect(updates().some((f) => 'emailed_at' in f)).toBe(false);
  });

  it('marks the purchase failed only on the last attempt: earlier failures leave it processing so no second job can be queued', async () => {
    vi.mocked(sendCareerMatchEmail).mockRejectedValue(new Error('resend down'));

    await expect(processCareerMatchReport('p1', { isLastAttempt: false })).rejects.toThrow();
    expect(statuses()).toEqual(['processing']);

    vi.mocked(db.updateCareerPurchase).mockClear();
    await expect(processCareerMatchReport('p1', { isLastAttempt: true })).rejects.toThrow();
    expect(statuses()).toEqual(['processing', 'failed']);
  });
});
