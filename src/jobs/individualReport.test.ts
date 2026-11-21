import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ladderBundle } from '../../tests/fixtures/bundle.js';
import { TRAITS } from '../scoring/traits.js';
import { archetypeFor } from '../careers/data.js';

vi.mock('../supabase.js', () => ({
  fetchAssessmentData: vi.fn(),
  updateAssessmentStatus: vi.fn(),
  uploadPdfToStorage: vi.fn(),
  getSupabase: vi.fn(),
}));
vi.mock('../pdf/render.js', () => ({ htmlToPdf: vi.fn(async () => Buffer.from('%PDF-fake')) }));
vi.mock('../email.js', () => ({
  sendIndividualReportEmail: vi.fn(),
  sendProfileReportEmail: vi.fn(),
}));
vi.mock('../queue/queues.js', () => ({ enqueueTeamReport: vi.fn() }));
// Keep the real data, but make the production guard observable and controllable.
vi.mock('../careers/data.js', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../careers/data.js')>()),
  assertOwnerDataReady: vi.fn(),
}));

import { env } from '../env.js';
import * as db from '../supabase.js';
import { htmlToPdf } from '../pdf/render.js';
import { sendIndividualReportEmail, sendProfileReportEmail } from '../email.js';
import { assertOwnerDataReady } from '../careers/data.js';
import { processIndividualReport } from './individualReport.js';

const finalStatus = () => vi.mocked(db.updateAssessmentStatus).mock.calls.at(-1)?.[1];

beforeEach(() => vi.clearAllMocks());
afterEach(() => {
  env.isProduction = false;
});

describe('processIndividualReport: free Strength Profile (psychometrics route)', () => {
  beforeEach(() => {
    vi.mocked(db.fetchAssessmentData).mockResolvedValue(ladderBundle({ assessment_type: 'psychometrics' }));
  });

  it('renders the short profile and emails it with a link to unlock Career Matches for this assessment', async () => {
    await processIndividualReport('assess-1');

    const html = vi.mocked(htmlToPdf).mock.calls[0][0];
    expect(html).toContain('Strength Profile');
    expect(html).toContain('Your Work Archetype');
    expect(html).not.toContain('Mapping to O*NET'); // the full paid-report pages are not included
    expect(sendProfileReportEmail).toHaveBeenCalledWith(
      'ada@example.com',
      'Ada',
      expect.any(Buffer),
      expect.stringContaining('Strength_Profile'),
      `${env.appUrl}/career-match/unlock?assessment=assess-1`,
    );
    expect(sendIndividualReportEmail).not.toHaveBeenCalled();
    expect(finalStatus()).toBe('completed');
  });

  it('shows the archetype of the #1 ranked strength only, not of any other strength', async () => {
    const order = ['Empathy', ...TRAITS.filter((t) => t !== 'Empathy')]; // #1 is deliberately not TRAITS[0]
    vi.mocked(db.fetchAssessmentData).mockResolvedValue(ladderBundle({ assessment_type: 'psychometrics' }, undefined, order));

    await processIndividualReport('assess-1');

    const html = vi.mocked(htmlToPdf).mock.calls[0][0];
    expect(html).toContain(archetypeFor('Empathy'));
    for (const t of TRAITS.filter((t) => t !== 'Empathy')) {
      expect(html).not.toContain(archetypeFor(t));
    }
  });

  it('refuses to email when the owner data guard refuses, and marks the assessment failed', async () => {
    vi.mocked(assertOwnerDataReady).mockImplementationOnce(() => {
      throw new Error('Refusing to send report with placeholder owner data');
    });
    await expect(processIndividualReport('assess-1')).rejects.toThrow(/placeholder/i);
    expect(finalStatus()).toBe('failed');
    expect(sendProfileReportEmail).not.toHaveBeenCalled();
  });
});

describe('processIndividualReport: free profile with a tie for first', () => {
  it('shows the archetype of every strength tied for #1 and of no other strength', async () => {
    // Top three form a cycle (A beats B, B beats C, C beats A), so they share rank 2 and tie for first.
    vi.mocked(db.fetchAssessmentData).mockResolvedValue(
      ladderBundle({ assessment_type: 'psychometrics' }, undefined, TRAITS, true),
    );

    await processIndividualReport('assess-1');

    const html = vi.mocked(htmlToPdf).mock.calls[0][0];
    for (const t of TRAITS.slice(0, 3)) expect(html).toContain(archetypeFor(t));
    for (const t of TRAITS.slice(3)) expect(html).not.toContain(archetypeFor(t));
    expect(html).toContain('tied');
  });
});

describe('processIndividualReport: paid individual assessment', () => {
  beforeEach(() => {
    vi.mocked(db.fetchAssessmentData).mockResolvedValue(ladderBundle({ assessment_type: 'participants' }));
  });

  it('is unchanged: full report, individual email, no unlock upsell, and no owner-data guard', async () => {
    await processIndividualReport('assess-1');
    expect(assertOwnerDataReady).not.toHaveBeenCalled(); // the guard applies only to the free profile's owner data

    const html = vi.mocked(htmlToPdf).mock.calls[0][0];
    expect(html).toContain('Mapping to O*NET');
    expect(html).not.toContain('Unlock Your Career Matches');
    expect(sendIndividualReportEmail).toHaveBeenCalled();
    expect(sendProfileReportEmail).not.toHaveBeenCalled();
    expect(finalStatus()).toBe('completed');
  });
});
