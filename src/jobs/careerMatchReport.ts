import { env } from '../env.js';
import { buildSurveyResultsFromDb } from '../scoring/fromAnswers.js';
import { rankTraits } from '../scoring/tieBreaker.js';
import {
  fetchAssessmentData,
  fetchCareerPurchase,
  updateCareerPurchase,
  uploadPdfToStorage,
} from '../supabase.js';
import { sendCareerMatchEmail } from '../email.js';
import { htmlToPdf } from '../pdf/render.js';
import { careerFilename, renderCareerHtml } from '../templates/career.js';
import { RIASEC_AREAS } from '../careers/constants.js';
import {
  archetypeFor,
  assertOwnerDataReady,
  occupations,
  onetAttribution,
} from '../careers/data.js';
import { matchCareers } from '../careers/matching.js';
import type { Riasec } from '../careers/types.js';
import { participantName, recipientEmail, reportDate } from './shared.js';

function asRiasec(raw: Record<string, number> | null): Riasec {
  if (!raw || RIASEC_AREAS.some((a) => typeof raw[a] !== 'number')) {
    throw new Error('Career match purchase has no complete RIASEC result yet');
  }
  return raw as unknown as Riasec;
}

/**
 * Paid Career Matches report. Reuses the stored 66 answers (the assessment is never retaken)
 * and the RIASEC result saved from the O*NET Interest Profiler after payment.
 *
 * Any failure, including a failed email, throws so BullMQ retries (the PDF is simply rebuilt).
 * The purchase is marked `failed` only on the LAST attempt: marking it earlier would let the
 * customer resubmit (and queue a second job) while a retry of the first is still pending.
 */
export async function processCareerMatchReport(
  purchaseId: string,
  { isLastAttempt = true }: { isLastAttempt?: boolean } = {},
): Promise<{ pdfPath: string }> {
  console.log(`[career] Starting ${purchaseId}`);
  await updateCareerPurchase(purchaseId, { status: 'processing' });

  try {
    assertOwnerDataReady();
    const purchase = await fetchCareerPurchase(purchaseId);
    const riasec = asRiasec(purchase.riasec);

    const data = await fetchAssessmentData(purchase.assessment_id);
    const results = buildSurveyResultsFromDb(data.answers, data.questions, data.options);
    const { ordered, ranks } = rankTraits(results);

    const matches = matchCareers(ranks, riasec, occupations);
    console.log(`[career] ${purchaseId}: ${matches.length} occupations shown`);

    // The report goes to the assessment's email; if there is none, to whoever paid.
    const email = recipientEmail(data.assessment, data.demographics) || purchase.customer_email || '';
    if (!email && !env.disableReportEmails) {
      throw new Error(`Career match ${purchaseId}: no recipient email on the assessment or the payment`);
    }

    const { first, last } = participantName(data.assessment, data.demographics);
    const dateStr = reportDate(data.assessment);
    const html = renderCareerHtml({
      first,
      last,
      dateStr,
      orderedTraits: ordered,
      ranks,
      riasec,
      matches,
      archetypes: ordered.map((trait) => ({ trait, text: archetypeFor(trait) })),
      attribution: onetAttribution,
    });
    const pdfBytes = await htmlToPdf(html);
    const filename = careerFilename(first, last, dateStr);
    const storagePath = `${purchase.assessment_id}/career/${filename}`;
    await uploadPdfToStorage(env.reportsBucket, storagePath, pdfBytes);

    let emailedAt: string | null = null;
    if (env.disableReportEmails) {
      console.log(`[career] Email disabled; skipping ${purchaseId}`);
    } else {
      await sendCareerMatchEmail(email, first, pdfBytes, filename);
      emailedAt = new Date().toISOString();
    }

    await updateCareerPurchase(purchaseId, {
      status: 'completed',
      pdf_storage_path: storagePath,
      processed_at: new Date().toISOString(),
      emailed_at: emailedAt,
    });
    return { pdfPath: storagePath };
  } catch (err) {
    if (isLastAttempt) {
      try {
        await updateCareerPurchase(purchaseId, { status: 'failed' });
      } catch {
        /* ignore: the original error is what matters */
      }
    }
    throw err;
  }
}
