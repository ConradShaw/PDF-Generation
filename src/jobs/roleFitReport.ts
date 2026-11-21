import { env } from '../env.js';
import { occupations } from '../careers/data.js';
import { sendRoleFitReportEmail } from '../email.js';
import { htmlToPdf } from '../pdf/render.js';
import { rankCandidates, targetProfile, type Candidate } from '../rolefit/engine.js';
import { buildSurveyResultsFromDb } from '../scoring/fromAnswers.js';
import { rankTraits } from '../scoring/tieBreaker.js';
import {
  fetchAssessmentData,
  fetchCompany,
  fetchMemberAssessmentIds,
  markCompanyReport,
  releaseCompanyReportLock,
  uploadPdfToStorage,
} from '../supabase.js';
import { renderRoleFitHtml, roleFitFilename } from '../templates/roleFit.js';

/** A candidate must have said yes to sharing their results with the buyer; the consent is stored with their demographics. */
const hasConsented = (demographics: Record<string, unknown>) => demographics.roleFitConsent === 'Yes';

function displayName(assessment: Record<string, unknown>, demographics: Record<string, unknown>, index: number): string {
  const name = String(demographics.name ?? assessment.participant_name ?? '').trim();
  return name || `Candidate ${index + 1}`;
}

/**
 * Role Fit report for a company whose members have all finished. Only candidates who completed the
 * assessment AND consented to sharing are included; anyone else is left out, not shown as a blank row.
 * Any failure throws so BullMQ retries (an email failure included).
 */
export async function processRoleFitReport(companyId: string): Promise<{ pdfPath: string; candidates: number }> {
  const company = await fetchCompany(companyId);
  if (company.report_kind !== 'role_fit') throw new Error(`Company ${companyId} is not a Role Fit purchase`);

  const occupation = occupations.find((o) => o.code === company.target_occupation_code);
  if (!occupation) throw new Error(`Unknown target occupation "${company.target_occupation_code}" for company ${companyId}`);

  const candidates: Candidate[] = [];
  let skippedIncomplete = 0;
  let skippedNoConsent = 0;
  for (const assessmentId of await fetchMemberAssessmentIds(companyId)) {
    const data = await fetchAssessmentData(assessmentId);
    if (data.assessment.status !== 'completed') {
      skippedIncomplete++;
      continue;
    }
    if (!hasConsented(data.demographics)) {
      skippedNoConsent++;
      continue;
    }
    const { ranks } = rankTraits(buildSurveyResultsFromDb(data.answers, data.questions, data.options));
    candidates.push({ id: assessmentId, name: displayName(data.assessment, data.demographics, candidates.length), ranks });
  }
  console.log(`[role-fit] ${companyId}: ${candidates.length} candidates (${skippedIncomplete} incomplete, ${skippedNoConsent} without consent)`);
  if (!candidates.length) throw new Error(`Role Fit ${companyId}: no candidate has both completed the assessment and consented`);

  const profile = targetProfile(occupation);
  const organisation = company.company_name;
  const html = renderRoleFitHtml({
    organisation,
    team: company.team_name || organisation,
    numMembers: company.team_size ?? candidates.length,
    dateStr: new Date().toISOString().slice(0, 10),
    occupation: { title: occupation.title, profileDate: occupation.profileDate },
    profile,
    candidates: rankCandidates(candidates, profile),
  });
  const pdfBytes = await htmlToPdf(html);
  const filename = roleFitFilename(organisation, new Date().toISOString().slice(0, 10));
  const storagePath = `role_fit_reports/${companyId}/${filename}`;
  await uploadPdfToStorage(env.reportsBucket, storagePath, pdfBytes);

  if (company.contact_email && !env.disableReportEmails) {
    await sendRoleFitReportEmail(company.contact_email, organisation, pdfBytes, filename);
  } else {
    console.log(`[role-fit] ${companyId}: email skipped`);
  }
  await markCompanyReport(companyId, storagePath);
  return { pdfPath: storagePath, candidates: candidates.length };
}

/**
 * The "all members finished" trigger sets team_report_generated as a lock before queueing this job.
 * If every attempt fails, release the lock so the report can be triggered again instead of being stuck.
 */
export async function runRoleFitJob(companyId: string, { isLastAttempt }: { isLastAttempt: boolean }) {
  try {
    return await processRoleFitReport(companyId);
  } catch (err) {
    if (isLastAttempt) await releaseCompanyReportLock(companyId).catch(() => {});
    throw err;
  }
}
