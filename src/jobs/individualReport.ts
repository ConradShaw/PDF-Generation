import { env } from '../env.js';
import { buildSurveyResultsFromDb } from '../scoring/fromAnswers.js';
import { rankTraits } from '../scoring/tieBreaker.js';
import {
  fetchAssessmentData,
  updateAssessmentStatus,
  uploadPdfToStorage,
  getSupabase,
} from '../supabase.js';
import { sendIndividualReportEmail } from '../email.js';
import { htmlToPdf } from '../pdf/render.js';
import {
  individualFilename,
  renderIndividualHtml,
} from '../templates/individual.js';
import { enqueueTeamReport } from '../queue/queues.js';

export async function processIndividualReport(
  assessmentId: string,
): Promise<{ pdfUrl: string }> {
  console.log(`[individual] Starting ${assessmentId}`);
  await updateAssessmentStatus(assessmentId, 'processing');

  try {
    const data = await fetchAssessmentData(assessmentId);
    const assessment = data.assessment;
    const demographics = data.demographics ?? {};

    let firstName = '';
    let lastName = '';
    const fullName = String(
      demographics.name ?? assessment.participant_name ?? '',
    );
    if (fullName.trim()) {
      const parts = fullName.trim().split(/\s+/);
      firstName = parts[0];
      lastName = parts.length > 1 ? parts.slice(1).join(' ') : parts[0];
    } else {
      firstName = 'Participant';
      lastName = 'Name';
    }

    const dateStr =
      String(assessment.submitted_at ?? '').slice(0, 10) ||
      new Date().toISOString().slice(0, 10);

    const results = buildSurveyResultsFromDb(
      data.answers,
      data.questions,
      data.options,
    );
    const { ordered, ranks } = rankTraits(results);

    const html = renderIndividualHtml({
      first: firstName,
      last: lastName,
      dateStr,
      orderedTraits: ordered,
      ranks,
    });
    const pdfBytes = await htmlToPdf(html);
    const pdfFilename = individualFilename(firstName, lastName, dateStr);
    const storagePath = `${assessmentId}/${pdfFilename}`;

    await uploadPdfToStorage(env.reportsBucket, storagePath, pdfBytes);
    await updateAssessmentStatus(assessmentId, 'completed', {
      pdf_storage_path: storagePath,
      processed_at: new Date().toISOString(),
    });
    console.log(`[individual] PDF uploaded: ${storagePath}`);

    const participantEmail = String(
      demographics.email ?? assessment.participant_email ?? '',
    );
    if (participantEmail && !env.disableReportEmails) {
      try {
        await sendIndividualReportEmail(
          participantEmail,
          firstName,
          assessmentId,
          pdfBytes,
          pdfFilename,
        );
      } catch (emailErr) {
        console.error('[individual] Email failed (non-fatal):', emailErr);
      }
    } else if (env.disableReportEmails) {
      console.log(`[individual] Email disabled; skipping ${assessmentId}`);
    }

    await checkAndEnqueueTeamReport(
      assessmentId,
      assessment,
      data.teamMember,
    );

    return { pdfUrl: storagePath };
  } catch (err) {
    try {
      await updateAssessmentStatus(assessmentId, 'failed');
    } catch {
      /* ignore */
    }
    throw err;
  }
}

async function checkAndEnqueueTeamReport(
  assessmentId: string,
  assessment: Record<string, unknown>,
  teamMember: Record<string, unknown> | null,
): Promise<void> {
  if (!teamMember) return;

  const sb = getSupabase();
  const companyId = String(
    assessment.company_id ?? teamMember.company_id ?? '',
  );
  if (!companyId) return;

  const { data: company } = await sb
    .from('companies')
    .select('*')
    .eq('id', companyId)
    .single();
  if (!company?.team_size) return;

  const { data: members } = await sb
    .from('team_assessment_members')
    .select('assessment_id')
    .eq('company_id', companyId);

  const assessmentIds = (members ?? [])
    .map((m) => m.assessment_id)
    .filter(Boolean) as string[];
  if (!assessmentIds.length) return;

  const { count } = await sb
    .from('assessments')
    .select('id', { count: 'exact', head: true })
    .in('id', assessmentIds)
    .eq('status', 'completed');

  const completedCount = count ?? 0;
  const teamSize = company.team_size as number;
  console.log(
    `[team-check] company=${companyId} completed=${completedCount}/${teamSize}`,
  );
  if (completedCount < teamSize) return;

  const { data: lock } = await sb
    .from('companies')
    .update({
      team_report_generated: true,
      team_report_generated_at: new Date().toISOString(),
    })
    .eq('id', companyId)
    .eq('team_report_generated', false)
    .select('id');

  if (!lock?.length) {
    console.log(`[team-check] Already generated for ${companyId} – skipping`);
    return;
  }

  console.log(
    `[team-check] All members complete – enqueue team report for ${companyId}`,
  );
  try {
    await enqueueTeamReport(companyId, true);
  } catch (err) {
    console.error('[team-check] Failed to enqueue team report:', err);
    await sb
      .from('companies')
      .update({
        team_report_generated: false,
        team_report_generated_at: null,
      })
      .eq('id', companyId);
  }
}
