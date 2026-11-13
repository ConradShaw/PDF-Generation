import { env } from '../env.js';
import { buildSurveyResultsFromDb } from '../scoring/fromAnswers.js';
import { rankTraits } from '../scoring/tieBreaker.js';
import { calculateTeamRankings } from '../scoring/teamRankings.js';
import {
  fetchAssessmentData,
  uploadPdfToStorage,
  getSupabase,
} from '../supabase.js';
import { sendTeamReportEmail } from '../email.js';
import { htmlToPdf } from '../pdf/render.js';
import { renderTeamHtml, teamFilename } from '../templates/team.js';

export async function processTeamReport(
  companyId: string,
  force = false,
): Promise<{ pdfUrl: string; message: string }> {
  const sb = getSupabase();

  const { data: company, error } = await sb
    .from('companies')
    .select('*')
    .eq('id', companyId)
    .single();
  if (error || !company) {
    throw new Error('Company not found');
  }

  if (company.team_report_generated && !force) {
    return {
      pdfUrl: String(company.team_pdf_url ?? ''),
      message: 'Team report already generated',
    };
  }

  const { data: members } = await sb
    .from('team_assessment_members')
    .select('assessment_id')
    .eq('company_id', companyId);

  const individualResults: Array<{
    ordered_traits: string[];
    ranks: Record<string, number>;
  }> = [];

  for (const member of members ?? []) {
    const aid = member.assessment_id as string | null;
    if (!aid) continue;
    try {
      const adata = await fetchAssessmentData(aid);
      if (adata.assessment.status !== 'completed') continue;
      const survey = buildSurveyResultsFromDb(
        adata.answers,
        adata.questions,
        adata.options,
      );
      const { ordered, ranks } = rankTraits(survey);
      individualResults.push({ ordered_traits: ordered, ranks });
    } catch (e) {
      console.warn(`[team-report] Skipping assessment ${aid}:`, e);
    }
  }

  if (!individualResults.length) {
    throw new Error('No completed individual results found');
  }

  const { orderedTraits, finalRanks, distributionData } =
    calculateTeamRankings(individualResults);

  const dateStr = new Date().toISOString().slice(0, 10);
  const companyName = String(company.company_name ?? 'Company');
  const teamName = String(
    company.team_name ?? company.company_name ?? 'Team',
  );
  const numMembers = Number(
    company.team_size ?? individualResults.length,
  );

  const html = renderTeamHtml({
    companyName,
    teamName,
    numMembers,
    dateStr,
    orderedTraits,
    ranks: finalRanks,
    distributionData,
  });
  const pdfBytes = await htmlToPdf(html);
  const pdfFilename = teamFilename(companyName, dateStr);
  const storagePath = `team_reports/${companyId}/${pdfFilename}`;

  await uploadPdfToStorage(env.reportsBucket, storagePath, pdfBytes);

  await sb
    .from('companies')
    .update({
      team_report_generated: true,
      team_report_generated_at: new Date().toISOString(),
      team_pdf_url: storagePath,
      team_pdf_storage_path: storagePath,
    })
    .eq('id', companyId);

  console.log(`[team-report] PDF uploaded: ${storagePath}`);

  const contactEmail = company.contact_email as string | null;
  if (contactEmail && !env.disableReportEmails) {
    try {
      await sendTeamReportEmail(
        contactEmail,
        companyName,
        String(company.company_code ?? ''),
        numMembers,
        pdfBytes,
        pdfFilename,
      );
    } catch (emailErr) {
      console.error('[team-report] Email failed (non-fatal):', emailErr);
    }
  } else if (env.disableReportEmails) {
    console.log(`[team-report] Email disabled; skipping ${companyId}`);
  }

  return {
    pdfUrl: storagePath,
    message: 'Team report generated successfully',
  };
}
