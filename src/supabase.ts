import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { env } from './env.js';

let client: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient {
  if (!client) {
    if (!env.supabaseUrl || !env.supabaseServiceRoleKey) {
      throw new Error('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set');
    }
    client = createClient(env.supabaseUrl, env.supabaseServiceRoleKey);
  }
  return client;
}

export type AssessmentBundle = {
  assessment: Record<string, unknown>;
  demographics: Record<string, unknown>;
  answers: Record<string, string>;
  questions: Array<{ id: number | string; question?: string; type?: string }>;
  options: Array<{
    id: string;
    question_id: number | string;
    option_text?: string;
    construct: string;
  }>;
  teamMember: Record<string, unknown> | null;
};

export async function fetchAssessmentData(
  assessmentId: string,
): Promise<AssessmentBundle> {
  const sb = getSupabase();

  const { data: assessment, error: aErr } = await sb
    .from('assessments')
    .select('*')
    .eq('id', assessmentId)
    .single();
  if (aErr || !assessment) {
    throw new Error(`Assessment ${assessmentId} not found`);
  }

  const { data: demoResp } = await sb
    .from('assessment_demographics')
    .select('demographic_data')
    .eq('assessment_id', assessmentId)
    .maybeSingle();

  const { data: answersRows } = await sb
    .from('assessment_answers')
    .select('question_id, option_id')
    .eq('assessment_id', assessmentId)
    .order('question_id');

  const { data: questionsRows } = await sb
    .from('assessment_questions')
    .select('id, question, type')
    .order('id');

  const { data: optionsRows } = await sb
    .from('assessment_question_options')
    .select('id, question_id, option_text, construct')
    .order('question_id')
    .order('id');

  const { data: teamMember } = await sb
    .from('team_assessment_members')
    .select('*')
    .eq('assessment_id', assessmentId)
    .maybeSingle();

  const answers: Record<string, string> = {};
  for (const a of answersRows ?? []) {
    answers[String(a.question_id)] = a.option_id;
  }

  return {
    assessment,
    demographics: (demoResp?.demographic_data as Record<string, unknown>) ?? {},
    answers,
    questions: questionsRows ?? [],
    options: optionsRows ?? [],
    teamMember: teamMember ?? null,
  };
}

export async function updateAssessmentStatus(
  assessmentId: string,
  status: string,
  extra?: Record<string, unknown>,
): Promise<void> {
  const sb = getSupabase();
  const { error } = await sb
    .from('assessments')
    .update({ status, ...(extra ?? {}) })
    .eq('id', assessmentId);
  if (error) throw error;
}

export type CareerPurchase = {
  id: string;
  assessment_id: string;
  status: string;
  riasec: Record<string, number> | null;
  customer_email: string | null;
};

export async function fetchCareerPurchase(purchaseId: string): Promise<CareerPurchase> {
  const { data, error } = await getSupabase()
    .from('career_match_purchases')
    .select('id, assessment_id, status, riasec, customer_email')
    .eq('id', purchaseId)
    .single();
  if (error || !data) throw new Error(`Career match purchase ${purchaseId} not found`);
  return data as CareerPurchase;
}

export async function updateCareerPurchase(
  purchaseId: string,
  fields: Record<string, unknown>,
): Promise<void> {
  const { error } = await getSupabase()
    .from('career_match_purchases')
    .update(fields)
    .eq('id', purchaseId);
  if (error) throw error;
}

export async function uploadPdfToStorage(
  bucket: string,
  path: string,
  pdfBytes: Buffer,
): Promise<void> {
  const sb = getSupabase();
  const { error } = await sb.storage.from(bucket).upload(path, pdfBytes, {
    contentType: 'application/pdf',
    upsert: true,
  });
  if (error) throw error;
}

export type CompanyRow = {
  id: string;
  company_name: string;
  team_name: string | null;
  contact_email: string | null;
  team_size: number | null;
  report_kind: string | null;
  target_occupation_code: string | null;
  team_pdf_url: string | null;
};

export async function fetchCompany(companyId: string): Promise<CompanyRow> {
  const { data, error } = await getSupabase().from('companies').select('*').eq('id', companyId).single();
  if (error || !data) throw new Error(`Company ${companyId} not found`);
  return data as CompanyRow;
}

export async function fetchMemberAssessmentIds(companyId: string): Promise<string[]> {
  const { data, error } = await getSupabase()
    .from('team_assessment_members')
    .select('assessment_id')
    .eq('company_id', companyId);
  if (error) throw error;
  return (data ?? []).map((m) => m.assessment_id as string | null).filter((id): id is string => !!id);
}

export async function markCompanyReport(companyId: string, storagePath: string): Promise<void> {
  const { error } = await getSupabase()
    .from('companies')
    .update({
      team_report_generated: true,
      team_report_generated_at: new Date().toISOString(),
      team_pdf_url: storagePath,
      team_pdf_storage_path: storagePath,
    })
    .eq('id', companyId);
  if (error) throw error;
}

/** Let a company whose report never got produced be triggered again (only if no report exists). */
export async function releaseCompanyReportLock(companyId: string): Promise<void> {
  const { error } = await getSupabase()
    .from('companies')
    .update({ team_report_generated: false, team_report_generated_at: null })
    .eq('id', companyId)
    .is('team_pdf_url', null);
  if (error) throw error;
}
