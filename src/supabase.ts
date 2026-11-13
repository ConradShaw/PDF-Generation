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
