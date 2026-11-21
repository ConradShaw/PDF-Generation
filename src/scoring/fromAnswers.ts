import { emptyResults, type SurveyResults } from './traits.js';

/**
 * Strength names stored in the question database under their former names. "Initiative" was
 * "Courage" until 2026-11; the option rows still say "Courage", so map them rather than edit the data.
 */
const LEGACY_CONSTRUCT_NAMES: Record<string, string> = { Courage: 'Initiative' };

export type QuestionRow = { id: number | string; question?: string; type?: string };
export type OptionRow = {
  id: string;
  question_id: number | string;
  option_text?: string;
  construct: string;
};

/**
 * Port of build_survey_results_from_db.
 */
export function buildSurveyResultsFromDb(
  answers: Record<string, string>,
  questions: QuestionRow[],
  options: OptionRow[],
): SurveyResults {
  const optionsByQuestion = new Map<string, OptionRow[]>();
  for (const opt of options) {
    const qid = String(opt.question_id);
    const list = optionsByQuestion.get(qid) ?? [];
    list.push(opt);
    optionsByQuestion.set(qid, list);
  }

  const results = emptyResults();
  const traitsSet = new Set(Object.keys(results));

  for (const q of questions) {
    const qid = String(q.id);
    const chosenOptionId = answers[qid];
    if (!chosenOptionId) continue;

    const qOpts = optionsByQuestion.get(qid) ?? [];
    if (qOpts.length !== 2) continue;

    // Match Python str.title()
    const title = (s: string) =>
      s
        .trim()
        .toLowerCase()
        .replace(/(^|[^A-Za-z0-9'])([a-z])/g, (_, p1: string, p2: string) => p1 + p2.toUpperCase());

    const canonical = (s: string) => LEGACY_CONSTRUCT_NAMES[title(s)] ?? title(s);
    const a = canonical(qOpts[0].construct);
    const b = canonical(qOpts[1].construct);

    if (!traitsSet.has(a) || !traitsSet.has(b) || a === b) continue;

    let winner: string;
    let loser: string;
    if (chosenOptionId === qOpts[0].id) {
      winner = a;
      loser = b;
    } else if (chosenOptionId === qOpts[1].id) {
      winner = b;
      loser = a;
    } else {
      continue;
    }

    results[winner][loser] = 1;
  }

  return results;
}

/** Set a directed win (winner beats loser). */
export function setWin(results: SurveyResults, winner: string, loser: string): void {
  if (!results[winner] || winner === loser) return;
  results[winner][loser] = 1;
  results[loser][winner] = 0;
}
