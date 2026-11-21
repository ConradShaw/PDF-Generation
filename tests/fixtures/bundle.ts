import { TRAITS } from '../../src/scoring/traits.js';
import type { AssessmentBundle } from '../../src/supabase.js';

/**
 * A complete 66-question assessment (every pair of the 12 strengths once) in which the
 * strength listed earlier in `order` always wins its pairs: a strict ladder, order[0] = rank 1.
 * `order` defaults to TRAITS; pass another order to make a different strength rank first.
 */
export function ladderBundle(
  assessment: Record<string, unknown> = {},
  demographics: Record<string, unknown> = { name: 'Ada Lovelace', email: 'ada@example.com' },
  order: readonly string[] = TRAITS,
  cycleTop = false,
): AssessmentBundle {
  const questions: AssessmentBundle['questions'] = [];
  const options: AssessmentBundle['options'] = [];
  const answers: Record<string, string> = {};
  let id = 0;
  for (let i = 0; i < order.length; i++) {
    for (let j = i + 1; j < order.length; j++) {
      id += 1;
      questions.push({ id });
      options.push({ id: `q${id}-a`, question_id: id, construct: order[i] });
      options.push({ id: `q${id}-b`, question_id: id, construct: order[j] });
      // cycleTop: order[0] beats order[1], order[1] beats order[2], order[2] beats order[0],
      // so the top three cannot be separated and tie for first.
      const reversed = cycleTop && i === 0 && j === 2;
      answers[String(id)] = reversed ? `q${id}-b` : `q${id}-a`;
    }
  }
  return {
    assessment: { id: 'assess-1', submitted_at: '2026-11-19T01:00:00Z', ...assessment },
    demographics,
    answers,
    questions,
    options,
    teamMember: null,
  };
}
