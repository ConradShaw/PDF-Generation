export const TRAITS = [
  'Initiative',
  'Practicality',
  'Curiosity',
  'Prudence',
  'Confidence',
  'Discernment',
  'Fairness',
  'Tenacity',
  'Foresight',
  'Discipline',
  'Objectivity',
  'Empathy',
] as const;

export type Trait = (typeof TRAITS)[number];

export type Category =
  | 'Signature'
  | 'Supporting'
  | 'Stretch'
  | 'Situational';

export type SurveyResults = Record<string, Record<string, number>>;

export function categoryForRankNumber(rankNum: number): Category {
  if (rankNum <= 3) return 'Signature';
  if (rankNum <= 6) return 'Supporting';
  if (rankNum <= 9) return 'Stretch';
  return 'Situational';
}

export function colorForCategory(cat: Category): string {
  switch (cat) {
    case 'Signature':
      return '#4d93d9';
    case 'Supporting':
      return '#94dcf8';
    case 'Stretch':
      return '#b7e4c7';
    default:
      return '#d0d0d0';
  }
}

export function formatRank(r: number): string {
  return Number.isInteger(r) ? String(r) : r.toFixed(1);
}

export function emptyResults(): SurveyResults {
  const results: SurveyResults = {};
  for (const t of TRAITS) {
    results[t] = {};
    for (const u of TRAITS) {
      if (u !== t) results[t][u] = 0;
    }
  }
  return results;
}

/**
 * The strength(s) ranked first: more than one when the tie-breaker could not separate them
 * (they then share the same average rank). `ordered` is best-first, as rankTraits returns it.
 */
export function leadingStrengths(
  ordered: string[],
  ranks: Record<string, number>,
): string[] {
  const best = ranks[ordered[0]];
  return ordered.filter((t) => ranks[t] === best);
}
