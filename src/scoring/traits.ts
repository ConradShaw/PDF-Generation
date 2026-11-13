export const TRAITS = [
  'Courage',
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
