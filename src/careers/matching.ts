import { TRAITS } from '../scoring/traits.js';
import {
  FINAL_LIST_SIZE,
  PRE_FILTER_SIZE,
  RIASEC_AREAS,
  STAR_LEVELS,
  STRENGTH_COUNT,
  WEIGHT_RATIO,
} from './constants.js';
import type { Occupation, RatedOccupation, Riasec } from './types.js';

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

/**
 * Interest fit: correlation (-1 to 1) between the user's and the occupation's six RIASEC scores.
 * Correlation is scale-free (the user's 5-25 sums and O*NET's 1-7 ratings compare directly) and
 * does not favour occupations merely for having strong peaks. 0 when either side has no variation.
 */
export function riasecFit(user: Riasec, occupation: Riasec): number {
  const u = RIASEC_AREAS.map((a) => user[a]);
  const o = RIASEC_AREAS.map((a) => occupation[a]);
  const um = mean(u);
  const om = mean(o);
  let dot = 0;
  let uu = 0;
  let oo = 0;
  u.forEach((ui, i) => {
    dot += (ui - um) * (o[i] - om);
    uu += (ui - um) ** 2;
    oo += (o[i] - om) ** 2;
  });
  return uu === 0 || oo === 0 ? 0 : dot / Math.sqrt(uu * oo);
}

/**
 * Stage 1: the occupations whose interests fit best, at most `size`, and only those with positive fit.
 * A user whose six scores are identical (e.g. "Unsure" to every question) has no preference: every fit
 * would be 0 and the report empty, so nothing is excluded for them.
 */
export function shortlistByInterest(
  user: Riasec,
  occupations: Occupation[],
  size: number = PRE_FILTER_SIZE,
): Occupation[] {
  const scores = RIASEC_AREAS.map((a) => user[a]);
  if (scores.every((s) => s === scores[0])) return occupations;
  return occupations
    .map((occ) => ({ occ, fit: riasecFit(user, occ.riasec) }))
    .filter((x) => x.fit > 0)
    .sort((a, b) => b.fit - a.fit || a.occ.code.localeCompare(b.occ.code))
    .slice(0, size)
    .map((x) => x.occ);
}

/** Rank 1 -> 100, rank 12 -> 0, linear. Ranks may be fractional after tie-breaking. */
function levelForRank(rank: number): number {
  return (100 * (STRENGTH_COUNT - rank)) / (STRENGTH_COUNT - 1);
}

/**
 * An occupation's requirements rescaled to 0-100 within that occupation (its least-demanded
 * strength = 0, its most-demanded = 100). The user's levels always span 0-100 (rank 12 to rank 1),
 * so comparing them with raw O*NET levels favoured occupations that merely demand a lot of
 * everything: the same occupations ranked top for nearly every user. Rescaling compares
 * "which strengths this job leans on" with "which strengths this person leads with".
 */
function relativeRequirements(occupation: Occupation): Record<string, number> {
  const values = TRAITS.map((t) => occupation.requirements[t]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  return Object.fromEntries(
    TRAITS.map((t) => [
      t,
      max === min ? 50 : (100 * (occupation.requirements[t] - min)) / (max - min),
    ]),
  );
}

/**
 * Stage 2: 0-100 fit between the user's ranked strengths and the shape of what the occupation demands.
 * Geometric declining weights: the user's top strengths count most (weight RATIO^(rank-1)).
 */
export function scoreOccupation(
  ranks: Record<string, number>,
  occupation: Occupation,
): number {
  const required = relativeRequirements(occupation);
  let weighted = 0;
  let totalWeight = 0;
  for (const trait of TRAITS) {
    const rank = ranks[trait];
    if (!Number.isFinite(rank)) throw new Error(`No rank for strength "${trait}"`);
    const weight = WEIGHT_RATIO ** (rank - 1);
    const closeness = 1 - Math.abs(levelForRank(rank) - required[trait]) / 100;
    weighted += weight * closeness;
    totalWeight += weight;
  }
  return (100 * weighted) / totalWeight;
}

/** 1-5 stars by position in a list of `count`: the best fifth gets 5 stars, the last fifth 1 star. */
export function starsForRank(index: number, count: number): number {
  return STAR_LEVELS - Math.floor((index * STAR_LEVELS) / count);
}

/**
 * Full pipeline: interest shortlist -> strength score -> final top list, best first, with stars.
 * Equal scores share the stars of the first of them, so a tie never straddles a star boundary.
 */
export function matchCareers(
  ranks: Record<string, number>,
  userRiasec: Riasec,
  occupations: Occupation[],
): RatedOccupation[] {
  const top = shortlistByInterest(userRiasec, occupations)
    .map((occ) => ({ code: occ.code, title: occ.title, score: scoreOccupation(ranks, occ) }))
    .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title))
    .slice(0, FINAL_LIST_SIZE);

  const firstIndexOfScore = new Map<number, number>();
  top.forEach((s, i) => {
    if (!firstIndexOfScore.has(s.score)) firstIndexOfScore.set(s.score, i);
  });
  return top.map((s, i) => ({
    code: s.code,
    title: s.title,
    rank: i + 1,
    stars: starsForRank(firstIndexOfScore.get(s.score)!, top.length),
  }));
}
