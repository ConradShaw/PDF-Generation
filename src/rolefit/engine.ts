import type { Occupation } from '../careers/types.js';
import { categoryForRankNumber, TRAITS, type Category, type Trait } from '../scoring/traits.js';

/**
 * Role Fit: how well each candidate's ranked strengths match the SSM profile of a target O*NET occupation.
 *
 * This follows the client's workbook ("Role mapping - v0 5 - FULL CALC") as explained in their video of
 * 24 November 2026 and shown on its Weights, Quintiles and BT Lead tabs:
 *
 *  1. The role's 12 mapped work-style values are ranked 1-12 (1 = highest). Equal values share the average
 *     of the positions they occupy (positions 2 and 3 tied -> both 2.5) and share the average of those
 *     positions' weights.
 *  2. Per strength, the shortfall is how much WORSE the candidate ranks it than the role does:
 *     max(0, candidate rank - role rank). Ranking a strength better than the role asks costs nothing.
 *  3. Each shortfall is multiplied by the weight of the strength's position in the role (26.1% for the most
 *     important, then 0.75x per step, 11 weights summing to 100%).
 *  4. Score = 1 - (the candidate's weighted shortfall / the largest weighted shortfall possible), where the
 *     largest gap for position p is 13 - 2p (a fully reversed profile). It is a fraction, shown as a
 *     percentage, and can fall slightly below 0.
 *  5. The score maps to a quintile (stars) by fixed cut-offs: the exact 20th/40th/60th/80th percentiles of
 *     the score over all 12! = 479,001,600 possible candidate profiles.
 */

/** Weights by role position 1-11, from the workbook's Weights tab (position 12 can never fall short). */
export const POSITION_WEIGHTS: readonly number[] = [26.102, 19.577, 14.683, 11.012, 8.259, 6.194, 4.646, 3.484, 2.613, 1.96, 1.47, 0].map(
  (w) => w / 100,
);

/** Exact 20th, 40th, 60th and 80th percentile fit scores (workbook's Quintiles tab). */
export const QUINTILE_CUTOFFS: readonly number[] = [0.30353, 0.40002, 0.48617, 0.58508];

/** The workbook's "job-fit category" for quintile 1 to 5, a one-to-one mapping. */
export const FIT_LABELS = ['Very Weak Fit', 'Weak Fit', 'Satisfactory Fit', 'Strong Fit', 'Very Strong Fit'] as const;
export type FitLabel = (typeof FIT_LABELS)[number];

const ALIGNED_LIMIT = 3;
const MISALIGNED_LIMIT = 3;
/** Aligned / misaligned strengths are chosen from the candidate's strongest six. */
const CANDIDATE_FOCUS = 6;

/**
 * One strength in the role, in role order. `rank` is the (possibly shared, fractional) rank, `position` the
 * place in the list 1-12 (ties keep the fixed SSM order), `category` follows the position, and `weight` is the
 * weight of the position, averaged across tied positions.
 */
export type RoleProfileRow = { rank: number; position: number; trait: Trait; category: Category; weight: number };

export type Candidate = { id: string; name: string; ranks: Record<string, number> };

export type RatedCandidate = {
  id: string;
  name: string;
  rank: number;
  /** Fit score as a percentage, e.g. 72.1. Can be slightly below 0. */
  score: number;
  /** Quintile 1-5, shown as stars. */
  stars: number;
  fitLabel: FitLabel;
  aligned: Trait[];
  misaligned: Trait[];
};

const average = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

/**
 * The target role as the 12 strengths ranked by how much the occupation demands them. Equal demand shares the
 * average rank and the average weight; the display order of tied strengths stays the fixed SSM order (Array.sort
 * is stable and TRAITS is already in that order), so the profile is the same on every run.
 */
export function targetProfile(occupation: Occupation): RoleProfileRow[] {
  const sorted = [...TRAITS]
    .map((trait) => ({ trait, demand: occupation.requirements[trait] }))
    .sort((a, b) => b.demand - a.demand);

  const rows: RoleProfileRow[] = [];
  for (let start = 0; start < sorted.length; ) {
    let end = start;
    while (end + 1 < sorted.length && sorted[end + 1].demand === sorted[start].demand) end++;
    const positions = Array.from({ length: end - start + 1 }, (_, k) => start + k + 1);
    const rank = average(positions);
    const weight = average(positions.map((p) => POSITION_WEIGHTS[p - 1]));
    for (const position of positions) {
      rows.push({ rank, position, trait: sorted[position - 1].trait, category: categoryForRankNumber(position), weight });
    }
    start = end + 1;
  }
  return rows;
}

/** Biggest possible gap at a position: the role's rank against a fully reversed candidate (11, 9, 7, 5, 3, 1, 0...). */
const maxGap = (position: number) => Math.max(0, 13 - 2 * position);

/** Weighted shortfall of one strength: only a candidate who ranks it worse than the role counts. */
const weightedShortfall = (row: RoleProfileRow, candidateRank: number) => row.weight * Math.max(0, candidateRank - row.rank);

/** 0-1 (a fraction, can be slightly negative): 1 when the candidate ranks no strength worse than the role. */
export function fitScore(ranks: Record<string, number>, profile: RoleProfileRow[]): number {
  const worst = profile.reduce((sum, row) => sum + row.weight * maxGap(row.position), 0);
  const shortfall = profile.reduce((sum, row) => sum + weightedShortfall(row, ranks[row.trait]), 0);
  return 1 - shortfall / worst;
}

/** Quintile 1-5 from the fixed cut-offs. A score exactly on a cut-off belongs to the lower quintile, as in the workbook. */
export function quintileOf(score: number): number {
  return 1 + QUINTILE_CUTOFFS.filter((cutoff) => score > cutoff).length;
}

/**
 * Up to three strengths from the candidate's top six with no shortfall (aligned, most important to the role
 * first), and up to three with a shortfall (misaligned, largest weighted shortfall first). Equal values go to
 * the strength more important to the role, as the template specifies ("left-most by column order").
 */
export function alignment(
  ranks: Record<string, number>,
  profile: RoleProfileRow[],
): { aligned: Trait[]; misaligned: Trait[] } {
  const byTrait = new Map(profile.map((row) => [row.trait, row]));
  const topSix = [...TRAITS]
    .sort((a, b) => ranks[a] - ranks[b] || byTrait.get(a)!.position - byTrait.get(b)!.position)
    .slice(0, CANDIDATE_FOCUS)
    .map((trait) => ({ trait, position: byTrait.get(trait)!.position, gap: weightedShortfall(byTrait.get(trait)!, ranks[trait]) }));

  return {
    aligned: topSix
      .filter((t) => t.gap === 0)
      .sort((a, b) => a.position - b.position)
      .slice(0, ALIGNED_LIMIT)
      .map((t) => t.trait),
    misaligned: topSix
      .filter((t) => t.gap > 0)
      .sort((a, b) => b.gap - a.gap || a.position - b.position)
      .slice(0, MISALIGNED_LIMIT)
      .map((t) => t.trait),
  };
}

/** The Role-Fit Analysis table: candidates best fit first (ties by name), each with score, stars and strengths. */
export function rankCandidates(candidates: Candidate[], profile: RoleProfileRow[]): RatedCandidate[] {
  return candidates
    .map((c) => ({ c, score: fitScore(c.ranks, profile) }))
    .sort((a, b) => b.score - a.score || a.c.name.localeCompare(b.c.name))
    .map(({ c, score }, i) => {
      const quintile = quintileOf(score);
      return {
        id: c.id,
        name: c.name,
        rank: i + 1,
        score: score * 100,
        stars: quintile,
        fitLabel: FIT_LABELS[quintile - 1],
        ...alignment(c.ranks, profile),
      };
    });
}
