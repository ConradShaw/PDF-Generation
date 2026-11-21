import { WEIGHT_RATIO } from '../careers/constants.js';
import { starsForRank } from '../careers/matching.js';
import type { Occupation } from '../careers/types.js';
import {
  categoryForRankNumber,
  TRAITS,
  type Category,
  type Trait,
} from '../scoring/traits.js';

/**
 * Role Fit: how well each internal candidate's ranked strengths match the SSM profile of a target
 * O*NET occupation. Definitions below follow the owner's template ("Role Fit template v1") and the
 * owner's decision of 2026-11-19 for the undefined "Wtd gap" (category distance x role importance).
 */

const CATEGORIES: readonly Category[] = ['Signature', 'Supporting', 'Stretch', 'Situational'];
const MAX_CATEGORY_DISTANCE = CATEGORIES.length - 1;
const ALIGNED_LIMIT = 3;
const MISALIGNED_LIMIT = 3;
/** Aligned / misaligned strengths are chosen from the candidate's strongest six. */
const CANDIDATE_FOCUS = 6;

export type RoleProfileRow = { rank: number; trait: Trait; category: Category };

export type Candidate = { id: string; name: string; ranks: Record<string, number> };

export type RatedCandidate = {
  id: string;
  name: string;
  rank: number;
  stars: number;
  aligned: Trait[];
  misaligned: Trait[];
};

const categoryIndex = (rank: number) => CATEGORIES.indexOf(categoryForRankNumber(rank));

/**
 * The target role as the 12 strengths ranked 1-12 by how much the occupation demands them.
 * Equal demand keeps the fixed SSM order (Array.sort is stable and TRAITS is already in that order),
 * so the profile is the same on every run.
 */
export function targetProfile(occupation: Occupation): RoleProfileRow[] {
  return [...TRAITS]
    .map((trait) => ({ trait, demand: occupation.requirements[trait] }))
    .sort((a, b) => b.demand - a.demand)
    .map(({ trait }, i) => ({
      rank: i + 1,
      trait,
      category: categoryForRankNumber(i + 1),
    }));
}

/**
 * "Wtd gap" for one strength: how many categories apart the candidate and the role place it (0-3),
 * weighted by how important the strength is to the role (0.8 per role rank, as in Career Matches).
 * Candidates above or below the role count the same: a surplus strength is also a mismatch.
 */
export function weightedGap(candidateRank: number, roleRank: number): number {
  const distance = Math.abs(categoryIndex(candidateRank) - categoryIndex(roleRank));
  return WEIGHT_RATIO ** (roleRank - 1) * distance;
}

function totalGap(ranks: Record<string, number>, profile: RoleProfileRow[]): number {
  return profile.reduce((sum, row) => sum + weightedGap(ranks[row.trait], row.rank), 0);
}

/** 0-100: 100 when every strength falls in the role's category; 0 when every one is as far off as possible. */
export function fitScore(ranks: Record<string, number>, profile: RoleProfileRow[]): number {
  const worst = profile.reduce(
    (sum, row) => sum + WEIGHT_RATIO ** (row.rank - 1) * MAX_CATEGORY_DISTANCE,
    0,
  );
  return 100 * (1 - totalGap(ranks, profile) / worst);
}

/**
 * Up to three strengths from the candidate's top six that sit in the role's category (aligned), and up to
 * three that do not (misaligned, largest weighted gap first). Aligned ones are listed most important to the role first.
 */
export function alignment(
  ranks: Record<string, number>,
  profile: RoleProfileRow[],
): { aligned: Trait[]; misaligned: Trait[] } {
  const roleRank = Object.fromEntries(profile.map((r) => [r.trait, r.rank])) as Record<Trait, number>;
  const topSix = [...TRAITS]
    .sort((a, b) => ranks[a] - ranks[b] || roleRank[a] - roleRank[b])
    .slice(0, CANDIDATE_FOCUS)
    .map((trait) => ({ trait, gap: weightedGap(ranks[trait], roleRank[trait]) }));

  const byImportance = (a: { trait: Trait }, b: { trait: Trait }) => roleRank[a.trait] - roleRank[b.trait];
  return {
    aligned: topSix.filter((t) => t.gap === 0).sort(byImportance).slice(0, ALIGNED_LIMIT).map((t) => t.trait),
    misaligned: topSix
      .filter((t) => t.gap > 0)
      .sort((a, b) => b.gap - a.gap)
      .slice(0, MISALIGNED_LIMIT)
      .map((t) => t.trait),
  };
}

/**
 * The Role-Fit Analysis table: candidates best fit first (ties by name), with stars in equal groups of the
 * candidates (best fifth = 5 stars). Equal fits share stars. The score itself is not exposed.
 */
export function rankCandidates(candidates: Candidate[], profile: RoleProfileRow[]): RatedCandidate[] {
  const scored = candidates
    .map((c) => ({ c, score: fitScore(c.ranks, profile) }))
    .sort((a, b) => b.score - a.score || a.c.name.localeCompare(b.c.name));

  const firstIndexOfScore = new Map<number, number>();
  scored.forEach((s, i) => {
    if (!firstIndexOfScore.has(s.score)) firstIndexOfScore.set(s.score, i);
  });

  return scored.map(({ c, score }, i) => ({
    id: c.id,
    name: c.name,
    rank: i + 1,
    stars: starsForRank(firstIndexOfScore.get(score)!, scored.length),
    ...alignment(c.ranks, profile),
  }));
}
