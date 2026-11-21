/**
 * Tunable constants for career matching. Owner assumptions to confirm are marked ASSUMPTION;
 * every value lives here so a change never touches the algorithm.
 */

/** ASSUMPTION: each step down the user's ranking weighs 80% of the one above (geometric decline). */
export const WEIGHT_RATIO = 0.8;

/**
 * Two-stage matching (owner's sheet): interest fit first, then strengths.
 * ASSUMPTION: the sheet asks for a "fairly generous" interest pre-filter of 100-200 occupations; 150.
 */
export const PRE_FILTER_SIZE = 150;

/** Owner spec: the final list shows the top 25 occupations. */
export const FINAL_LIST_SIZE = 25;

/**
 * Owner decision 2026-11-19: stars are equal groups of the final list (best fifth = 5 stars ... last
 * fifth = 1 star). Fixed bands of the fit score were tried and gave every customer 25 five-star results,
 * because the top 25 all score above 80.
 */
export const STAR_LEVELS = 5;

export const STRENGTH_COUNT = 12;

/** Holland codes, in the order the O*NET Interest Profiler reports them. */
export const RIASEC_AREAS = ['R', 'I', 'A', 'S', 'E', 'C'] as const;

/** O*NET Work Styles Impact runs -3..3; negative impact means the style is not wanted, so it counts as 0. */
export const WORK_STYLE_IMPACT_MAX = 3;
