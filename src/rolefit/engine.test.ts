import { describe, it, expect } from 'vitest';
import { TRAITS, type Trait } from '../scoring/traits.js';
import type { Occupation } from '../careers/types.js';
import { occupations } from '../careers/data.js';
import {
  FIT_LABELS,
  POSITION_WEIGHTS,
  QUINTILE_CUTOFFS,
  alignment,
  fitScore,
  quintileOf,
  rankCandidates,
  targetProfile,
} from './engine.js';

const occupation = (requirements: Partial<Record<Trait, number>>): Occupation => ({
  code: 'X',
  title: 'Test Role',
  riasec: { R: 4, I: 4, A: 4, S: 4, E: 4, C: 4 },
  requirements: Object.fromEntries(TRAITS.map((t) => [t, requirements[t] ?? 0])) as Record<Trait, number>,
});

/** A role with no ties whose most important strength is TRAITS[0] and least important TRAITS[11]. */
const ladderRole = () => targetProfile(occupation(Object.fromEntries(TRAITS.map((t, i) => [t, 100 - i * 5]))));
/** Candidate ranks from a best-first list of strengths (rank 1 = first in the list). */
const ranksOf = (order: readonly string[]) => Object.fromEntries(order.map((t, i) => [t, i + 1]));
/** Candidate ranks for the ladder role given as ranks in role order (index 0 = the role's most important strength). */
const inRoleOrder = (ranksByPosition: number[]) => Object.fromEntries(TRAITS.map((t, i) => [t, ranksByPosition[i]]));

const ceo = () => targetProfile(occupations.find((o) => o.code === '11-1011.00')!);
/** The three sample candidates from the client's workbook, ranks given in the sheet's column order for Chief Executives. */
const SHEET_ORDER = ['Discipline', 'Foresight', 'Confidence', 'Initiative', 'Tenacity', 'Practicality', 'Objectivity', 'Curiosity', 'Fairness', 'Discernment', 'Empathy', 'Prudence'];
const sheetCandidate = (ranks: number[]) => Object.fromEntries(SHEET_ORDER.map((t, i) => [t, ranks[i]]));
const JOE_17 = sheetCandidate([1, 2, 5, 12, 8, 10, 7, 3, 6, 4, 9, 11]);
const JOE_4 = sheetCandidate([2, 4, 8, 9, 10, 12, 5, 1, 11, 6, 7, 3]);
const JOE_20 = sheetCandidate([2, 7, 8, 11, 1, 10, 5, 6, 9, 3, 4, 12]);

describe('the client\'s fixed tables', () => {
  it('has 11 position weights summing to 100%, each 0.75x the one before (the workbook\'s Weights tab)', () => {
    expect(POSITION_WEIGHTS).toHaveLength(12);
    expect(POSITION_WEIGHTS.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
    for (let i = 1; i < 11; i++) expect(POSITION_WEIGHTS[i] / POSITION_WEIGHTS[i - 1]).toBeCloseTo(0.75, 2);
    expect(POSITION_WEIGHTS[0]).toBeCloseTo(0.26102, 5);
    expect(POSITION_WEIGHTS[11]).toBe(0); // the 12th position can never fall short
  });

  it('has four ascending quintile cut-offs (workbook: 30.353, 40.002, 48.617, 58.508 percent)', () => {
    expect(QUINTILE_CUTOFFS).toEqual([0.30353, 0.40002, 0.48617, 0.58508]);
  });
});

describe('targetProfile: the role as a ranked list of the 12 strengths', () => {
  it('ranks strengths 1-12 by how much the occupation demands them, most demanded first, with the position weights', () => {
    const profile = ladderRole();
    expect(profile.map((r) => r.trait)).toEqual([...TRAITS]);
    expect(profile.map((r) => r.rank)).toEqual(Array.from({ length: 12 }, (_, i) => i + 1));
    expect(profile.map((r) => r.position)).toEqual(Array.from({ length: 12 }, (_, i) => i + 1));
    expect(profile.map((r) => r.weight)).toEqual([...POSITION_WEIGHTS]);
  });

  it('puts three strengths in each category by position: Signature 1-3, Supporting 4-6, Stretch 7-9, Situational 10-12', () => {
    const cats = ladderRole().map((r) => r.category);
    expect(cats).toEqual([...Array(3).fill('Signature'), ...Array(3).fill('Supporting'), ...Array(3).fill('Stretch'), ...Array(3).fill('Situational')]);
  });

  it('gives equally demanded strengths the AVERAGE of their positions, and the next strength carries on after them (positions 2 and 3 tied: 1, 2.5, 2.5, 4)', () => {
    const profile = targetProfile(occupation({ [TRAITS[0]]: 90, [TRAITS[1]]: 80, [TRAITS[2]]: 80, [TRAITS[3]]: 70 }));
    expect(profile.slice(0, 4).map((r) => r.rank)).toEqual([1, 2.5, 2.5, 4]);
  });

  it('averages three tied strengths the way the client does: positions 7, 8, 9 tied all become 8, then 10, 11, 12 continue', () => {
    const demand: Partial<Record<Trait, number>> = {};
    TRAITS.forEach((t, i) => (demand[t] = i < 6 ? 100 - i * 5 : i < 9 ? 50 : 40 - (i - 9) * 5));
    const ranks = targetProfile(occupation(demand)).map((r) => r.rank);
    expect(ranks).toEqual([1, 2, 3, 4, 5, 6, 8, 8, 8, 10, 11, 12]);
  });

  it('gives tied strengths the AVERAGE of the weights of the positions they share, so a tie never favours one of them', () => {
    const profile = targetProfile(occupation({ [TRAITS[0]]: 90, [TRAITS[1]]: 80, [TRAITS[2]]: 80, [TRAITS[3]]: 70 }));
    const shared = (POSITION_WEIGHTS[1] + POSITION_WEIGHTS[2]) / 2;
    expect(profile[1].weight).toBeCloseTo(shared, 10);
    expect(profile[2].weight).toBeCloseTo(shared, 10);
    expect(profile[0].weight).toBe(POSITION_WEIGHTS[0]);
    expect(profile[3].weight).toBe(POSITION_WEIGHTS[3]);
  });

  it('keeps tied strengths in the fixed SSM order, so the profile never changes between runs', () => {
    const profile = targetProfile(occupation({ [TRAITS[5]]: 50, [TRAITS[2]]: 50, [TRAITS[9]]: 80 }));
    expect(profile.slice(0, 3).map((r) => r.trait)).toEqual([TRAITS[9], TRAITS[2], TRAITS[5]]);
    expect(profile[1].rank).toBe(profile[2].rank);
  });

  it('a tie across the Signature/Supporting boundary shares one rank but each strength keeps the category of its position', () => {
    const demand: Partial<Record<Trait, number>> = {};
    TRAITS.forEach((t, i) => (demand[t] = i < 2 ? 100 - i * 5 : i < 4 ? 80 : 60 - i * 3));
    const profile = targetProfile(occupation(demand));
    expect(profile[2].rank).toBe(3.5);
    expect(profile[3].rank).toBe(3.5);
    expect([profile[2].category, profile[3].category]).toEqual(['Signature', 'Supporting']);
  });

  it('reads the real Chief Executives data as the client\'s sheet does: Discipline first, Foresight and Confidence tied at 2.5, Prudence last', () => {
    const profile = ceo();
    expect(profile[0].trait).toBe('Discipline');
    expect(profile.filter((r) => r.rank === 2.5).map((r) => r.trait).sort()).toEqual(['Confidence', 'Foresight']);
    expect(profile[11].trait).toBe('Prudence');
    expect(profile.map((r) => r.rank)).toEqual([1, 2.5, 2.5, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });
});

describe('fitScore: the client\'s workbook formula', () => {
  it('reproduces the three sample candidates in the client\'s workbook for Chief Executives: 72.1%, 56.0% and 53.8%', () => {
    const profile = ceo();
    expect(fitScore(JOE_17, profile) * 100).toBeCloseTo(72.1, 1);
    expect(fitScore(JOE_4, profile) * 100).toBeCloseTo(56.0, 1);
    expect(fitScore(JOE_20, profile) * 100).toBeCloseTo(53.8, 1);
  });

  it('is exactly 1 for a candidate whose ranks equal the role\'s, and 0 for the completely reversed candidate of a role with no ties', () => {
    const profile = ladderRole();
    expect(fitScore(ranksOf(TRAITS), profile)).toBe(1);
    expect(fitScore(ranksOf([...TRAITS].reverse()), profile)).toBeCloseTo(0, 10);
  });

  it('only counts a strength the candidate ranks WORSE than the role: ranking one better than the role asks costs nothing', () => {
    const profile = ladderRole();
    // Swap positions 5 and 6: the position-5 strength drops one place (costs w5), the position-6 strength rises one (free).
    const swapped = inRoleOrder([1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12]);
    const worst = [11, 9, 7, 5, 3, 1].reduce((sum, gap, p) => sum + POSITION_WEIGHTS[p] * gap, 0);
    expect(fitScore(swapped, profile)).toBeCloseTo(1 - POSITION_WEIGHTS[4] / worst, 10);
  });

  it('costs more to fall short on the strengths the role relies on most: swapping the top two is worse than swapping positions 5 and 6', () => {
    const profile = ladderRole();
    const topSwap = fitScore(inRoleOrder([2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), profile);
    const midSwap = fitScore(inRoleOrder([1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12]), profile);
    expect(topSwap).toBeLessThan(midSwap);
  });

  it('gives a candidate the same score whichever of two tied role strengths they rank higher, because tied strengths weigh the same', () => {
    const profile = ceo();
    const a = { ...JOE_17, Foresight: 2, Confidence: 5 };
    const b = { ...JOE_17, Foresight: 5, Confidence: 2 };
    expect(fitScore(a, profile)).toBeCloseTo(fitScore(b, profile), 12);
  });

  it('accepts tied (fractional) candidate ranks and stays finite', () => {
    const tied = Object.fromEntries(TRAITS.map((t) => [t, 6.5]));
    const score = fitScore(tied, ladderRole());
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeLessThan(1);
  });

  it('can fall slightly below 0 for a very poor fit, which the lowest quintile absorbs', () => {
    const profile = ceo();
    const reversed = Object.fromEntries(profile.map((r) => [r.trait, 13 - r.position]));
    expect(fitScore(reversed, profile)).toBeLessThan(0.05);
    expect(quintileOf(fitScore(reversed, profile))).toBe(1);
  });
});

describe('quintileOf and the fit labels', () => {
  it('maps a score to a quintile with the fixed cut-offs, a score exactly on a cut-off belonging to the lower one', () => {
    expect(quintileOf(-0.05)).toBe(1);
    expect(quintileOf(0.30353)).toBe(1);
    expect(quintileOf(0.30354)).toBe(2);
    expect(quintileOf(0.40002)).toBe(2);
    expect(quintileOf(0.40003)).toBe(3);
    expect(quintileOf(0.48617)).toBe(3);
    expect(quintileOf(0.48618)).toBe(4);
    expect(quintileOf(0.58508)).toBe(4);
    expect(quintileOf(0.58509)).toBe(5);
    expect(quintileOf(1)).toBe(5);
  });

  it('puts the three sample candidates in the client\'s quintiles: 5th, 4th, 4th', () => {
    const profile = ceo();
    expect([JOE_17, JOE_4, JOE_20].map((c) => quintileOf(fitScore(c, profile)))).toEqual([5, 4, 4]);
  });

  it('names the quintiles one-to-one as the workbook does', () => {
    expect([...FIT_LABELS]).toEqual(['Very Weak Fit', 'Weak Fit', 'Satisfactory Fit', 'Strong Fit', 'Very Strong Fit']);
  });
});

describe('alignment: top-3 aligned and misaligned strengths from the candidate\'s top 6', () => {
  it('lists up to three aligned strengths (no shortfall), most important to the role first, for a candidate who matches the role', () => {
    const { aligned, misaligned } = alignment(ranksOf(TRAITS), ladderRole());
    expect(aligned).toEqual([TRAITS[0], TRAITS[1], TRAITS[2]]); // six qualify, three shown
    expect(misaligned).toEqual([]);
  });

  it('lists aligned strengths in the role\'s order, not in the candidate\'s own order', () => {
    const candidate = ranksOf([TRAITS[2], TRAITS[1], TRAITS[0], ...TRAITS.slice(3)]);
    // The candidate ranks TRAITS[2] first (better than the role's 3: free), TRAITS[0] third (worse than the role's 1: a shortfall).
    expect(alignment(candidate, ladderRole()).aligned).toEqual([TRAITS[1], TRAITS[2], TRAITS[3]]);
  });

  it('counts a strength the candidate ranks better than the role as aligned, and one ranked worse as misaligned, largest weighted shortfall first', () => {
    // The role\'s least important strength is ranked first; everyone else slides down one place.
    const candidate = ranksOf([TRAITS[11], ...TRAITS.slice(0, 11)]);
    const { aligned, misaligned } = alignment(candidate, ladderRole());
    expect(aligned).toEqual([TRAITS[11]]);
    expect(misaligned).toEqual([TRAITS[0], TRAITS[1], TRAITS[2]]); // each one place short; role position 1 weighs most
  });

  it('only looks at the candidate\'s top 6: a strength outside it is never listed', () => {
    const { aligned, misaligned } = alignment(ranksOf(TRAITS), ladderRole());
    for (const t of TRAITS.slice(6)) expect([...aligned, ...misaligned]).not.toContain(t);
  });

  it('breaks a tie in shortfall by the role order (left-most wins), as the template specifies', () => {
    const profile = targetProfile(occupation({ [TRAITS[0]]: 90, [TRAITS[1]]: 90, ...Object.fromEntries(TRAITS.slice(2).map((t, i) => [t, 80 - i * 5])) }));
    // The two tied top strengths (shared weight and rank 1.5) are both ranked 4 by the candidate: equal shortfall.
    const candidate = { ...Object.fromEntries(TRAITS.map((t, i) => [t, i + 5])), [TRAITS[0]]: 4, [TRAITS[1]]: 4 };
    const { misaligned } = alignment({ ...candidate, [TRAITS[2]]: 1, [TRAITS[3]]: 2, [TRAITS[4]]: 3, [TRAITS[5]]: 5 }, profile);
    expect(misaligned.slice(0, 2)).toEqual([TRAITS[0], TRAITS[1]]);
  });

  it('lists misaligned strengths largest weighted shortfall first, even when the candidate ranks a smaller shortfall higher', () => {
    // Role order = TRAITS. The candidate's top six by rank is: TRAITS[1], TRAITS[3], TRAITS[4], TRAITS[2] (one place short),
    // TRAITS[5], TRAITS[0] (five places short, the role's most important strength). Candidate order differs from gap order.
    const candidate = inRoleOrder([6, 1, 4, 2, 3, 5, 7, 8, 9, 10, 11, 12]);
    const { aligned, misaligned } = alignment(candidate, ladderRole());
    expect(misaligned).toEqual([TRAITS[0], TRAITS[2]]);
    expect(aligned).toEqual([TRAITS[1], TRAITS[3], TRAITS[4]]);
  });

  it('breaks an exact tie in weighted shortfall by the role\'s order, whatever order the candidate ranks them in', () => {
    const profile = ladderRole().map((row) => ({ ...row }));
    profile[0].weight = 0.25; // role position 1: shortfall 3 -> 0.75
    profile[1].weight = 0.75; // role position 2: shortfall 1 -> 0.75
    const candidate = inRoleOrder([4, 3, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]); // position 2's strength is ranked higher by the candidate
    expect(alignment(candidate, profile).misaligned).toEqual([TRAITS[0], TRAITS[1]]);
  });

  it('when a candidate ties all strengths, picks the top six in the role\'s order even where that differs from the SSM order (real Chief Executives data)', () => {
    const tied = Object.fromEntries(TRAITS.map((t) => [t, 6.5]));
    const { aligned, misaligned } = alignment(tied, ceo());
    expect(aligned).toEqual([]);
    // Discipline (5.5 places short at the heaviest weight), then the tied Confidence and Foresight with equal shortfall in role order.
    expect(misaligned).toEqual(['Discipline', 'Confidence', 'Foresight']);
  });

  it('never lists more than 3 in either list, and a candidate with every strength tied is judged from the role\'s most important six', () => {
    const tied = Object.fromEntries(TRAITS.map((t) => [t, 6.5]));
    const { aligned, misaligned } = alignment(tied, ladderRole());
    expect(aligned).toEqual([]);
    expect(misaligned).toEqual([TRAITS[0], TRAITS[1], TRAITS[2]]);
  });
});

describe('rankCandidates: the Role-Fit Analysis table', () => {
  it('ranks candidates best fit first, numbering them 1..n, and gives score, stars and label', () => {
    const rated = rankCandidates(
      [
        { id: 'a', name: 'Joe 20', ranks: JOE_20 },
        { id: 'b', name: 'Joe 17', ranks: JOE_17 },
        { id: 'c', name: 'Joe 4', ranks: JOE_4 },
      ],
      ceo(),
    );
    expect(rated.map((r) => r.name)).toEqual(['Joe 17', 'Joe 4', 'Joe 20']);
    expect(rated.map((r) => r.rank)).toEqual([1, 2, 3]);
    expect(rated.map((r) => r.score.toFixed(1))).toEqual(['72.1', '56.0', '53.8']);
    expect(rated.map((r) => r.stars)).toEqual([5, 4, 4]);
    expect(rated.map((r) => r.fitLabel)).toEqual(['Very Strong Fit', 'Strong Fit', 'Strong Fit']);
  });

  it('gives stars from fixed cut-offs, not from the other candidates: a lone perfect candidate has 5 stars and a lone reversed one has 1', () => {
    const profile = ladderRole();
    const [perfect] = rankCandidates([{ id: '1', name: 'Perfect', ranks: ranksOf(TRAITS) }], profile);
    const [worst] = rankCandidates([{ id: '2', name: 'Worst', ranks: ranksOf([...TRAITS].reverse()) }], profile);
    expect(perfect.stars).toBe(5);
    expect(perfect.fitLabel).toBe('Very Strong Fit');
    expect(worst.stars).toBe(1);
    expect(worst.fitLabel).toBe('Very Weak Fit');
  });

  it('gives a candidate the same stars whoever else is in the report', () => {
    const profile = ceo();
    const alone = rankCandidates([{ id: 'a', name: 'Joe 4', ranks: JOE_4 }], profile)[0];
    const crowd = rankCandidates(
      [
        { id: 'a', name: 'Joe 4', ranks: JOE_4 },
        { id: 'b', name: 'Joe 17', ranks: JOE_17 },
        { id: 'c', name: 'Joe 20', ranks: JOE_20 },
      ],
      profile,
    ).find((r) => r.id === 'a')!;
    expect(crowd.stars).toBe(alone.stars);
    expect(crowd.score).toBeCloseTo(alone.score, 10);
  });

  it('orders candidates with identical scores by name', () => {
    const profile = ladderRole();
    const same = ranksOf(TRAITS);
    const rated = rankCandidates(
      [
        { id: '1', name: 'Zed', ranks: same },
        { id: '2', name: 'Amy', ranks: same },
      ],
      profile,
    );
    expect(rated.map((r) => r.name)).toEqual(['Amy', 'Zed']);
    expect(rated[0].stars).toBe(rated[1].stars);
  });

  it('attaches each candidate\'s aligned and misaligned strengths', () => {
    const [joe] = rankCandidates([{ id: 'a', name: 'Joe 17', ranks: JOE_17 }], ceo());
    // Joe 17's top six is Discipline, Foresight, Curiosity, Discernment, Confidence, Fairness. Only Confidence (ranked 5th,
    // the role asks for 2.5) falls short; the other five are ranked at or better than the role asks, three shown in role order.
    expect(joe.aligned).toEqual(['Discipline', 'Foresight', 'Curiosity']);
    expect(joe.misaligned).toEqual(['Confidence']);
  });

  it('handles a single candidate and an empty list', () => {
    expect(rankCandidates([], ladderRole())).toEqual([]);
    expect(rankCandidates([{ id: '1', name: 'Only', ranks: ranksOf(TRAITS) }], ladderRole())).toHaveLength(1);
  });
});
