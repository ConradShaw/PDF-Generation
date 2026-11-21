import { describe, it, expect } from 'vitest';
import { TRAITS, type Trait } from '../scoring/traits.js';
import {
  riasecFit,
  shortlistByInterest,
  scoreOccupation,
  starsForRank,
  matchCareers,
} from './matching.js';
import type { Occupation, Riasec } from './types.js';

/** Strict ladder: TRAITS[0] is rank 1, TRAITS[11] is rank 12. */
const ladderRanks = (): Record<string, number> =>
  Object.fromEntries(TRAITS.map((t, i) => [t, i + 1]));

const flatRequirements = (value: number): Record<Trait, number> =>
  Object.fromEntries(TRAITS.map((t) => [t, value])) as Record<Trait, number>;

/** A different requirement profile (shape) for each seed. */
const shapedRequirements = (seed: number): Record<Trait, number> =>
  Object.fromEntries(TRAITS.map((t, j) => [t, ((seed + 1) * (j + 3) * 37) % 101])) as Record<Trait, number>;

function occ(
  code: string,
  riasec: Partial<Riasec>,
  requirements: Record<Trait, number> = flatRequirements(50),
): Occupation {
  return {
    code,
    title: code,
    riasec: { R: 4, I: 4, A: 4, S: 4, E: 4, C: 4, ...riasec },
    requirements,
  };
}

const investigativeUser: Riasec = { R: 5, I: 25, A: 10, S: 10, E: 10, C: 5 };

describe('riasecFit (correlation, -1 to 1)', () => {
  it('is +1 for the same shape and -1 for the opposite shape', () => {
    const a: Riasec = { R: 1, I: 2, A: 3, S: 4, E: 5, C: 6 };
    expect(riasecFit(a, { R: 2, I: 4, A: 6, S: 8, E: 10, C: 12 })).toBeCloseTo(1, 9);
    expect(riasecFit(a, { R: 6, I: 5, A: 4, S: 3, E: 2, C: 1 })).toBeCloseTo(-1, 9);
  });

  it('is positive when the occupation peaks where the user peaks, negative where the user is weakest', () => {
    expect(riasecFit(investigativeUser, occ('sci', { I: 7, R: 1 }).riasec)).toBeGreaterThan(0);
    expect(riasecFit(investigativeUser, occ('mech', { R: 7, I: 1 }).riasec)).toBeLessThan(0);
  });

  it('is scale-free: the same preference gives the same fit whether answers are scored 0-25 or 0-100', () => {
    const scaled: Riasec = { R: 20, I: 100, A: 40, S: 40, E: 40, C: 20 };
    const o = occ('sci', { I: 7, R: 1 }).riasec;
    expect(riasecFit(scaled, o)).toBeCloseTo(riasecFit(investigativeUser, o), 9);
  });

  it('is 0, not NaN, when either profile has no variation', () => {
    const flat: Riasec = { R: 15, I: 15, A: 15, S: 15, E: 15, C: 15 };
    expect(riasecFit(flat, occ('x', { I: 7, R: 1 }).riasec)).toBe(0);
    expect(riasecFit(investigativeUser, occ('flat', {}).riasec)).toBe(0);
  });
});

describe('shortlistByInterest', () => {
  const candidates = [
    occ('best', { I: 7, R: 1 }),
    occ('good', { I: 6, S: 5, R: 2 }),
    occ('opposite', { R: 7, I: 1 }),
    occ('flat', {}),
  ];

  it('keeps the occupations whose interests fit best, up to the shortlist size', () => {
    const kept = shortlistByInterest(investigativeUser, candidates, 1);
    expect(kept.map((o) => o.code)).toEqual(['best']);
  });

  it('finds the best fit wherever it appears in the input list', () => {
    const reversed = [...candidates].reverse(); // best now comes last
    expect(shortlistByInterest(investigativeUser, reversed, 1).map((o) => o.code)).toEqual(['best']);
  });

  it('orders the shortlist best fit first, and breaks exact ties by occupation code so results never depend on input order', () => {
    const tiedB = occ('b-code', { I: 7, R: 1 });
    const tiedA = occ('a-code', { I: 7, R: 1 });
    const [best, good] = candidates;
    // best, tiedA and tiedB fit identically; 'good' fits less well. Input order is deliberately scrambled.
    const ordered = shortlistByInterest(investigativeUser, [tiedB, good, best, tiedA], 10).map((o) => o.code);
    expect(ordered).toEqual(['a-code', 'b-code', 'best', 'good']);
  });

  it('never includes an occupation that fits the user\'s interests negatively or not at all, even when the shortlist has room', () => {
    const kept = shortlistByInterest(investigativeUser, candidates, 100);
    expect(kept.map((o) => o.code).sort()).toEqual(['best', 'good']);
  });

  it('keeps every occupation for a user with no interest preference, instead of silently producing an empty report', () => {
    // Answering "Unsure" to all 30 questions gives six identical scores.
    const flat: Riasec = { R: 15, I: 15, A: 15, S: 15, E: 15, C: 15 };
    expect(shortlistByInterest(flat, candidates, 1)).toHaveLength(candidates.length);
  });
});

describe('scoreOccupation', () => {
  it('weights the strongest strengths most: matching rank 1 beats matching rank 12', () => {
    const ranks = ladderRanks();
    const demandsTop = { ...flatRequirements(0), [TRAITS[0]]: 100 } as Record<Trait, number>;
    const demandsBottom = { ...flatRequirements(0), [TRAITS[11]]: 100 } as Record<Trait, number>;
    expect(scoreOccupation(ranks, occ('a', {}, demandsTop))).toBeGreaterThan(scoreOccupation(ranks, occ('b', {}, demandsBottom)));
  });

  it('stays within 0-100', () => {
    const ranks = ladderRanks();
    for (const v of [0, 50, 100]) {
      const s = scoreOccupation(ranks, occ('x', {}, flatRequirements(v)));
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(100);
    }
  });

  it('scores a perfect profile match at 100', () => {
    const ranks = ladderRanks();
    const perfect = Object.fromEntries(TRAITS.map((t, i) => [t, (100 * (12 - (i + 1))) / 11])) as Record<Trait, number>;
    expect(scoreOccupation(ranks, occ('p', {}, perfect))).toBeCloseTo(100, 6);
  });

  it('compares the shape of the demands, not their absolute level: an occupation that demands more of everything scores the same', () => {
    const ranks = ladderRanks();
    const shape = TRAITS.map((_, i) => (100 * (11 - i)) / 11);
    const strict = Object.fromEntries(TRAITS.map((t, i) => [t, shape[i]])) as Record<Trait, number>;
    const demandingEverything = Object.fromEntries(TRAITS.map((t, i) => [t, 50 + shape[i] / 2])) as Record<Trait, number>;
    expect(scoreOccupation(ranks, occ('a', {}, demandingEverything))).toBeCloseTo(scoreOccupation(ranks, occ('b', {}, strict)), 6);
  });

  it('rates a perfect shape match above an occupation with no preference between strengths, and that above an opposite shape', () => {
    const ranks = ladderRanks();
    const levels = TRAITS.map((_, i) => (100 * (11 - i)) / 11);
    const perfect = Object.fromEntries(TRAITS.map((t, i) => [t, levels[i]])) as Record<Trait, number>;
    const opposite = Object.fromEntries(TRAITS.map((t, i) => [t, levels[11 - i]])) as Record<Trait, number>;
    const flat = scoreOccupation(ranks, occ('flat', {}, flatRequirements(80)));
    expect(scoreOccupation(ranks, occ('p', {}, perfect))).toBeGreaterThan(flat);
    expect(flat).toBeGreaterThan(scoreOccupation(ranks, occ('o', {}, opposite)));
  });

  it('scores an occupation with no differences between strengths the same for every user, so it cannot dominate every list', () => {
    const flat = occ('flat', {}, flatRequirements(80));
    const reversed = Object.fromEntries(TRAITS.map((t, i) => [t, 12 - i]));
    expect(scoreOccupation(ladderRanks(), flat)).toBeCloseTo(scoreOccupation(reversed, flat), 6);
  });

  it('fails loudly when a strength has no rank, instead of silently scoring NaN and tying every occupation', () => {
    const { [TRAITS[3]]: _omitted, ...incomplete } = ladderRanks();
    expect(() => scoreOccupation(incomplete, occ('x', {}))).toThrow(new RegExp(TRAITS[3]));
  });

  it('handles tied (fractional) ranks without producing NaN', () => {
    const ranks = { ...ladderRanks(), [TRAITS[0]]: 1.5, [TRAITS[1]]: 1.5 };
    expect(Number.isFinite(scoreOccupation(ranks, occ('t', {})))).toBe(true);
  });
});

describe('starsForRank (equal groups of the list)', () => {
  it('gives each star level an equal share of a list of 25: five results each, best first', () => {
    const stars = Array.from({ length: 25 }, (_, i) => starsForRank(i, 25));
    expect(stars.slice(0, 5)).toEqual([5, 5, 5, 5, 5]);
    expect(stars.slice(5, 10)).toEqual([4, 4, 4, 4, 4]);
    expect(stars.slice(10, 15)).toEqual([3, 3, 3, 3, 3]);
    expect(stars.slice(15, 20)).toEqual([2, 2, 2, 2, 2]);
    expect(stars.slice(20)).toEqual([1, 1, 1, 1, 1]);
  });

  it('keeps stars between 1 and 5, never rising down the list, for any list size', () => {
    for (let n = 1; n <= 30; n++) {
      const stars = Array.from({ length: n }, (_, i) => starsForRank(i, n));
      expect(Math.min(...stars)).toBeGreaterThanOrEqual(1);
      expect(Math.max(...stars)).toBeLessThanOrEqual(5);
      expect(stars).toEqual([...stars].sort((a, b) => b - a));
    }
  });

  it('gives the best result 5 stars even in a list of one', () => {
    expect(starsForRank(0, 1)).toBe(5);
  });
});

describe('matchCareers', () => {
  const science = Array.from({ length: 60 }, (_, i) => occ(`sci${i}`, { I: 7, R: 1 }, shapedRequirements(i)));
  const trades = Array.from({ length: 60 }, (_, i) => occ(`trade${i}`, { R: 7, I: 1 }, shapedRequirements(i)));

  it('shortlists by interest, re-ranks by strengths, and returns the final 25 with ranks 1..25', () => {
    const result = matchCareers(ladderRanks(), investigativeUser, [...science, ...trades]);
    expect(result).toHaveLength(25);
    expect(result.every((r) => r.code.startsWith('sci'))).toBe(true);
    expect(result.map((r) => r.rank)).toEqual(Array.from({ length: 25 }, (_, i) => i + 1));
  });

  it('orders the final list best fit first, with five results at each star level so the stars always tell results apart', () => {
    const result = matchCareers(ladderRanks(), investigativeUser, science);
    const stars = result.map((r) => r.stars);
    expect(stars).toEqual([...stars].sort((a, b) => b - a));
    for (const level of [5, 4, 3, 2, 1]) expect(stars.filter((s) => s === level)).toHaveLength(5);
  });

  it('gives occupations with an identical fit the same stars, so a tie is never split across a star boundary', () => {
    const same = Array.from({ length: 30 }, (_, i) => occ(`same${i}`, { I: 7, R: 1 }, shapedRequirements(3)));
    const stars = new Set(matchCareers(ladderRanks(), investigativeUser, same).map((r) => r.stars));
    expect(stars).toEqual(new Set([5]));
  });

  it('considers only the best-interest shortlist (150, per the spec; change this test if the spec changes): an occupation outside it cannot win on strengths alone', () => {
    const perfect = Object.fromEntries(TRAITS.map((t, i) => [t, (100 * (11 - i)) / 11])) as Record<Trait, number>;
    const strongInterest = Array.from({ length: 150 }, (_, i) => occ(`strong${i}`, { I: 7, R: 1 }, shapedRequirements(i)));
    const weakerInterest = Array.from({ length: 30 }, (_, i) => occ(`weak${i}`, { I: 5, R: 3 }, perfect));
    const result = matchCareers(ladderRanks(), investigativeUser, [...strongInterest, ...weakerInterest]);
    expect(result.some((r) => r.code.startsWith('weak'))).toBe(false);
  });

  it('spreads the stars over a shorter list too: five results get 5, 4, 3, 2 and 1 stars, not five stars each', () => {
    const five = Array.from({ length: 5 }, (_, i) => occ(`sci${i}`, { I: 7, R: 1 }, shapedRequirements(i)));
    const stars = matchCareers(ladderRanks(), investigativeUser, five).map((r) => r.stars);
    expect(stars).toEqual([5, 4, 3, 2, 1]);
  });

  it('returns fewer than 25 when fewer occupations pass the interest shortlist', () => {
    expect(matchCareers(ladderRanks(), investigativeUser, [...science.slice(0, 5), ...trades])).toHaveLength(5);
  });

  it('does not expose the underlying score to templates', () => {
    const [first] = matchCareers(ladderRanks(), investigativeUser, science);
    expect(first).not.toHaveProperty('score');
  });
});
