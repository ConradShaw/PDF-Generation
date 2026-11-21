import { describe, it, expect } from 'vitest';
import { TRAITS, categoryForRankNumber, type Trait } from '../scoring/traits.js';
import type { Occupation } from '../careers/types.js';
import {
  targetProfile,
  weightedGap,
  fitScore,
  alignment,
  rankCandidates,
  type Candidate,
} from './engine.js';

const occupation = (requirements: Partial<Record<Trait, number>>): Occupation => ({
  code: 'X',
  title: 'Test Role',
  riasec: { R: 4, I: 4, A: 4, S: 4, E: 4, C: 4 },
  requirements: Object.fromEntries(TRAITS.map((t) => [t, requirements[t] ?? 0])) as Record<Trait, number>,
});

/** The role's profile when TRAITS[0] is its most important strength, TRAITS[11] its least. */
const ladderRole = () => targetProfile(occupation(Object.fromEntries(TRAITS.map((t, i) => [t, 100 - i * 5]))));

/** Candidate ranks from a best-first list of strengths. */
const ranksOf = (order: readonly string[]) => Object.fromEntries(order.map((t, i) => [t, i + 1]));
const sameAsRole = ranksOf(TRAITS);
const oppositeOfRole = ranksOf([...TRAITS].reverse());

describe('targetProfile: the role as a ranked list of the 12 strengths', () => {
  it('ranks strengths 1-12 by how much the occupation demands them, most demanded first', () => {
    const profile = ladderRole();
    expect(profile.map((r) => r.trait)).toEqual([...TRAITS]);
    expect(profile.map((r) => r.rank)).toEqual(Array.from({ length: 12 }, (_, i) => i + 1));
  });

  it('puts three strengths in each category: Signature 1-3, Supporting 4-6, Stretch 7-9, Situational 10-12', () => {
    const cats = ladderRole().map((r) => r.category);
    expect(cats).toEqual([...Array(3).fill('Signature'), ...Array(3).fill('Supporting'), ...Array(3).fill('Stretch'), ...Array(3).fill('Situational')]);
  });

  it('breaks ties between equally demanded strengths in the fixed SSM order, so the profile never changes between runs', () => {
    const flat = targetProfile(occupation({}));
    expect(flat.map((r) => r.trait)).toEqual([...TRAITS]);
  });

  it('ranks a real occupation sensibly: a Chief Executive leans on Discipline, not Prudence', async () => {
    const { occupations } = await import('../careers/data.js');
    const ceo = targetProfile(occupations.find((o) => o.title === 'Chief Executives')!);
    expect(ceo[0].trait).toBe('Discipline');
    expect(ceo.findIndex((r) => r.trait === 'Prudence')).toBeGreaterThan(8);
  });
});

describe('weightedGap: how far a candidate is from the role on one strength', () => {
  it('is 0 when the candidate and the role put the strength in the same category, even at different ranks', () => {
    expect(weightedGap(1, 3)).toBe(0); // both Signature
    expect(weightedGap(4, 6)).toBe(0); // both Supporting
    expect(weightedGap(12, 10)).toBe(0); // both Situational
  });

  it('grows with the number of categories apart: 1, 2 or 3 steps (role rank 1 has weight 1)', () => {
    expect(weightedGap(4, 1)).toBeCloseTo(1);
    expect(weightedGap(7, 1)).toBeCloseTo(2);
    expect(weightedGap(12, 1)).toBeCloseTo(3);
  });

  it('counts a candidate who rates a strength HIGHER than the role the same as one who rates it LOWER, at equal distance', () => {
    // role rank 4 is Supporting: candidate rank 1 (Signature) and rank 7 (Stretch) are each one step away
    expect(weightedGap(1, 4)).toBeCloseTo(weightedGap(7, 4));
    expect(weightedGap(1, 4)).toBeGreaterThan(0);
  });

  it('weights a mismatch on a strength the role relies on more than one on a peripheral strength: 0.8 per role rank', () => {
    // role ranks 1 and 2 are both Signature; a candidate rank 7 (Stretch) is two steps from both
    expect(weightedGap(7, 2) / weightedGap(7, 1)).toBeCloseTo(0.8, 6);
    expect(weightedGap(12, 1)).toBeGreaterThan(weightedGap(1, 12)); // same 3 steps, but rank 1 matters far more
  });

  it('uses the category of a tied (fractional) candidate rank: 3.5 sits in Supporting', () => {
    expect(weightedGap(3.5, 1)).toBeCloseTo(weightedGap(4, 1));
    expect(weightedGap(3.5, 1)).toBeGreaterThan(weightedGap(3, 1));
  });
});

describe('fitScore: 0-100 for a candidate against the role', () => {
  it('is 100 for a candidate whose strengths fall in exactly the role\'s categories', () => {
    expect(fitScore(sameAsRole, ladderRole())).toBe(100);
  });

  it('is lowest for the exact opposite of the role', () => {
    const opposite = fitScore(oppositeOfRole, ladderRole());
    expect(opposite).toBeLessThan(40);
    expect(opposite).toBeGreaterThanOrEqual(0);
  });

  it('falls as a candidate moves away from the role', () => {
    const swapSignatureWithSituational = ranksOf([TRAITS[9], ...TRAITS.slice(1, 9), TRAITS[0], TRAITS[10], TRAITS[11]]);
    expect(fitScore(swapSignatureWithSituational, ladderRole())).toBeLessThan(fitScore(sameAsRole, ladderRole()));
  });

  it('costs more to mismatch the strengths the role relies on than peripheral ones, for the same size of mismatch', () => {
    // Each candidate swaps one neighbouring pair across a category boundary (one step apart each).
    const swapAtTop = ranksOf([TRAITS[0], TRAITS[1], TRAITS[3], TRAITS[2], ...TRAITS.slice(4)]); // role ranks 3 and 4
    const swapAtBottom = ranksOf([...TRAITS.slice(0, 8), TRAITS[9], TRAITS[8], ...TRAITS.slice(10)]); // role ranks 9 and 10
    expect(fitScore(swapAtTop, ladderRole())).toBeLessThan(fitScore(swapAtBottom, ladderRole()));
    expect(fitScore(swapAtBottom, ladderRole())).toBeLessThan(100);
  });

  it('is unchanged by reshuffling strengths within a category: only categories count', () => {
    expect(fitScore(ranksOf([TRAITS[2], TRAITS[0], TRAITS[1], ...TRAITS.slice(3)]), ladderRole())).toBe(100);
    expect(fitScore(ranksOf([...TRAITS.slice(0, 9), TRAITS[11], TRAITS[9], TRAITS[10]]), ladderRole())).toBe(100);
  });

  it('stays within 0-100 and is finite for any ranking', () => {
    for (const order of [TRAITS, [...TRAITS].reverse(), [...TRAITS].sort(() => 0.5 - Math.random())]) {
      const s = fitScore(ranksOf(order), ladderRole());
      expect(Number.isFinite(s)).toBe(true);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(100);
    }
  });
});

describe('alignment: top-3 aligned and misaligned strengths from the candidate\'s top 6', () => {
  it('lists as aligned the candidate\'s top-6 strengths that sit in the same category as the role, most important to the role first, at most 3', () => {
    const { aligned } = alignment(sameAsRole, ladderRole());
    expect(aligned).toEqual([TRAITS[0], TRAITS[1], TRAITS[2]]); // all six qualify; the three most important to the role
  });

  it('lists none as misaligned when the candidate matches the role', () => {
    expect(alignment(sameAsRole, ladderRole()).misaligned).toEqual([]);
  });

  it('only looks at the candidate\'s top 6: a strength outside it is never listed', () => {
    const { aligned, misaligned } = alignment(oppositeOfRole, ladderRole());
    const top6 = [...TRAITS].reverse().slice(0, 6);
    for (const t of [...aligned, ...misaligned]) expect(top6).toContain(t);
  });

  it('lists misaligned strengths by weighted gap, largest first: a role-critical strength the candidate ranks lower outweighs a peripheral one they over-rate', () => {
    // Candidate's top 6: three peripheral strengths (ranks 1-3, three steps from the role) and the role's three
    // key strengths (ranks 4-6, only one step from the role's Signature) - but the key ones weigh far more.
    const order = [TRAITS[11], TRAITS[10], TRAITS[9], TRAITS[0], TRAITS[1], TRAITS[2], TRAITS[3], TRAITS[4], TRAITS[5], TRAITS[6], TRAITS[7], TRAITS[8]];
    const { misaligned } = alignment(ranksOf(order), ladderRole());
    expect(misaligned).toEqual([TRAITS[0], TRAITS[1], TRAITS[2]]);
  });

  it('reaches the peripheral over-rated strengths once the role-critical ones are used up', () => {
    // Only one role-critical strength in the candidate's top six; the rest are peripheral over-rated ones.
    const order = [TRAITS[11], TRAITS[10], TRAITS[9], TRAITS[8], TRAITS[0], TRAITS[7], TRAITS[1], TRAITS[2], TRAITS[3], TRAITS[4], TRAITS[5], TRAITS[6]];
    const { misaligned } = alignment(ranksOf(order), ladderRole());
    expect(misaligned).toHaveLength(3);
    expect(misaligned[0]).toBe(TRAITS[0]); // biggest gap: weight 1, one step
    const gaps = misaligned.map((t) => weightedGap(ranksOf(order)[t], ladderRole().find((r) => r.trait === t)!.rank));
    expect(gaps).toEqual([...gaps].sort((x, y) => y - x));
  });

  it('never lists more than 3 in either list, even when more qualify', () => {
    const { aligned, misaligned } = alignment(oppositeOfRole, ladderRole());
    expect(aligned.length).toBeLessThanOrEqual(3);
    expect(misaligned.length).toBeLessThanOrEqual(3);
  });

  it('treats a candidate with every strength tied for first as strongest where the role matters most: aligned are the role\'s Signature strengths, misaligned its next ones', () => {
    const tied = Object.fromEntries(TRAITS.map((t) => [t, 1])); // all tied: top 6 by the role's importance order
    const { aligned, misaligned } = alignment(tied, ladderRole());
    expect(aligned).toEqual([TRAITS[0], TRAITS[1], TRAITS[2]]);
    expect(misaligned).toEqual([TRAITS[3], TRAITS[4], TRAITS[5]]); // one step off, biggest weight first
  });
});

describe('rankCandidates: the Role-Fit Analysis table', () => {
  const cand = (name: string, order: readonly string[]): Candidate => ({ id: name, name, ranks: ranksOf(order) });

  it('ranks candidates best fit first, numbering them 1..n', () => {
    const rated = rankCandidates([cand('Far', [...TRAITS].reverse()), cand('Perfect', TRAITS)], ladderRole());
    expect(rated.map((c) => [c.rank, c.name])).toEqual([[1, 'Perfect'], [2, 'Far']]);
  });

  it('gives stars in equal groups of the candidates: the best fifth 5 stars, the last fifth 1 star', () => {
    const many = Array.from({ length: 10 }, (_, i) => {
      const order = [...TRAITS];
      // progressively worse: move the role's top strength down by i places
      order.splice(i, 0, order.splice(0, 1)[0]);
      return cand(`C${String(i).padStart(2, '0')}`, order);
    });
    const rated = rankCandidates(many, ladderRole());
    const stars = rated.map((c) => c.stars);
    expect(stars).toEqual([...stars].sort((a, b) => b - a));
    expect(stars[0]).toBe(5);
    expect(stars[9]).toBe(1);
  });

  it('gives candidates with identical fit the same rank order by name and the same stars, so a tie is never split', () => {
    const rated = rankCandidates([cand('Zed', TRAITS), cand('Amy', TRAITS), cand('Bob', [...TRAITS].reverse())], ladderRole());
    expect(rated.map((c) => c.name)).toEqual(['Amy', 'Zed', 'Bob']);
    expect(rated[0].stars).toBe(rated[1].stars);
  });

  it('shows no percentage or raw score to the template', () => {
    const [first] = rankCandidates([cand('A', TRAITS)], ladderRole());
    expect(first).not.toHaveProperty('score');
  });

  it('attaches each candidate\'s aligned and misaligned strengths', () => {
    const [first] = rankCandidates([cand('A', [...TRAITS].reverse())], ladderRole());
    expect(first.aligned.length + first.misaligned.length).toBeGreaterThan(0);
  });

  it('handles a single candidate and an empty list', () => {
    expect(rankCandidates([cand('Solo', TRAITS)], ladderRole())).toHaveLength(1);
    expect(rankCandidates([], ladderRole())).toEqual([]);
  });
});

describe('targetProfile ties and alignment ordering', () => {
  it('breaks a tie between two equally demanded strengths by the fixed SSM order, whatever else is ranked around them', () => {
    const profile = targetProfile(occupation({ [TRAITS[5]]: 50, [TRAITS[2]]: 50, [TRAITS[9]]: 80 }));
    expect(profile.slice(0, 3).map((r) => r.trait)).toEqual([TRAITS[9], TRAITS[2], TRAITS[5]]);
  });
});

describe('alignment: which strengths are shown, and in what order', () => {
  it('lists aligned strengths most important to the role first, not in the candidate\'s own order', () => {
    const profile = ladderRole();
    const candidate = ranksOf([TRAITS[2], TRAITS[1], TRAITS[0], ...TRAITS.slice(3)]);
    expect(alignment(candidate, profile).aligned).toEqual([TRAITS[0], TRAITS[1], TRAITS[2]]);
  });

  it('never lists a top-six strength that sits in the wrong category as aligned, and orders the misaligned by weighted gap, largest first', () => {
    const profile = ladderRole();
    // Top six: the role\'s three Signature strengths (aligned), then three Situational ones the candidate rates Supporting.
    const candidate = ranksOf([TRAITS[0], TRAITS[1], TRAITS[2], TRAITS[11], TRAITS[10], TRAITS[9], ...TRAITS.slice(3, 9)]);
    const { aligned, misaligned } = alignment(candidate, profile);
    expect(aligned).toEqual([TRAITS[0], TRAITS[1], TRAITS[2]]);
    expect(misaligned).toEqual([TRAITS[9], TRAITS[10], TRAITS[11]]); // role rank 10 outweighs 11 outweighs 12
  });

  it('when the candidate\'s ranks are all tied, picks the top six by importance to the role, not by the SSM order', () => {
    const upsideDown = targetProfile(occupation(Object.fromEntries(TRAITS.map((t, i) => [t, i * 5])))); // role needs TRAITS[11] most
    const tied = Object.fromEntries(TRAITS.map((t) => [t, 6.5]));
    const { aligned, misaligned } = alignment(tied, upsideDown);
    expect(aligned).toEqual([]);
    expect(misaligned).toEqual([TRAITS[11], TRAITS[10], TRAITS[9]]); // the role's three most important, the biggest gaps
  });
});
