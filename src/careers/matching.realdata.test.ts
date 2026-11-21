import { describe, it, expect } from 'vitest';
import { TRAITS } from '../scoring/traits.js';
import { occupations } from './data.js';
import { scoreOccupation } from './matching.js';

/** Deterministic pseudo-random numbers so the test never flakes. */
function lcg(seed: number) {
  let s = seed;
  return () => (s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296;
}

/** The 25 best strength fits among ALL occupations (before the interest shortlist). */
const top25 = (ranks: Record<string, number>) =>
  occupations
    .map((o) => ({ code: o.code, score: scoreOccupation(ranks, o) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 25)
    .map((s) => s.code);

describe('career matching on the real O*NET data', () => {
  it('gives opposite strength rankings mostly different matches: the report must depend on the person', () => {
    const forward = Object.fromEntries(TRAITS.map((t, i) => [t, i + 1]));
    const reverse = Object.fromEntries(TRAITS.map((t, i) => [t, 12 - i]));
    const b = new Set(top25(reverse));
    const shared = top25(forward).filter((c) => b.has(c)).length;
    expect(shared).toBeLessThan(13);
  });

  it('does not put the same occupations at the top for almost everyone', () => {
    // Regression: comparing user levels with raw O*NET levels put the same occupations in the
    // top of nearly every user's list, whoever they were.
    const random = lcg(42);
    const users = 100;
    const counts = new Map<string, number>();
    for (let u = 0; u < users; u++) {
      const order = [...TRAITS].sort(() => random() - 0.5);
      const ranks = Object.fromEntries(order.map((t, i) => [t, i + 1]));
      for (const code of top25(ranks)) counts.set(code, (counts.get(code) ?? 0) + 1);
    }
    expect([...counts.values()].filter((c) => c >= users * 0.5).length).toBe(0);
  });
});
